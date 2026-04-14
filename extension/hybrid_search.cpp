/*
 * hybrid_search.cpp
 *
 * SQLite loadable extension: "hybridsearch" virtual table module.
 *
 * FAISS search is performed in Python (which already has the index loaded).
 * This extension handles FTS5 keyword search and RRF fusion in C++.
 *
 * Virtual table (two visible, three hidden columns):
 *   CREATE VIRTUAL TABLE hybrid_search USING hybridsearch();
 *   SELECT chunk_id, score FROM hybrid_search
 *     WHERE faiss_results = ?   -- BLOB: packed int64 chunk_ids in FAISS rank order
 *     AND   query_text   = ?   -- TEXT: raw query for FTS5 (optional)
 *     AND   top_k        = ?   -- INTEGER: candidates to return (optional, default 50)
 *
 * BLOB format for faiss_results:
 *   Sequence of little-endian int64 values: chunk_id at rank 1, rank 2, ...
 *   The rank of chunk_id[i] is (i + 1).
 *
 * xFilter logic:
 *   1. Unpack FAISS ranks from the BLOB.
 *   2. Run FTS5 MATCH on `chunks_fts` to get keyword ranks.
 *   3. Fuse both ranked lists with RRF (equal weights when both present).
 *   4. Return up to top_k rows sorted by fused score descending.
 */

#include <sqlite3ext.h>
SQLITE_EXTENSION_INIT1

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

// constants
static const int DEFAULT_TOP_K = 50;
static const int RRF_K         = 60;

// virtual table structures
struct HybridVtab {
    sqlite3_vtab base;  // must be first
    sqlite3*     db;
};

struct HybridCursor {
    sqlite3_vtab_cursor                          base;  // must be first
    std::vector<std::pair<sqlite3_int64, double>> results;
    int                                          pos;
};

// helpers
static double rrf_score(int rank) {
    return 1.0 / (RRF_K + rank);
}

// xCreate / xConnect
static int hybridInit(sqlite3* db, void* /*pAux*/, int /*argc*/,
                      const char* const* /*argv*/, sqlite3_vtab** ppVtab,
                      char** /*pzErr*/) {
    // columns:
    //   0  chunk_id      INTEGER  (visible, output)
    //   1  score         REAL     (visible, output)
    //   2  faiss_results BLOB     (hidden, input - FAISS chunk_ids in rank order)
    //   3  query_text    TEXT     (hidden, input - for FTS5, optional)
    //   4  top_k         INTEGER  (hidden, input - optional)
    int rc = sqlite3_declare_vtab(db,
        "CREATE TABLE x("
        "  chunk_id      INTEGER,"
        "  score         REAL,"
        "  faiss_results BLOB    HIDDEN,"
        "  query_text    TEXT    HIDDEN,"
        "  top_k         INTEGER HIDDEN"
        ")");
    if (rc != SQLITE_OK) return rc;

    HybridVtab* vtab = new HybridVtab();
    vtab->db         = db;
    *ppVtab          = &vtab->base;
    return SQLITE_OK;
}

static int hybridCreate(sqlite3* db, void* pAux, int argc,
                        const char* const* argv, sqlite3_vtab** ppVtab,
                        char** pzErr) {
    return hybridInit(db, pAux, argc, argv, ppVtab, pzErr);
}

static int hybridConnect(sqlite3* db, void* pAux, int argc,
                         const char* const* argv, sqlite3_vtab** ppVtab,
                         char** pzErr) {
    return hybridInit(db, pAux, argc, argv, ppVtab, pzErr);
}

static int hybridDisconnect(sqlite3_vtab* pVtab) {
    delete reinterpret_cast<HybridVtab*>(pVtab);
    return SQLITE_OK;
}

static int hybridDestroy(sqlite3_vtab* pVtab) {
    return hybridDisconnect(pVtab);
}

// xBestIndex
static int hybridBestIndex(sqlite3_vtab* /*pVtab*/, sqlite3_index_info* pInfo) {
    int argv_idx = 1;
    for (int i = 0; i < pInfo->nConstraint; i++) {
        if (!pInfo->aConstraint[i].usable) continue;
        int col = pInfo->aConstraint[i].iColumn;
        int op  = pInfo->aConstraint[i].op;
        if (op == SQLITE_INDEX_CONSTRAINT_EQ && (col == 2 || col == 3 || col == 4)) {
            pInfo->aConstraintUsage[i].argvIndex = argv_idx++;
            pInfo->aConstraintUsage[i].omit      = 1;
        }
    }
    pInfo->estimatedCost = 1000.0;
    return SQLITE_OK;
}

// xOpen / xClose
static int hybridOpen(sqlite3_vtab* /*pVtab*/, sqlite3_vtab_cursor** ppCursor) {
    HybridCursor* cur = new HybridCursor();
    cur->pos          = 0;
    *ppCursor         = &cur->base;
    return SQLITE_OK;
}

static int hybridClose(sqlite3_vtab_cursor* pCursor) {
    delete reinterpret_cast<HybridCursor*>(pCursor);
    return SQLITE_OK;
}

// xFilter
static int hybridFilter(sqlite3_vtab_cursor* pCursor, int /*idxNum*/,
                        const char* /*idxStr*/, int argc,
                        sqlite3_value** argv) {
    HybridCursor* cur  = reinterpret_cast<HybridCursor*>(pCursor);
    HybridVtab*   vtab = reinterpret_cast<HybridVtab*>(pCursor->pVtab);
    cur->results.clear();
    cur->pos = 0;

    // parse hidden-column arguments
    const void* faiss_blob  = nullptr;
    int         faiss_bytes = 0;
    const char* query_text  = nullptr;
    int         top_k       = DEFAULT_TOP_K;

    for (int i = 0; i < argc; i++) {
        int vtype = sqlite3_value_type(argv[i]);
        if (vtype == SQLITE_BLOB && !faiss_blob) {
            faiss_blob  = sqlite3_value_blob(argv[i]);
            faiss_bytes = sqlite3_value_bytes(argv[i]);
        } else if (vtype == SQLITE_TEXT && !query_text) {
            query_text = reinterpret_cast<const char*>(sqlite3_value_text(argv[i]));
        } else if (vtype == SQLITE_INTEGER) {
            top_k = sqlite3_value_int(argv[i]);
        }
    }

    // unpack FAISS ranks from BLOB
    // layout: sequence of little-endian int64 chunk_ids, rank = index+1
    std::unordered_map<sqlite3_int64, int> faiss_ranks;
    if (faiss_blob && faiss_bytes >= 8) {
        int n = faiss_bytes / 8;
        const uint8_t* p = static_cast<const uint8_t*>(faiss_blob);
        for (int r = 0; r < n; r++, p += 8) {
            int64_t id;
            memcpy(&id, p, 8);
            if (id >= 0) faiss_ranks[static_cast<sqlite3_int64>(id)] = r + 1;
        }
    }

    // FTS5 search
    std::unordered_map<sqlite3_int64, int> fts5_ranks;
    if (query_text && query_text[0] != '\0') {
        const char* sql =
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY rank LIMIT ?";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(vtab->db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, query_text, -1, SQLITE_STATIC);
            sqlite3_bind_int(stmt, 2, top_k);
            int rank = 1;
            while (sqlite3_step(stmt) == SQLITE_ROW) {
                fts5_ranks[sqlite3_column_int64(stmt, 0)] = rank++;
            }
        }
        sqlite3_finalize(stmt);
    }

    // RRF fusion
    bool   have_faiss = !faiss_ranks.empty();
    bool   have_fts5  = !fts5_ranks.empty();
    double w_faiss    = have_fts5  ? 0.5 : 1.0;
    double w_fts5     = have_faiss ? 0.5 : 1.0;

    std::unordered_map<sqlite3_int64, double> fused;
    for (auto& [id, rank] : faiss_ranks)
        fused[id] += w_faiss * rrf_score(rank);
    for (auto& [id, rank] : fts5_ranks)
        fused[id] += w_fts5 * rrf_score(rank);

    cur->results.reserve(fused.size());
    for (auto& [id, score] : fused)
        cur->results.push_back({id, score});

    std::sort(cur->results.begin(), cur->results.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    if (static_cast<int>(cur->results.size()) > top_k)
        cur->results.resize(static_cast<size_t>(top_k));

    return SQLITE_OK;
}

// xNext / xEof / xColumn / xRowid
static int hybridNext(sqlite3_vtab_cursor* pCursor) {
    reinterpret_cast<HybridCursor*>(pCursor)->pos++;
    return SQLITE_OK;
}

static int hybridEof(sqlite3_vtab_cursor* pCursor) {
    HybridCursor* cur = reinterpret_cast<HybridCursor*>(pCursor);
    return cur->pos >= static_cast<int>(cur->results.size()) ? 1 : 0;
}

static int hybridColumn(sqlite3_vtab_cursor* pCursor, sqlite3_context* ctx,
                        int col) {
    HybridCursor* cur = reinterpret_cast<HybridCursor*>(pCursor);
    if (cur->pos < 0 || cur->pos >= static_cast<int>(cur->results.size()))
        return SQLITE_OK;
    auto& [id, score] = cur->results[static_cast<size_t>(cur->pos)];
    if (col == 0)
        sqlite3_result_int64(ctx, id);
    else if (col == 1)
        sqlite3_result_double(ctx, score);
    return SQLITE_OK;
}

static int hybridRowid(sqlite3_vtab_cursor* pCursor, sqlite3_int64* pRowid) {
    *pRowid = static_cast<sqlite3_int64>(
        reinterpret_cast<HybridCursor*>(pCursor)->pos);
    return SQLITE_OK;
}

// module registration
static sqlite3_module hybridSearchModule = {
    0,                // iVersion
    hybridCreate,     // xCreate
    hybridConnect,    // xConnect
    hybridBestIndex,  // xBestIndex
    hybridDisconnect, // xDisconnect
    hybridDestroy,    // xDestroy
    hybridOpen,       // xOpen
    hybridClose,      // xClose
    hybridFilter,     // xFilter
    hybridNext,       // xNext
    hybridEof,        // xEof
    hybridColumn,     // xColumn
    hybridRowid,      // xRowid
    nullptr,          // xUpdate
    nullptr,          // xBegin
    nullptr,          // xSync
    nullptr,          // xCommit
    nullptr,          // xRollback
    nullptr,          // xFindMethod
    nullptr,          // xRename
};

extern "C" {
#ifdef _WIN32
__declspec(dllexport)
#endif
int sqlite3_hybrid_search_init(sqlite3* db, char** /*pzErrMsg*/,
                               const sqlite3_api_routines* pApi) {
    SQLITE_EXTENSION_INIT2(pApi);
    return sqlite3_create_module(db, "hybridsearch", &hybridSearchModule,
                                 nullptr);
}
} // extern "C"
