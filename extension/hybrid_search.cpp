/*
 * hybrid_search.cpp
 *
 * SQLite loadable extension: "hybridsearch" virtual table module.
 *
 * FAISS search runs fully inside C++: on xCreate/xConnect the extension
 * loads the serialised IndexFlatL2 from the `faiss_index` table. xFilter
 * accepts a float32 embedding BLOB, calls index->search() with an optional
 * IDSelectorBatch when metadata filters are supplied, runs FTS5 on the
 * filtered row set, and fuses both ranked lists with RRF.
 *
 * Virtual table schema (2 visible, 7 hidden columns):
 *   SELECT chunk_id, score FROM hybrid_search
 *     WHERE query_embedding = ?   -- BLOB: float32 query vector (D floats)
 *     AND   query_text      = ?   -- TEXT: raw query for FTS5 (optional)
 *     AND   top_k           = ?   -- INTEGER: candidates to return (optional, default 50)
 *     AND   source          = ?   -- TEXT: filter by source path prefix (optional)
 *     AND   section         = ?   -- TEXT: filter by section prefix (optional)
 *     AND   page_start      = ?   -- INTEGER: filter page_start >= value (optional)
 *     AND   page_end        = ?   -- INTEGER: filter page_end   <= value (optional)
 *
 * idxNum bitmask for xBestIndex → xFilter:
 *   bit 0: source filter present
 *   bit 1: section filter present
 *   bit 2: page_start filter present
 *   bit 3: page_end filter present
 */

#include <sqlite3ext.h>
SQLITE_EXTENSION_INIT1

#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <faiss/impl/io.h>
#include <faiss/impl/IDSelector.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// constants
static const int DEFAULT_TOP_K = 50;
static const int RRF_K         = 60;

// Column indices (0-based)
enum Col {
    COL_CHUNK_ID        = 0,
    COL_SCORE           = 1,
    COL_QUERY_EMBEDDING = 2,  // hidden
    COL_QUERY_TEXT      = 3,  // hidden
    COL_TOP_K           = 4,  // hidden
    COL_SOURCE          = 5,  // hidden — metadata filter
    COL_SECTION         = 6,  // hidden — metadata filter
    COL_PAGE_START      = 7,  // hidden — metadata filter
    COL_PAGE_END        = 8,  // hidden — metadata filter
};

// idxNum bitmask bits
static const int FILT_SOURCE     = 1 << 0;
static const int FILT_SECTION    = 1 << 1;
static const int FILT_PAGE_START = 1 << 2;
static const int FILT_PAGE_END   = 1 << 3;

// virtual table structures
struct HybridVtab {
    sqlite3_vtab                  base;        // must be first
    sqlite3*                      db;
    std::unique_ptr<faiss::Index> faiss_idx;
    faiss::idx_t                  txn_ntotal{-1};  // ntotal snapshot at xBegin; -1 = no active txn
    bool                          dirty{false};    // true if in-memory index has unflushed adds
};

struct HybridCursor {
    sqlite3_vtab_cursor                           base;  // must be first
    std::vector<std::pair<sqlite3_int64, double>> results;
    int                                           pos;
};

// helpers
static double rrf_score(int rank) {
    return 1.0 / (RRF_K + rank);
}

// Load FAISS index from the faiss_index table in the same database.
static faiss::Index* load_faiss_index(sqlite3* db, char** pzErr) {
    const char* sql = "SELECT data FROM faiss_index LIMIT 1";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        if (pzErr)
            *pzErr = sqlite3_mprintf("hybrid_search: cannot query faiss_index: %s",
                                     sqlite3_errmsg(db));
        return nullptr;
    }

    faiss::Index* idx = nullptr;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        const void* blob   = sqlite3_column_blob(stmt, 0);
        int         nbytes = sqlite3_column_bytes(stmt, 0);
        if (blob && nbytes > 0) {
            faiss::VectorIOReader reader;
            reader.data.assign(
                static_cast<const uint8_t*>(blob),
                static_cast<const uint8_t*>(blob) + nbytes);
            try {
                idx = faiss::read_index(&reader);
            } catch (...) {
                if (pzErr)
                    *pzErr = sqlite3_mprintf(
                        "hybrid_search: failed to deserialize FAISS index");
            }
        }
    }
    sqlite3_finalize(stmt);

    if (!idx && pzErr && !*pzErr)
        *pzErr = sqlite3_mprintf("hybrid_search: faiss_index table is empty");
    return idx;
}

// xCreate / xConnect
static int hybridInit(sqlite3* db, void* /*pAux*/, int /*argc*/,
                      const char* const* /*argv*/, sqlite3_vtab** ppVtab,
                      char** pzErr) {
    int rc = sqlite3_declare_vtab(db,
        "CREATE TABLE x("
        "  chunk_id        INTEGER,"   // 0 visible
        "  score           REAL,"      // 1 visible
        "  query_embedding BLOB    HIDDEN,"  // 2
        "  query_text      TEXT    HIDDEN,"  // 3
        "  top_k           INTEGER HIDDEN,"  // 4
        "  source          TEXT    HIDDEN,"  // 5 metadata filter
        "  section         TEXT    HIDDEN,"  // 6 metadata filter
        "  page_start      INTEGER HIDDEN,"  // 7 metadata filter
        "  page_end        INTEGER HIDDEN"   // 8 metadata filter
        ")");
    if (rc != SQLITE_OK) return rc;

    faiss::Index* fidx = load_faiss_index(db, pzErr);
    if (!fidx) return SQLITE_ERROR;

    HybridVtab* vtab   = new HybridVtab();
    vtab->db           = db;
    vtab->faiss_idx.reset(fidx);
    *ppVtab            = &vtab->base;
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

// xBestIndex — advertise metadata filter columns to the query planner
static int hybridBestIndex(sqlite3_vtab* /*pVtab*/, sqlite3_index_info* pInfo) {
    int argv_idx = 1;
    int idx_num  = 0;

    // Pass 1: assign argvIndex to core input columns
    for (int i = 0; i < pInfo->nConstraint; i++) {
        if (!pInfo->aConstraint[i].usable) continue;
        int col = pInfo->aConstraint[i].iColumn;
        int op  = pInfo->aConstraint[i].op;
        if (op == SQLITE_INDEX_CONSTRAINT_EQ &&
            (col == COL_QUERY_EMBEDDING || col == COL_QUERY_TEXT || col == COL_TOP_K)) {
            pInfo->aConstraintUsage[i].argvIndex = argv_idx++;
            pInfo->aConstraintUsage[i].omit      = 1;
        }
    }

    // Pass 2: detect which metadata filters are present
    for (int i = 0; i < pInfo->nConstraint; i++) {
        if (!pInfo->aConstraint[i].usable) continue;
        int col = pInfo->aConstraint[i].iColumn;
        int op  = pInfo->aConstraint[i].op;
        if (op == SQLITE_INDEX_CONSTRAINT_EQ && col == COL_SOURCE)
            idx_num |= FILT_SOURCE;
        else if (op == SQLITE_INDEX_CONSTRAINT_EQ && col == COL_SECTION)
            idx_num |= FILT_SECTION;
        else if ((op == SQLITE_INDEX_CONSTRAINT_EQ || op == SQLITE_INDEX_CONSTRAINT_GE)
                 && col == COL_PAGE_START)
            idx_num |= FILT_PAGE_START;
        else if ((op == SQLITE_INDEX_CONSTRAINT_EQ || op == SQLITE_INDEX_CONSTRAINT_LE)
                 && col == COL_PAGE_END)
            idx_num |= FILT_PAGE_END;
    }

    // Pass 3: assign argvIndex to meta filters in fixed order so xFilter can
    // read them positionally: source, section, page_start, page_end.
    struct { int bit; int col; } meta_order[] = {
        {FILT_SOURCE,     COL_SOURCE},
        {FILT_SECTION,    COL_SECTION},
        {FILT_PAGE_START, COL_PAGE_START},
        {FILT_PAGE_END,   COL_PAGE_END},
    };
    for (auto& m : meta_order) {
        if (!(idx_num & m.bit)) continue;
        for (int i = 0; i < pInfo->nConstraint; i++) {
            if (!pInfo->aConstraint[i].usable) continue;
            if (pInfo->aConstraint[i].iColumn != m.col) continue;
            pInfo->aConstraintUsage[i].argvIndex = argv_idx++;
            pInfo->aConstraintUsage[i].omit      = 1;
            break;
        }
    }

    pInfo->idxNum = idx_num;
    pInfo->estimatedCost = (idx_num != 0) ? 10.0 : 1000.0;
    if (idx_num != 0) pInfo->estimatedRows = 100;
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

// Build the set of chunk_ids that satisfy the metadata filters.
// Returns false if no filters were requested (caller should use full index).
static bool build_filter_set(sqlite3* db, int idx_num,
                              const char* source_val, const char* section_val,
                              int page_start_val, int page_end_val,
                              std::unordered_set<faiss::idx_t>& id_set) {
    if (idx_num == 0) return false;

    std::string sql = "SELECT chunk_id FROM chunks WHERE 1=1";
    std::vector<std::string> binds;

    if (idx_num & FILT_SOURCE) {
        sql += " AND source LIKE ?";
        binds.push_back(std::string(source_val ? source_val : "") + "%");
    }
    if (idx_num & FILT_SECTION) {
        sql += " AND section LIKE ?";
        binds.push_back(std::string(section_val ? section_val : "") + "%");
    }
    if (idx_num & FILT_PAGE_START) {
        sql += " AND page_start >= ?";
        binds.push_back(std::to_string(page_start_val));
    }
    if (idx_num & FILT_PAGE_END) {
        sql += " AND page_end <= ?";
        binds.push_back(std::to_string(page_end_val));
    }

    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK)
        return false;

    for (int b = 0; b < static_cast<int>(binds.size()); b++) {
        const std::string& v = binds[b];
        // Try binding as integer first if no alpha chars
        bool all_digits = !v.empty();
        for (char c : v) if (!isdigit(c)) { all_digits = false; break; }
        if (all_digits)
            sqlite3_bind_int(stmt, b + 1, std::stoi(v));
        else
            sqlite3_bind_text(stmt, b + 1, v.c_str(), -1, SQLITE_TRANSIENT);
    }

    while (sqlite3_step(stmt) == SQLITE_ROW)
        id_set.insert(static_cast<faiss::idx_t>(sqlite3_column_int64(stmt, 0)));
    sqlite3_finalize(stmt);
    return true;
}

// xFilter
static int hybridFilter(sqlite3_vtab_cursor* pCursor, int idxNum,
                        const char* /*idxStr*/, int argc,
                        sqlite3_value** argv) {
    HybridCursor* cur  = reinterpret_cast<HybridCursor*>(pCursor);
    HybridVtab*   vtab = reinterpret_cast<HybridVtab*>(pCursor->pVtab);
    cur->results.clear();
    cur->pos = 0;

    // parse hidden-column arguments (order follows argvIndex assignment)
    const void* emb_blob   = nullptr;
    int         emb_bytes  = 0;
    const char* query_text = nullptr;
    int         top_k      = DEFAULT_TOP_K;
    const char* source_val  = nullptr;
    const char* section_val = nullptr;
    int         page_start_val = 0;
    int         page_end_val   = 0;

    // argv ordering mirrors argvIndex assignment in xBestIndex:
    // core cols first (embedding, text, top_k), then meta filters.
    // We detect type because order may vary by which constraints are present.
    int meta_arg_start = 0;
    for (int i = 0; i < argc; i++) {
        int vtype = sqlite3_value_type(argv[i]);
        if (vtype == SQLITE_BLOB && !emb_blob) {
            emb_blob  = sqlite3_value_blob(argv[i]);
            emb_bytes = sqlite3_value_bytes(argv[i]);
            meta_arg_start = i + 1;
        } else if (vtype == SQLITE_TEXT && !query_text && i < 3) {
            query_text = reinterpret_cast<const char*>(sqlite3_value_text(argv[i]));
            meta_arg_start = i + 1;
        } else if (vtype == SQLITE_INTEGER && top_k == DEFAULT_TOP_K && i < 3) {
            top_k = sqlite3_value_int(argv[i]);
            meta_arg_start = i + 1;
        }
    }

    // Parse meta filter args that follow the core args
    int meta_i = meta_arg_start;
    if ((idxNum & FILT_SOURCE) && meta_i < argc)
        source_val = reinterpret_cast<const char*>(sqlite3_value_text(argv[meta_i++]));
    if ((idxNum & FILT_SECTION) && meta_i < argc)
        section_val = reinterpret_cast<const char*>(sqlite3_value_text(argv[meta_i++]));
    if ((idxNum & FILT_PAGE_START) && meta_i < argc)
        page_start_val = sqlite3_value_int(argv[meta_i++]);
    if ((idxNum & FILT_PAGE_END) && meta_i < argc)
        page_end_val = sqlite3_value_int(argv[meta_i++]);

    // Build metadata filter set (empty set means no filtering)
    std::unordered_set<faiss::idx_t> filter_ids;
    bool has_filter = build_filter_set(vtab->db, idxNum,
                                       source_val, section_val,
                                       page_start_val, page_end_val,
                                       filter_ids);

    // FAISS search using the in-memory index
    std::unordered_map<sqlite3_int64, int> faiss_ranks;
    if (emb_blob && emb_bytes > 0 && vtab->faiss_idx) {
        int dim = vtab->faiss_idx->d;
        int n_floats = emb_bytes / sizeof(float);
        if (n_floats == dim) {
            faiss::idx_t k = std::min(top_k, static_cast<int>(vtab->faiss_idx->ntotal));
            if (k > 0) {
                std::vector<faiss::idx_t> labels(k, -1);
                std::vector<float>       distances(k, 0.0f);
                const float* q = static_cast<const float*>(emb_blob);

                if (has_filter && !filter_ids.empty()) {
                    // Build IDSelectorBatch to restrict FAISS search
                    std::vector<faiss::idx_t> id_vec(filter_ids.begin(), filter_ids.end());
                    faiss::IDSelectorBatch selector(id_vec.size(), id_vec.data());
                    faiss::SearchParameters params;
                    params.sel = &selector;
                    vtab->faiss_idx->search(1, q, k, distances.data(), labels.data(), &params);
                } else if (has_filter && filter_ids.empty()) {
                    // Filter matched nothing — skip FAISS search
                } else {
                    vtab->faiss_idx->search(1, q, k, distances.data(), labels.data());
                }

                for (int r = 0; r < k; r++) {
                    if (labels[r] >= 0)
                        faiss_ranks[static_cast<sqlite3_int64>(labels[r])] = r + 1;
                }
            }
        }
    }

    // FTS5 search (restrict to filtered chunk_ids via rowid IN subquery if filtered)
    std::unordered_map<sqlite3_int64, int> fts5_ranks;
    if (query_text && query_text[0] != '\0') {
        std::string fts_sql;
        if (has_filter && !filter_ids.empty()) {
            // Build comma-separated list of allowed rowids
            std::string id_list;
            bool first = true;
            for (auto id : filter_ids) {
                if (!first) id_list += ',';
                id_list += std::to_string(id);
                first = false;
            }
            fts_sql = "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
                      "AND rowid IN (" + id_list + ") ORDER BY rank LIMIT ?";
        } else if (has_filter && filter_ids.empty()) {
            // Filter matched nothing — skip FTS5
            fts_sql = "";
        } else {
            fts_sql = "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
                      "ORDER BY rank LIMIT ?";
        }

        if (!fts_sql.empty()) {
            sqlite3_stmt* stmt = nullptr;
            if (sqlite3_prepare_v2(vtab->db, fts_sql.c_str(), -1, &stmt, nullptr) == SQLITE_OK) {
                sqlite3_bind_text(stmt, 1, query_text, -1, SQLITE_STATIC);
                sqlite3_bind_int(stmt, 2, top_k);
                int rank = 1;
                while (sqlite3_step(stmt) == SQLITE_ROW) {
                    fts5_ranks[sqlite3_column_int64(stmt, 0)] = rank++;
                }
            }
            sqlite3_finalize(stmt);
        }
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
    if (col == COL_CHUNK_ID)
        sqlite3_result_int64(ctx, id);
    else if (col == COL_SCORE)
        sqlite3_result_double(ctx, score);
    return SQLITE_OK;
}

static int hybridRowid(sqlite3_vtab_cursor* pCursor, sqlite3_int64* pRowid) {
    *pRowid = static_cast<sqlite3_int64>(
        reinterpret_cast<HybridCursor*>(pCursor)->pos);
    return SQLITE_OK;
}

// xUpdate — INSERT a new chunk atomically (updates chunks, chunks_fts, and faiss_index BLOB)
//
// Virtual table has 9 columns (indices 0..8):
//   0 chunk_id, 1 score, 2 query_embedding, 3 query_text, 4 top_k,
//   5 source, 6 section, 7 page_start, 8 page_end
//
// argv layout for INSERT (argc == 11 = 2 + 9):
//   argv[0]  = old rowid (NULL for INSERT)
//   argv[1]  = new rowid
//   argv[2]  = chunk_id        — ignored (auto-assigned by chunks table)
//   argv[3]  = score           — ignored
//   argv[4]  = query_embedding — repurposed: float32 embedding of new chunk
//   argv[5]  = query_text      — repurposed: text content of new chunk
//   argv[6]  = top_k           — ignored
//   argv[7]  = source          — chunk source path
//   argv[8]  = section         — chunk section name
//   argv[9]  = page_start      — chunk page start
//   argv[10] = page_end        — chunk page end
//
// Example usage:
//   INSERT INTO hybrid_search(query_text, query_embedding, source, section, page_start, page_end)
//   VALUES('text...', X'...float32 bytes...', 'source.pdf', 'Ch 1', 1, 5)
static int hybridUpdate(sqlite3_vtab* pVtab, int argc, sqlite3_value** argv,
                        sqlite3_int64* pRowid) {
    if (argc <= 1) return SQLITE_OK;  // DELETE — not implemented
    if (sqlite3_value_type(argv[1]) != SQLITE_NULL) return SQLITE_OK; // UPDATE — skip

    HybridVtab* vtab = reinterpret_cast<HybridVtab*>(pVtab);
    sqlite3*    db   = vtab->db;

    // Need argv[0..10] for our 9-column virtual table
    if (argc < 11) return SQLITE_OK;

    const char* text_val  = reinterpret_cast<const char*>(sqlite3_value_text(argv[5]));
    const void* emb_blob  = sqlite3_value_blob(argv[4]);
    int         emb_bytes = sqlite3_value_bytes(argv[4]);

    if (!text_val || !emb_blob) return SQLITE_OK;

    const char* source_val  = reinterpret_cast<const char*>(sqlite3_value_text(argv[7]));
    const char* section_val = reinterpret_cast<const char*>(sqlite3_value_text(argv[8]));
    int page_start = (sqlite3_value_type(argv[9])  != SQLITE_NULL) ? sqlite3_value_int(argv[9])  : 0;
    int page_end   = (sqlite3_value_type(argv[10]) != SQLITE_NULL) ? sqlite3_value_int(argv[10]) : 0;

    // SQLite already wraps the xUpdate call in its own transaction/savepoint
    // (WAL journal mode), so we do NOT start a nested SAVEPOINT here.
    // All three writes (chunks, chunks_fts, faiss_index) will be committed
    // or rolled back atomically by the caller's transaction.

    // 1. INSERT into chunks table (chunk_id auto-assigned = ntotal, matching FAISS add())
    const char* ins_sql =
        "INSERT INTO chunks(text, source, section, page_start, page_end) "
        "VALUES(?, ?, ?, ?, ?)";
    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db, ins_sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) return rc;
    sqlite3_bind_text(stmt, 1, text_val,    -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, source_val,  -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, section_val, -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt,  4, page_start);
    sqlite3_bind_int(stmt,  5, page_end);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE) return SQLITE_ERROR;

    sqlite3_int64 new_chunk_id = sqlite3_last_insert_rowid(db);
    if (pRowid) *pRowid = new_chunk_id;

    // 2. INSERT into chunks_fts (external content table — update the FTS index)
    // For FTS5 content tables, insert the text with its content rowid.
    const char* fts_sql = "INSERT INTO chunks_fts(rowid, text) VALUES(?, ?)";
    rc = sqlite3_prepare_v2(db, fts_sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) return rc;
    sqlite3_bind_int64(stmt, 1, new_chunk_id);
    sqlite3_bind_text(stmt,  2, text_val, -1, SQLITE_STATIC);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE) return SQLITE_ERROR;

    // 3. Add vector to in-memory FAISS index; BLOB is flushed once in xSync.
    if (vtab->faiss_idx &&
        emb_bytes == vtab->faiss_idx->d * static_cast<int>(sizeof(float))) {
        const float* vec = static_cast<const float*>(emb_blob);
        vtab->faiss_idx->add(1, vec);
        vtab->dirty = true;
    }

    return SQLITE_OK;
}

// xBegin / xCommit / xRollback
static int hybridBegin(sqlite3_vtab* pVtab) {
    HybridVtab* vtab   = reinterpret_cast<HybridVtab*>(pVtab);
    vtab->txn_ntotal   = vtab->faiss_idx ? vtab->faiss_idx->ntotal : 0;
    return SQLITE_OK;
}

static int hybridCommit(sqlite3_vtab* pVtab) {
    HybridVtab* vtab = reinterpret_cast<HybridVtab*>(pVtab);
    vtab->txn_ntotal = -1;
    vtab->dirty      = false;
    return SQLITE_OK;
}

// Called once before the transaction commits — serialize the index here so
// N inserts in one transaction cost one BLOB write instead of N.
static int hybridSync(sqlite3_vtab* pVtab) {
    HybridVtab* vtab = reinterpret_cast<HybridVtab*>(pVtab);
    if (!vtab->dirty || !vtab->faiss_idx) return SQLITE_OK;

    faiss::VectorIOWriter writer;
    try {
        faiss::write_index(vtab->faiss_idx.get(), &writer);
    } catch (...) {
        return SQLITE_ERROR;
    }

    const char* upd_sql = "UPDATE faiss_index SET data=? WHERE id=1";
    sqlite3_stmt* stmt  = nullptr;
    int rc = sqlite3_prepare_v2(vtab->db, upd_sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) return rc;
    sqlite3_bind_blob(stmt, 1, writer.data.data(),
                      static_cast<int>(writer.data.size()), SQLITE_TRANSIENT);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE) return SQLITE_ERROR;

    vtab->dirty = false;
    return SQLITE_OK;
}

// SQLite has already rolled back the DB before calling this.
// Truncate the in-memory codes array to its pre-transaction length — O(1), no I/O.
// This is better compared to resetting complete faiss index from database.
static int hybridRollback(sqlite3_vtab* pVtab) {
    HybridVtab* vtab = reinterpret_cast<HybridVtab*>(pVtab);
    if (vtab->txn_ntotal >= 0 && vtab->faiss_idx) {
        auto* flat = dynamic_cast<faiss::IndexFlatCodes*>(vtab->faiss_idx.get());
        if (flat) {
            flat->codes.resize(static_cast<size_t>(vtab->txn_ntotal) *
                               flat->d * sizeof(float));
            flat->ntotal = vtab->txn_ntotal;
        }
    }
    vtab->txn_ntotal = -1;
    vtab->dirty      = false;
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
    hybridUpdate,     // xUpdate
    hybridBegin,      // xBegin
    hybridSync,       // xSync
    hybridCommit,     // xCommit
    hybridRollback,   // xRollback
    nullptr,          // xFindMethod
    nullptr,          // xRename
};

extern "C" {
#ifdef _WIN32
__declspec(dllexport)
#endif
int sqlite3_hybrid_search_init(sqlite3* db, char** pzErrMsg,
                               const sqlite3_api_routines* pApi) {
    SQLITE_EXTENSION_INIT2(pApi);
    return sqlite3_create_module(db, "hybridsearch", &hybridSearchModule,
                                 nullptr);
}
} // extern "C"
