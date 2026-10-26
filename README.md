# RC Brain retrieval

Hybrid search over the RC Brain `KnowledgeDocument` library. Base44 calls `POST /retrieve`. The service runs a Postgres full-text leg and an exact pgvector leg, fuses them, reranks with Voyage, and returns passages.

Railway auto-deploys this repo from `main`. The running service is the web process defined in `railway.toml` (`uvicorn api:app`, healthcheck `GET /health`).

## What broke the indexer

The last successful index was 2026-06-25 (1,267 documents, 8,284 chunks). Later indexer deploys failed for two stacked reasons:

1. **The Base44 call cannot succeed anymore.** `indexer.py` requested `https://api.base44.com/.../KnowledgeDocument?limit=&skip=`. That host now returns an HTML 404. The live API is `https://app.base44.com/api/apps/{app_id}/entities/KnowledgeDocument/v2/list`, which pages with a cursor. Sending `skip` is an HTTP 400. `raise_for_status()` was not caught, so the process exited and Railway marked the deployment failed.
2. **A one-shot indexer cannot satisfy this service's healthcheck.** `railway.toml` probes `GET /health`. A service whose start command is `python indexer.py` never serves that route, so Railway fails the deploy even when the script itself is healthy. Indexing now runs inside the web process, which already serves `/health`.

Large documents (up to about 100k characters) made the old page of 100 full bodies likely to time out as well. The new walk lists metadata only, then downloads a body when that document actually needs work. Voyage calls are batched under the per-request token cap and retried on 429 and 5xx. Base44 calls are paced (default 60/minute, under the list endpoint's base limit of 70/minute) and retried the same way.

## What a reindex does

It does **not** drop `knowledge_chunks` or `indexed_documents`.

- Pages through every `KnowledgeDocument` with the cursor API.
- Skips a document when its stored `updated_date` matches Base44, its embed version is current, and it already has chunks.
- Otherwise downloads the body and MD5s it the same way the June index did. It re-embeds when the hash changed, the stored embed version is missing or older, the title changed, or `--force` / `{"force": true}` is set. A matching hash with a current embed version and the same title still refreshes chunk metadata (`source_date`, category, and the rest) without calling Voyage.
- Deletes chunks and the registry row for documents Base44 no longer returns.
- Refuses to delete anything if Base44 returns an empty catalog or the walk stops early. A partial failure cannot wipe the index.
- Commits each document on its own, so a Voyage or database error on one document leaves the others in place.

`POST /retrieve` still returns the same JSON. Each passage has `document_id`, `title`, `category`, `topic`, `source`, `source_url`, `source_date`, `is_authoritative`, `chunk_text`, and `relevance_score`. `chunk_text` is still the passage itself. `source_date` is filled in when a document is revisited; it used to be selected and always null. How those passages are chosen is described under [Retrieval](#retrieval).

The next reindex after this change re-embeds every document once, because `embed_version` is null on existing rows. Later Monday runs download a body only when `updated_date` moved, and call Voyage only when the body, the title, or the embed version requires it.

## Retrieval

`POST /retrieve` still takes `question`, and optionally `top_k`, `category`, and `source`. Two optional filters were added:

- `exclude_titles`: title substrings to drop, as a list or a single string. Matched case-insensitively, in addition to the built-in exclusion.
- `exclude_categories`: category values to drop, matched exactly.

Titles containing `RC Brain Agent Evaluation Q&A` are always excluded. Set `RETRIEVAL_EXCLUDE_TITLES` to more substrings separated by `||` when another set of docs should stay out of answers. The built-in eval-key exclusion stays on either way.

When the caller omits `top_k`, the response has 8 passages (`RERANK_TOP_N`). A caller that sends `top_k` gets that many, up to 100. Each leg reads a wider pool (at least 80 rows, and at least three times the candidate count). Fusion keeps up to 40 candidates (`TOP_K`) with at most 4 chunks from one document, the reranker orders those, and the response keeps at most 2 chunks from one document (`MAX_CHUNKS_PER_DOCUMENT`). If the cap would leave the answer short of the requested count, later chunks from those same documents fill the rest, so a question that only matches one transcript still returns a full set.

The vector leg is an exact nearest-neighbor scan. `SET LOCAL enable_indexscan = off` and `SET LOCAL enable_bitmapscan = off` run on that query so an IVFFlat or HNSW index created outside this repo cannot return fewer rows than it should. At about 9,000 chunks the sequential scan is the right trade. `ivfflat.probes` and `hnsw.ef_search` are also raised, which covers a planner that still chooses the index. Set `VECTOR_EXACT_SEARCH=false` to leave index scans enabled. The ANN index is not dropped.

The keyword leg uses a `simple` tsvector on the title, topic, and passage (`knowledge_chunks.search_tsv`, weight A on the title and topic). Terms are OR'd, alphabetic words are prefix-matched so `RGU` hits `RGUs`, and codes, acronyms, and ids such as `02`, `RGU`, and `MGB-020` get an exact-token bonus. A keyword hit is still returned when the vector leg comes back empty. The two rankings are merged with reciprocal rank fusion before the reranker runs. The reranker sees `Title: <title>` followed by the passage. The JSON `chunk_text` does not include that prefix.

Full-text search works as soon as this build boots: startup adds the column, the GIN index, and a trigger, and backfills rows whose `search_tsv` is still null. That migration only adds objects. It does not drop tables, columns, or embeddings.

Embeddings learn the title on the next reindex. `indexed_documents.embed_version` is null on existing rows, and a null version is re-embedded even when the body hash and `updated_date` are unchanged. The stored passage text and the content hash stay the same. After that run, unchanged documents are skipped again. `POST /reindex` with `{"force": false}` is enough; `{"force": true}` is not required. The run re-embeds the whole library once (about 9,100 chunks) and spends the corresponding Voyage credits. If startup sees a last successful reindex older than 8 days, that catch-up does the re-embed on its own.

## Schedule

The web process schedules a reindex for **Mondays at 19:00 America/Chicago** (after the Monday Drive sync). The clock follows CST/CDT. No Railway cron service is required, and none should be added.

On startup the service also starts a reindex when `indexer_state` has no finished run, or the last finished run is older than 8 days. The first boot after this change therefore begins the catch-up on its own. A database advisory lock keeps a second replica from indexing at the same time.

Set `REINDEX_SCHEDULER=false` to turn the Monday job off. Set `REINDEX_ON_STARTUP=false` to turn the catch-up off. Leave both unset in production.

## Manual trigger

Same API key as `/retrieve` (`X-API-Key: $RETRIEVAL_API_KEY`).

```bash
curl -sS -X POST "https://<railway-domain>/reindex" \
  -H "X-API-Key: $RETRIEVAL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"force": false}'
```

`force: false` is the incremental reconciliation. `{"force": true}` re-embeds every document and is not needed for the first catch-up. An empty body means `force: false`.

The handler returns **202** immediately (`status: "started"`). A second call while one is running returns **409**.

```bash
curl -sS "https://<railway-domain>/reindex/status" \
  -H "X-API-Key: $RETRIEVAL_API_KEY"
```

`state` moves through `running` → `succeeded`, `completed_with_errors`, or `failed`. `stats` has `fetched`, `indexed`, `skipped`, `deleted`, `empty`, `errors`, and `title_reembeds` (documents re-embedded only so the vector includes the title).

`python indexer.py` and `python indexer.py --force` still work for a local run. Do not use that command as the Railway start command.

## What to change in Railway

Auto-deploy from `main` updates the web service. These are the only hand edits:

1. **Put `BASE44_API_KEY` on the web service** (the one that serves `/retrieve`), not only on the old indexer service. The value should be a Base44 **personal access token** with read access to the RC Brain app: workspace name → Settings → Secrets → Personal access tokens. Account API keys are being shut off, and a token sent as the old `api_key` / `x-api-key` header is rejected once that key is disabled. The indexer tries `Authorization: Bearer` first, then the legacy headers, so an account key that still works will keep working until it is turned off. Do not commit the token.
2. **Confirm `BASE44_APP_ID`** on that same service if it should not use the id already baked into `indexer.py` (`69ef93740d8ea9e5021345b0`).
3. **Confirm the web service already has** `DATABASE_URL`, `VOYAGE_API_KEY`, `VOYAGE_EMBED_MODEL`, `VOYAGE_RERANK_MODEL`, and `RETRIEVAL_API_KEY`. Leave `VOYAGE_EMBED_MODEL` as `voyage-large-2`. The live vectors are 1536-dimensional; a different model will not match the column.
4. **Delete or pause the failed indexer service.** Do not add a Railway cron schedule, and do not point a second service at `python indexer.py` while this repo's healthcheck is `/health`.
5. Leave the web service as a single replica, and do not add `uvicorn --workers`. Each process would start its own scheduler. The database lock still stops two of them from writing at once, but one process is the setup this schedule expects.
6. After the deploy is healthy, open `GET /reindex/status`. The startup catch-up should already be `running`. If `BASE44_API_KEY` was missing on the first boot, set it and then `POST /reindex` with `{"force": false}`.

The next reconciliation downloads every current KnowledgeDocument (about 3,000) and re-embeds all of them once, so each vector includes the document title. At the default 60 Base44 requests per minute that download takes on the order of an hour, and the Voyage re-embed of about 9,100 chunks is on top of that. Later runs re-embed only new or changed documents. `/health` and `/retrieve` keep serving while it runs. Raise `BASE44_RPM` only if the Base44 plan's rate-limit multiplier allows it. The list endpoint's base limit is 70 requests per minute and record fetches are 150; the default stays under both.

Optional variables, all with defaults:

| Variable | Default | Role |
| --- | --- | --- |
| `BASE44_API_BASE` | `https://app.base44.com` | API origin |
| `BASE44_RPM` | `60` | Cap on Base44 requests per minute |
| `BASE44_PAGE_SIZE` | `100` | Cursor page size (max 500) |
| `VOYAGE_EMBED_BATCH_SIZE` | `64` | Texts per Voyage request |
| `VOYAGE_MAX_TOKENS_PER_REQUEST` | `100000` | Estimated-token cap per Voyage request |
| `VOYAGE_MIN_INTERVAL_SECONDS` | `0.5` | Pause between Voyage calls |
| `CHUNK_SIZE` | `800` | Characters per chunk (clamped at 20000) |
| `CHUNK_OVERLAP` | `150` | Overlap; values at or above the chunk size are clamped |
| `REINDEX_SCHEDULER` | on | Monday 19:00 America/Chicago job |
| `REINDEX_ON_STARTUP` | on | Catch up when the last finished run is missing or older than 8 days |
| `TOP_K` | `40` | Candidates kept after fusion, before reranking |
| `RERANK_TOP_N` | `8` | Passages returned when the request omits `top_k` |
| `MAX_CHUNKS_PER_DOCUMENT` | `2` | Final cap per document. `0` disables it |
| `CANDIDATE_CHUNKS_PER_DOCUMENT` | `4` | Cap per document in the rerank set |
| `VECTOR_EXACT_SEARCH` | on | Exact vector scan. `false` leaves ANN index scans on |
| `RETRIEVAL_EXCLUDE_TITLES` | empty | Extra title substrings to drop, separated by `\|\|` |
| `RETRIEVAL_SCHEMA_ON_START` | on | Additive full-text migration at boot |

## Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

The tests cover cursor pagination (no `skip`, every page, no wipe on a partial walk), Voyage batching and retries, incremental skips, deletion of documents removed in Base44, the Monday trigger, the API key on `/reindex`, the `/retrieve` response shape, hybrid fusion, the per-document cap, eval-key exclusion, and the title prefix on embeddings.
