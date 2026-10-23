# RC Brain retrieval

Semantic search over the RC Brain `KnowledgeDocument` library. Base44 calls `POST /retrieve`. The service embeds the question with Voyage AI, reads the nearest chunks from Postgres + pgvector, reranks them, and returns passages.

Railway auto-deploys this repo from `main`. The running service is the web process defined in `railway.toml` (`uvicorn api:app`, healthcheck `GET /health`).

## What broke the indexer

The last successful index was 2026-06-25 (1,267 documents, 8,284 chunks). Later indexer deploys failed for two stacked reasons:

1. **The Base44 call cannot succeed anymore.** `indexer.py` requested `https://api.base44.com/.../KnowledgeDocument?limit=&skip=`. That host now returns an HTML 404. The live API is `https://app.base44.com/api/apps/{app_id}/entities/KnowledgeDocument/v2/list`, which pages with a cursor. Sending `skip` is an HTTP 400. `raise_for_status()` was not caught, so the process exited and Railway marked the deployment failed.
2. **A one-shot indexer cannot satisfy this service's healthcheck.** `railway.toml` probes `GET /health`. A service whose start command is `python indexer.py` never serves that route, so Railway fails the deploy even when the script itself is healthy. Indexing now runs inside the web process, which already serves `/health`.

Large documents (up to about 100k characters) made the old page of 100 full bodies likely to time out as well. The new walk lists metadata only, then downloads a body when that document actually needs work. Voyage calls are batched under the per-request token cap and retried on 429 and 5xx. Base44 calls are paced (default 60/minute, under the list endpoint's base limit of 70/minute) and retried the same way.

## What a reindex does

It does **not** drop `knowledge_chunks` or `indexed_documents`.

- Pages through every `KnowledgeDocument` with the cursor API.
- Skips a document when its stored `updated_date` matches Base44 and it already has chunks.
- Otherwise downloads the body, MD5s it the same way the June index did, and re-embeds only when the hash changed or `--force` / `{"force": true}` is set. A matching hash still refreshes chunk metadata (`source_date`, title, category, and the rest) without calling Voyage.
- Deletes chunks and the registry row for documents Base44 no longer returns.
- Refuses to delete anything if Base44 returns an empty catalog or the walk stops early. A partial failure cannot wipe the index.
- Commits each document on its own, so a Voyage or database error on one document leaves the others in place.

`POST /retrieve` is unchanged. Passages still include `document_id`, `title`, `category`, `topic`, `source`, `source_url`, `source_date`, `is_authoritative`, `chunk_text`, and `relevance_score`. `source_date` is now filled in when a document is revisited; it used to be selected and always null.

The first run after deploy still downloads every body once, because `updated_date` was not stored before. Unchanged documents are not re-embedded. Later Monday runs only download documents whose `updated_date` moved.

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

`state` moves through `running` → `succeeded`, `completed_with_errors`, or `failed`. `stats` has `fetched`, `indexed`, `skipped`, `deleted`, `empty`, and `errors`.

`python indexer.py` and `python indexer.py --force` still work for a local run. Do not use that command as the Railway start command.

## What to change in Railway

Auto-deploy from `main` updates the web service. These are the only hand edits:

1. **Put `BASE44_API_KEY` on the web service** (the one that serves `/retrieve`), not only on the old indexer service. The value should be a Base44 **personal access token** with read access to the RC Brain app: workspace name → Settings → Secrets → Personal access tokens. Account API keys are being shut off, and a token sent as the old `api_key` / `x-api-key` header is rejected once that key is disabled. The indexer tries `Authorization: Bearer` first, then the legacy headers, so an account key that still works will keep working until it is turned off. Do not commit the token.
2. **Confirm `BASE44_APP_ID`** on that same service if it should not use the id already baked into `indexer.py` (`69ef93740d8ea9e5021345b0`).
3. **Confirm the web service already has** `DATABASE_URL`, `VOYAGE_API_KEY`, `VOYAGE_EMBED_MODEL`, `VOYAGE_RERANK_MODEL`, and `RETRIEVAL_API_KEY`. Leave `VOYAGE_EMBED_MODEL` as `voyage-large-2`. The live vectors are 1536-dimensional; a different model will not match the column.
4. **Delete or pause the failed indexer service.** Do not add a Railway cron schedule, and do not point a second service at `python indexer.py` while this repo's healthcheck is `/health`.
5. Leave the web service as a single replica, and do not add `uvicorn --workers`. Each process would start its own scheduler. The database lock still stops two of them from writing at once, but one process is the setup this schedule expects.
6. After the deploy is healthy, open `GET /reindex/status`. The startup catch-up should already be `running`. If `BASE44_API_KEY` was missing on the first boot, set it and then `POST /reindex` with `{"force": false}`.

The first reconciliation downloads every current KnowledgeDocument (about 3,000) and re-embeds only new or changed ones. At the default 60 Base44 requests per minute that download takes on the order of an hour. `/health` and `/retrieve` keep serving. Raise `BASE44_RPM` only if the Base44 plan's rate-limit multiplier allows it. The list endpoint's base limit is 70 requests per minute and record fetches are 150; the default stays under both.

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

## Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

The tests cover cursor pagination (no `skip`, every page, no wipe on a partial walk), Voyage batching and retries, incremental skips, deletion of documents removed in Base44, the Monday trigger, the API key on `/reindex`, and the `/retrieve` response shape.
