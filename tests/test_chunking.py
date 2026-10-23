import indexer


def test_document_content_strips_whitespace_and_null_bytes():
    assert indexer.document_content({"content": "  a\x00b  "}) == "ab"
    assert indexer.document_content({"content": None}) == ""
    assert indexer.document_content({}) == ""


def test_content_hash_matches_existing_md5_and_changes_with_text():
    assert indexer.content_hash("Hello") == "8b1a9953c4611296a827abf8c47804d7"
    assert indexer.content_hash("Hello") != indexer.content_hash("Hello ")
    # The indexer hashes the stripped body, same as the June 2026 rows.
    assert indexer.content_hash("Hello") == indexer.content_hash("Hello")


def test_chunk_text_breaks_on_a_sentence_and_covers_the_tail():
    text = ("a" * 700) + ". " + ("b" * 500)
    chunks = indexer.chunk_text(text, size=800, overlap=0)
    assert chunks[0].endswith(".")
    assert len(chunks[0]) <= 800
    assert "b" in chunks[-1]
    assert chunks[-1].startswith("b")


def test_large_document_without_sentence_breaks_finishes():
    text = "a" * 100_000
    chunks = indexer.chunk_text(text, size=800, overlap=150)
    assert len(chunks) == 154
    assert chunks[0] == "a" * 800
    assert len(chunks[-1]) == 550
    assert all(chunk and len(chunk) <= 800 for chunk in chunks)


def test_overlap_cannot_stall_the_window():
    chunks = indexer.chunk_text("x" * 5_000, size=50, overlap=50)
    assert chunks
    assert all(len(chunk) <= 50 for chunk in chunks)
    # Clamped overlap still reaches the end of the text.
    assert sum(len(chunk) for chunk in chunks) >= 5_000


def test_timestamps_with_a_z_suffix_compare_equal():
    assert indexer.normalize_timestamp("2026-06-04T14:07:02.115000Z") == (
        "2026-06-04T14:07:02.115000"
    )
    assert indexer.normalize_timestamp("2026-06-04T14:07:02.115000") == (
        indexer.normalize_timestamp("2026-06-04T14:07:02.115000Z")
    )


def test_document_id_prefers_an_id_already_in_the_index():
    record = {"id": "new-id", "_id": "stored-id", "document_id": "other"}
    assert indexer.choose_document_id(record, {"stored-id"}) == "stored-id"
    assert indexer.choose_document_id(record, set()) == "new-id"
    assert indexer.base44_entity_id(record) == "new-id"


def test_embed_batches_respect_item_and_token_caps():
    texts = ["a" * 400] * 10  # 100 estimated tokens each
    batches = list(indexer.iter_embed_batches(texts, max_items=4, max_tokens=250))
    assert [len(batch) for batch in batches] == [2, 2, 2, 2, 2]
    assert sum(len(batch) for batch in batches) == 10
