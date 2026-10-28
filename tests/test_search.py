import re

import search
from indexer import chunk_text, passage_for_model
from search import (
    BUILTIN_TITLE_EXCLUSION,
    ChunkHit,
    apply_vector_settings,
    canonical_title,
    collapse_similar_passages,
    collect_duplicate_chunks,
    ilike_contains,
    keyword_search_sql,
    make_group_key,
    neighbor_budget,
    plan_keyword_query,
    retrieval_where,
    rrf_merge,
    select_with_document_cap,
    splice_neighbors,
    title_exclusion_substrings,
    title_is_excluded,
    vector_search_sql,
)


def hit(doc_id, index, text=None, title=None):
    return ChunkHit(
        document_id=doc_id,
        title=doc_id if title is None else title,
        category=None,
        topic=None,
        source=None,
        source_url=None,
        source_date=None,
        is_authoritative=False,
        chunk_text=text if text is not None else f"{doc_id}-{index}",
        chunk_index=index,
        score=1.0,
    )


def test_keyword_plan_keeps_codes_acronyms_and_names():
    codes = plan_keyword_query("What does scanner code 02 mean?")
    assert "02" in codes.lexemes
    assert "scanner" in codes.lexemes
    assert "code" in codes.lexemes
    assert "what" not in codes.lexemes
    assert "does" not in codes.lexemes
    assert "code:*" in codes.tsquery
    assert "02" in codes.tsquery.split(" | ")
    assert any(pattern.endswith("02([^[:alnum:]]|$)") for pattern in codes.exact_patterns)

    form = plan_keyword_query("Where is form MGB-020 filed?")
    assert "mgb" in form.lexemes
    assert "020" in form.lexemes
    assert any("MGB\\-020" in pattern or "MGB-020" in pattern for pattern in form.exact_patterns)

    acronym = plan_keyword_query("What is an RGU?")
    assert acronym.lexemes == ("rgu",)
    assert "rgu:*" in acronym.tsquery

    plural = plan_keyword_query("Regional Geographic Units, or RGUs")
    assert "rgus" in plural.lexemes
    assert "rgu" in plural.lexemes

    hans = plan_keyword_query("Hans revenue examples")
    assert "hans" in hans.lexemes
    assert "revenue" in hans.lexemes


def test_keyword_plan_cannot_carry_sql_into_the_tsquery():
    plan = plan_keyword_query("Robert'); DROP TABLE knowledge_chunks;--")
    assert plan.tsquery
    assert re.fullmatch(r"[a-z0-9:|* ]+", plan.tsquery)
    assert plan_keyword_query("what is the").tsquery is None


def test_keyword_and_vector_sql_keep_their_placeholders_aligned():
    plan = plan_keyword_query("What does scanner code 02 mean for MGB-020 and RGU?")
    where, filters = retrieval_where(
        category="hr",
        source=None,
        exclude_titles=["draft notes"],
        exclude_categories=["Archive"],
    )
    sql, params = keyword_search_sql(where, plan, filters, 80)
    assert sql is not None
    assert sql.count("%s") == len(params)
    assert "to_tsquery('simple'" in sql
    assert "search_tsv @@" in sql
    assert "ORDER BY score DESC" in sql
    assert "text_rank = 1" in sql
    assert "doc_rank <=" in sql
    assert sql.index("text_rank = 1") < sql.index("LIMIT %s")
    assert params[-1] == 80
    assert "NOT ILIKE ALL" in sql
    assert BUILTIN_TITLE_EXCLUSION in str(params)

    vector = vector_search_sql(where)
    assert vector.count("%s") == 1 + where.count("%s") + 2
    assert "embedding <=>" in vector
    assert "text_rank = 1" in vector
    assert "doc_rank <=" in vector
    assert vector.index("doc_rank <=") < vector.index("text_rank = 1") < vector.index("LIMIT %s")
    assert "LIMIT %s" in vector
    lowered = vector.lower()
    assert "similarity >" not in lowered
    assert "embedding >" not in lowered
    assert keyword_search_sql("", plan_keyword_query("what is the"), [], 10) == (None, [])


def test_rrf_promotes_a_chunk_found_by_both_legs_and_keeps_keyword_only_hits():
    alpha = hit("alpha", 0)
    beta = hit("beta", 0)
    keyword_only = hit("hans", 0, "Hans revenue")
    merged = rrf_merge([[alpha, beta], [keyword_only, alpha]])
    assert [item.document_id for item in merged] == ["alpha", "hans", "beta"]


def test_document_cap_leaves_room_for_other_sources_and_backfills_a_single_source():
    crowded = [hit("transcript", index) for index in range(6)]
    crowded.extend([hit("official", 0), hit("portal", 0)])
    chosen = select_with_document_cap(crowded, limit=10, per_document=2, floor=4)
    assert [item.document_id for item in chosen] == ["transcript", "transcript", "official", "portal"]

    only = [hit("hans", index) for index in range(4)]
    filled = select_with_document_cap(only, limit=10, per_document=2, floor=4)
    assert len(filled) == 4
    assert {item.document_id for item in filled} == {"hans"}


def test_eval_title_exclusion_is_always_on_and_literals_are_escaped():
    assert title_exclusion_substrings(None) == [BUILTIN_TITLE_EXCLUSION]
    merged = title_exclusion_substrings(
        ["draft notes", "rc brain agent evaluation q&a", "draft notes"]
    )
    assert merged == [BUILTIN_TITLE_EXCLUSION, "draft notes"]
    assert ilike_contains("100%_done") == "%100\\%\\_done%"
    assert title_is_excluded(
        "Folder / RC Brain Agent Evaluation Q&A 10",
        title_exclusion_substrings(None),
    )
    assert not title_is_excluded("Bread Acquisition Strategy", title_exclusion_substrings(None))


def test_passage_for_the_model_prefixes_the_title_once():
    assert passage_for_model("Vendor Leads", "Call the vendor.") == (
        "Title: Vendor Leads\n\nCall the vendor."
    )
    assert passage_for_model("Sponsorships", "Call the vendor.") != passage_for_model(
        "Vendor Leads", "Call the vendor."
    )
    assert passage_for_model("", "Body only.") == "Body only."
    assert passage_for_model("Untitled", "Body only.") == "Body only."
    prefixed = "Title: Vendor Leads\n\nCall the vendor."
    assert passage_for_model("Vendor Leads", prefixed) == prefixed


CHECKLIST = (
    "1. Inspect the vehicle before the shift starts. "
    "2. Check the tires, lights, and fuel. "
    "3. Confirm the route sheet matches the scanner. "
    "4. Record the odometer. "
    "5. Repeat this opening checklist every morning before the first stop."
)


def _legacy_document_cap(items, limit, per_document, floor):
    """The PR #2 cap. It counted document ids and then backfilled duplicates."""
    chosen = []
    overflow = []
    counts = {}
    for item in items:
        if len(chosen) >= limit:
            break
        seen = counts.get(item.document_id, 0)
        if per_document <= 0 or seen < per_document:
            chosen.append(item)
            counts[item.document_id] = seen + 1
        else:
            overflow.append(item)
    if len(chosen) < floor:
        for item in overflow:
            if len(chosen) >= floor:
                break
            chosen.append(item)
    return chosen


def test_drive_copies_collapse_and_the_cap_then_backfills_distinct_passages():
    """The reported failure: top_k 12, one chunk in 8 of the 12 slots."""
    same_chunk = []
    next_chunk = []
    for index in range(1, 9):
        suffix = f" ({index})" if index < 8 else ""
        doc_id = "original" if suffix == "" else f"copy-{index}"
        title = f"Opening Checklist{suffix}"
        same_chunk.append(hit(doc_id, 4, CHECKLIST, title=title))
        next_chunk.append(
            hit(
                doc_id,
                5,
                "After the checklist, fuel every truck before noon.",
                title=title,
            )
        )
    others = [
        hit(f"memo-{index}", 0, f"Distinct memo {index} about a different topic.", title=f"Memo {index}")
        for index in range(10)
    ]
    # The repeated chunk ranks above everything else, which is what filled the list.
    ranked = same_chunk + next_chunk + others

    before = _legacy_document_cap(ranked, limit=12, per_document=2, floor=12)
    before_texts = [item.chunk_text for item in before]
    assert before_texts.count(CHECKLIST) == 8
    assert len(set(before_texts)) == 2

    collapsed = collapse_similar_passages(ranked)
    after = select_with_document_cap(
        collapsed,
        limit=12,
        per_document=2,
        floor=12,
        group_key=make_group_key(collapsed),
    )
    after_texts = [item.chunk_text for item in after]
    assert after_texts.count(CHECKLIST) == 1
    assert len(after) == 12
    assert len(set(after_texts)) == 12
    checklist_slots = [item for item in after if (item.title or "").startswith("Opening Checklist")]
    assert len(checklist_slots) == 2
    assert all(item.title == "Opening Checklist" for item in checklist_slots)
    assert all(item.document_id == "original" for item in checklist_slots)


def test_near_duplicate_text_collapses_across_docs_not_inside_one_list():
    variant = CHECKLIST + " Page 2."
    collapsed = collapse_similar_passages(
        [
            hit("copy", 0, variant, title="Opening Checklist (2)"),
            hit("original", 0, CHECKLIST, title="Opening Checklist"),
            hit("other", 0, "Wear gloves on the dock and sign the log.", title="Safety"),
        ]
    )
    assert [item.document_id for item in collapsed] == ["original", "other"]
    assert canonical_title("Opening Checklist (2)") == "opening checklist"

    steps = "\n".join(
        f"{number}. Step {number} of the opening checklist for this route."
        for number in range(1, 40)
    )
    parts = chunk_text(steps, size=220, overlap=40)
    assert len(parts) >= 3
    same_doc = [
        hit("checklist", index, part, title="Opening Checklist")
        for index, part in enumerate(parts)
    ]
    assert len(collapse_similar_passages(same_doc)) == len(parts)


def test_neighbor_splice_keeps_a_split_list_and_stays_within_top_k():
    middle = hit("checklist", 2, "4. Delta 5. Echo 6. Foxtrot", title="Opening Checklist")
    primary = [middle] + [
        hit(f"memo-{index}", 0, f"Memo body {index}.", title=f"Memo {index}")
        for index in range(7)
    ]
    neighbors = [
        hit("checklist", 1, "1. Alpha 2. Bravo 3. Charlie", title="Opening Checklist"),
        hit("checklist", 3, "7. Golf 8. Hotel 9. India", title="Opening Checklist"),
    ]
    assert neighbor_budget(12) == 4
    assert neighbor_budget(8) == 2
    assert neighbor_budget(3) == 0
    spliced = splice_neighbors(
        primary,
        neighbors,
        limit=8,
        expand_doc_ids=["checklist"],
        max_neighbors=neighbor_budget(8),
    )
    assert [item.chunk_text for item in spliced[:3]] == [
        "1. Alpha 2. Bravo 3. Charlie",
        "4. Delta 5. Echo 6. Foxtrot",
        "7. Golf 8. Hotel 9. India",
    ]
    assert len(spliced) == 8
    assert len({item.chunk_text for item in spliced}) == 8


def test_duplicate_chunk_report_does_not_delete_unless_apply_is_set():
    class Cur:
        def __init__(self):
            self.statements = []
            self.rowcount = 2

        def execute(self, sql, params=None):
            self.statements.append(sql)

        def fetchall(self):
            sql = self.statements[-1]
            if "duplicate_identical" in sql:
                return [("doc-1", "Notes (1)", 0, 3, "alpha passage")]
            if "duplicate_cross_doc" in sql:
                return [(8, 8, "alpha passage", ["Notes", "Notes (1)"])]
            if "duplicate_repeated_within" in sql:
                return [("doc-1", "Notes", 2, 2, "repeated paragraph")]
            return []

    cur = Cur()
    report = collect_duplicate_chunks(cur, apply=False)
    assert report["mode"] == "report"
    assert report["deleted"] == 0
    assert report["extra_identical_rows"] == 2
    assert report["identical_row_groups"] == 1
    assert report["cross_document_text_groups"] == 1
    assert report["cross_document_preview"][0]["documents"] == 8
    assert not any("duplicate_delete" in sql for sql in cur.statements)

    cur = Cur()
    deleted = collect_duplicate_chunks(cur, apply=True)
    assert deleted["mode"] == "deleted"
    assert deleted["deleted"] == 2
    assert any("duplicate_delete" in sql for sql in cur.statements)


def test_exact_scan_settings_survive_an_unknown_guc():
    class Cur:
        def __init__(self):
            self.statements = []

        def execute(self, sql, params=None):
            self.statements.append(sql)
            if "iterative_scan" in sql:
                raise RuntimeError("unrecognized configuration parameter")

    cur = Cur()
    apply_vector_settings(cur, exact=True)
    text = "\n".join(cur.statements)
    assert "SET LOCAL enable_indexscan = off" in text
    assert "SET LOCAL enable_bitmapscan = off" in text
    assert "SET LOCAL hnsw.ef_search = 1000" in text
    assert "ROLLBACK TO SAVEPOINT retrieval_guc" in text
    assert "SET LOCAL ivfflat.probes = 1000" in text
