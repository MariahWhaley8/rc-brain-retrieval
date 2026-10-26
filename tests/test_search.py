import re

import search
from indexer import passage_for_model
from search import (
    BUILTIN_TITLE_EXCLUSION,
    ChunkHit,
    apply_vector_settings,
    ilike_contains,
    keyword_search_sql,
    plan_keyword_query,
    retrieval_where,
    rrf_merge,
    select_with_document_cap,
    title_exclusion_substrings,
    title_is_excluded,
    vector_search_sql,
)


def hit(doc_id, index, text=None):
    return ChunkHit(
        document_id=doc_id,
        title=doc_id,
        category=None,
        topic=None,
        source=None,
        source_url=None,
        source_date=None,
        is_authoritative=False,
        chunk_text=text if text is not None else f"{doc_id}-{index}",
        chunk_index=index,
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
    assert "ORDER BY rank DESC" in sql
    assert params[-1] == 80
    assert "NOT ILIKE ALL" in sql
    assert BUILTIN_TITLE_EXCLUSION in str(params)

    vector = vector_search_sql(where)
    assert vector.count("%s") == 2 + where.count("%s") + 1
    assert "ORDER BY embedding <=>" in vector
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
