"""
tests/test_phase1_changes.py
─────────────────────────────
Regression tests for the four Phase-1 accuracy fixes.  Each test is
deliberately tied to a SPECIFIC failure pattern observed in
batch-run-output-20260627_131858_054696.jsonl so that if the test
ever fails again, it's clear which production bug it was supposed
to prevent.

Run with:    pytest tests/test_phase1_changes.py -v
Or stand-alone:  python tests/test_phase1_changes.py

NOTE: this file uses lightweight mocks for the validator's heavy
external deps (psycopg2, MCP, transformers tokenizer, etc.) so it
runs anywhere.  The Phase-1 changes themselves don't need any of
those — they are pure-AST + schema-map logic.
"""

import sys
import types
import unittest


def _install_stubs():
    """Install lightweight stubs for modules the validator imports."""
    for name, attrs in [
        ('utils.heuristics', {'HEURISTICS': {
            'ordinal_columns': [], 'phantom_patterns': [], 'safe_literals': [],
            'anti_join_negation_phrases': [], 'anti_join_triggers': set(),
            'trigger_tables': {}, 'trigger_words': {}, 'column_blocklist': [],
        }}),
        ('utils.tokenizer', {'count_tokens': lambda s: len(s) // 4}),
    ]:
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
    class _Silent:
        def __getattr__(self, _n): return lambda *a, **k: None
    ll = types.ModuleType('utils.logging_config')
    ll.get_logger = lambda _n: _Silent()
    sys.modules['utils.logging_config'] = ll
    cs = types.ModuleType('config.settings')
    class _Pg:
        statement_timeout_ms = 30000; max_rows = 1000; pool_min = 2; pool_max = 20; host = ''
    class _Val:
        max_retries = 2; explain_cost_threshold = 1_000_000
    class _LLM: context_size = 32768
    class _Settings:
        use_mcp_servers = False; debug_mode = False
        postgres = _Pg(); validation = _Val(); llm = _LLM()
    cs.settings = _Settings()
    sys.modules['config.settings'] = cs
    mc = types.ModuleType('mcp_tools.client')
    mc.MCPCallError = Exception
    mc.call_postgres_explain = lambda *a, **k: {"error": "no MCP"}
    mc.call_postgres_execute = lambda *a, **k: {"error": "no MCP"}
    mc.call_corpus_log_failure = lambda *a, **k: None
    mc.QdrantMCPClient = object
    mc.OpenSearchMCPClient = object
    sys.modules['mcp_tools.client'] = mc
    sys.modules['mcp_tools'] = types.ModuleType('mcp_tools')
    psy = types.ModuleType('psycopg2')
    class _PE(Exception): pass
    psy.Error = _PE
    sys.modules['psycopg2'] = psy
    pp = types.ModuleType('psycopg2.pool')
    pp.ThreadedConnectionPool = object
    pp.PoolError = Exception
    psy.pool = pp
    sys.modules['psycopg2.pool'] = pp


_install_stubs()

# Ensure the project root is on sys.path when run standalone
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.schema import TableInventory, ColumnInfo
from validation.semantic.semantic_checks import check_groupby_alignment, check_semantic
from validation.core.sql_validator import SQLValidator


def _mk_tbl(name: str, cols_with_allowed):
    inv = TableInventory(table_name=name)
    for cname, allowed in cols_with_allowed:
        ci = ColumnInfo(name=cname, data_type='varchar')
        if allowed is not None:
            ci.allowed_values = set(allowed)
        inv.columns[cname] = ci
    return inv


_SAMPLE_SCHEMA = {
    'board':             _mk_tbl('board', [
        ('id', None), ('course_id', None), ('exam_id', None),
        ('status', ['OPEN', 'CLOSED']), ('deadline', None),
    ]),
    'board_coordinator': _mk_tbl('board_coordinator', [
        ('id', None), ('board_id', None), ('faculty_cache_id', None),
    ]),
    'faculty_cache':     _mk_tbl('faculty_cache', [
        ('id', None), ('employee_erp_id', None), ('name', None), ('email', None),
    ]),
    'answer_key':        _mk_tbl('answer_key', [
        ('id', None), ('qp_id', None),
        ('status', ['DRAFT', 'APPROVED', 'LOCKED']),
    ]),
    'attempt_rule':      _mk_tbl('attempt_rule', [
        ('id', None), ('question_id', None),
        ('rule_type', ['GROUP', 'PICK_N', 'FIRST_N', 'BEST_N']),
    ]),
    'question_paper':    _mk_tbl('question_paper', [('id', None), ('title', None)]),
    'question':          _mk_tbl('question', [('id', None), ('qp_id', None)]),
    'app_user':          _mk_tbl('app_user', [
        ('id', None), ('display_name', None),
        ('user_type', ['STUDENT', 'FACULTY', 'EVALUATOR', 'ADMIN_STAFF']),
    ]),
    'bulk_operation_log': _mk_tbl('bulk_operation_log', [
        ('id', None), ('started_by', None), ('operation_type', None),
    ]),
    'answer_script':     _mk_tbl('answer_script', [
        ('id', None), ('urn', None),
        ('lifecycle_status', ['ADMITTED', 'ELIGIBLE', 'ATTEMPTED', 'ABSENT']),
    ]),
    'academic_unit':     _mk_tbl('academic_unit', [
        ('id', None), ('parent_id', None), ('name', None),
    ]),
}


# ═════════════════════════════════════════════════════════════════════════════
# Change 2: PostgreSQL planner-hint autofix
# ═════════════════════════════════════════════════════════════════════════════

class TestPlannerHintAutofix(unittest.TestCase):
    """Q34-style: `b.name` does not exist, PG suggests `fc.name` — auto-fix."""

    def setUp(self):
        self.v = SQLValidator(schema_map=_SAMPLE_SCHEMA)

    def test_accepts_hint_when_target_in_scope_and_in_ddl(self):
        bad_sql = (
            "SELECT b.id, b.name "
            "FROM board b "
            "JOIN board_coordinator bc ON bc.board_id = b.id "
            "JOIN faculty_cache fc ON fc.id = bc.faculty_cache_id "
            "WHERE b.id = 5"
        )
        pg_err = (
            'column "b.name" does not exist\n'
            'HINT: Perhaps you meant to reference the column "fc.name"'
        )
        fixed, desc = self.v._attempt_pg_autofix(
            bad_sql, pg_err, run_explain=lambda _: (None, None)
        )
        self.assertIsNotNone(fixed)
        self.assertIn('fc.name', fixed)
        self.assertNotIn('b.name', fixed)
        self.assertIn('autofix', desc)

    def test_rejects_hint_when_target_table_not_in_sql(self):
        """If PG suggests fc.name but fc isn't in our FROM, refuse to fabricate."""
        bad_sql = "SELECT b.color FROM board b"
        pg_err = (
            'column "b.color" does not exist\n'
            'HINT: Perhaps you meant to reference the column "fc.name"'
        )
        fixed, _ = self.v._attempt_pg_autofix(
            bad_sql, pg_err, run_explain=lambda _: (None, None)
        )
        self.assertIsNone(fixed)

    def test_rejects_hint_when_target_column_not_in_ddl(self):
        """Defence against malformed/corrupt PG hints."""
        bad_sql = (
            "SELECT b.foo FROM board b "
            "JOIN faculty_cache fc ON fc.id = b.id"
        )
        pg_err = (
            'column "b.foo" does not exist\n'
            'HINT: Perhaps you meant to reference the column "fc.nonexistent"'
        )
        fixed, _ = self.v._attempt_pg_autofix(
            bad_sql, pg_err, run_explain=lambda _: (None, None)
        )
        self.assertIsNone(fixed)

    def test_no_autofix_without_planner_hint(self):
        """If PG provides no hint, we don't speculate."""
        bad_sql = "SELECT b.foo FROM board b"
        pg_err  = 'column "b.foo" does not exist'
        fixed, _ = self.v._attempt_pg_autofix(
            bad_sql, pg_err, run_explain=lambda _: (None, None)
        )
        self.assertIsNone(fixed)

    def test_rejects_autofix_when_re_explain_still_fails(self):
        """Defence: if the rewritten SQL still fails, fall back to retry."""
        bad_sql = (
            "SELECT b.id, b.name "
            "FROM board b "
            "JOIN faculty_cache fc ON fc.id = b.id"
        )
        pg_err = (
            'column "b.name" does not exist\n'
            'HINT: Perhaps you meant to reference the column "fc.name"'
        )
        fixed, _ = self.v._attempt_pg_autofix(
            bad_sql, pg_err,
            run_explain=lambda _: ("42703", "still broken"),
        )
        self.assertIsNone(fixed)


# ═════════════════════════════════════════════════════════════════════════════
# Change 3: Static GROUP BY / SELECT alignment
# ═════════════════════════════════════════════════════════════════════════════

class TestStaticGroupByCheck(unittest.TestCase):

    def test_clean_aggregate_with_groupby_passes(self):
        sql = "SELECT b.status, COUNT(*) FROM board b GROUP BY b.status"
        r = check_groupby_alignment(sql)
        self.assertTrue(r.passed)

    def test_q36_pattern_order_by_uncovered(self):
        """Q36: ORDER BY a.id with GROUP BY not including a.id."""
        sql = (
            "SELECT a.id, ea.status, COUNT(*) "
            "FROM answer_script a "
            "JOIN evaluation_attempt ea ON ea.script_id = a.id "
            "GROUP BY ea.status "
            "ORDER BY a.id LIMIT 100"
        )
        # Note: a.id IS in SELECT too; the check exempts when "id" is in GROUP BY.
        # Here id is NOT in GROUP BY, so both SELECT and ORDER BY references should
        # be flagged.
        r = check_groupby_alignment(sql)
        self.assertFalse(r.passed)
        self.assertEqual(r.step, "schema")

    def test_q155_pattern_case_alias_in_groupby_passes(self):
        """Q155 was a former FP: CASE-aliased projection where GROUP BY uses the alias."""
        sql = (
            "SELECT CASE WHEN EXISTS ("
            "SELECT 1 FROM academic_unit_closure WHERE ancestor_id = au.id) "
            "THEN 'HAS_CHILDREN' ELSE 'NO_CHILDREN' END AS has_children, "
            "COUNT(au.id) AS count "
            "FROM academic_unit au GROUP BY has_children"
        )
        r = check_groupby_alignment(sql)
        self.assertTrue(r.passed, msg=f"Q155 was a former false positive; should pass now. msg: {r.message[:200]}")

    def test_window_function_passes(self):
        sql = (
            "SELECT b.status, ROW_NUMBER() OVER (PARTITION BY b.course_id) "
            "FROM board b GROUP BY b.status"
        )
        r = check_groupby_alignment(sql)
        self.assertTrue(r.passed)

    def test_no_groupby_clause_means_no_check(self):
        sql = "SELECT b.status, b.deadline FROM board b"
        r = check_groupby_alignment(sql)
        self.assertTrue(r.passed)

    def test_groupby_id_triggers_fd_relaxation(self):
        """PG's FD relaxation: GROUP BY id allows projecting any same-table col."""
        sql = "SELECT b.id, b.status, b.deadline, COUNT(*) FROM board b GROUP BY b.id"
        r = check_groupby_alignment(sql)
        self.assertTrue(r.passed)


# ═════════════════════════════════════════════════════════════════════════════
# Change 4: Schema-driven defensive-filter check (Check 18b)
# ═════════════════════════════════════════════════════════════════════════════

class TestSchemaDrivenDefensiveFilter(unittest.TestCase):

    def _is_18b_message(self, msg: str) -> bool:
        return "defensive filter not supported" in (msg or "")

    def test_q67_pattern_status_in_join_on(self):
        """Q67: ak.status='APPROVED' in JOIN ON; NL never mentions 'approved'."""
        sql = (
            "SELECT fc.name FROM answer_key ak "
            "JOIN question_paper qp ON qp.id = ak.qp_id "
            "JOIN board b ON b.qp_id = ak.qp_id "
            "JOIN board_coordinator bc ON bc.board_id = b.id "
            "JOIN faculty_cache fc ON fc.id = bc.faculty_cache_id "
            "WHERE qp.title ILIKE '%Algorithms%' AND ak.status = 'APPROVED'"
        )
        nl = "List all faculty members who prepared the answer key for the Algorithms exam"
        r = check_semantic(sql, nl, schema_map=_SAMPLE_SCHEMA)
        # Either Check 18b catches it OR an earlier check catches it for a
        # different reason — we accept either as long as the test query is
        # flagged.  The fundamental property we want is "this defensive
        # filter is detected somehow".
        self.assertFalse(r.passed)

    def test_q162_pattern_rule_type_unjustified(self):
        """Q162: ar.rule_type='PICK_N' without NL ever asking about PICK_N."""
        sql = (
            "SELECT ar.id, ar.rule_type "
            "FROM attempt_rule ar "
            "JOIN question q ON q.id = ar.question_id "
            "JOIN question_paper qp ON qp.id = q.qp_id "
            "WHERE qp.title ILIKE '%Data Structures%' AND ar.rule_type = 'PICK_N'"
        )
        nl = "Show the attempt rule configuration for question 2 in the Data Structures paper."
        r = check_semantic(sql, nl, schema_map=_SAMPLE_SCHEMA)
        self.assertFalse(r.passed)
        self.assertTrue(self._is_18b_message(r.message))

    def test_active_marking_filter_is_exempt(self):
        """answer_script.lifecycle_status='ATTEMPTED' is the documented active-marking pattern; not a defensive filter."""
        sql = "SELECT a.urn FROM answer_script a WHERE a.lifecycle_status = 'ATTEMPTED'"
        nl = "Show all scripts where the primary evaluator has frozen the attempt"
        r = check_semantic(sql, nl, schema_map=_SAMPLE_SCHEMA)
        # Existing Check 18 may pass it via keyword whitelist; our Check 18b
        # exempts answer_script status columns explicitly.  Either way the
        # result is "not flagged by 18b".  We test the negative property:
        self.assertFalse(self._is_18b_message(r.message or ''))

    def test_value_paraphrased_in_nl_is_justified(self):
        """END_SEM with 'end-semester' in NL is justified via underscore-to-hyphen normalisation."""
        schema = dict(_SAMPLE_SCHEMA)
        schema['exam_schedule_cache'] = _mk_tbl('exam_schedule_cache', [
            ('id', None), ('exam_type', ['END_SEM', 'MID_SEM']),
        ])
        sql = (
            "SELECT id FROM exam_schedule_cache esc "
            "WHERE esc.exam_type = 'END_SEM'"
        )
        nl = "List all end-semester exam schedules"
        r = check_semantic(sql, nl, schema_map=schema)
        # Justification should make this pass — value 'END_SEM' matches NL
        # 'end-semester' via the '_'→'-' normalisation.
        self.assertFalse(self._is_18b_message(r.message or ''))


# ═════════════════════════════════════════════════════════════════════════════
# Change 1: Column Cheatsheet (smoke test only — full prompt build is heavy)
# ═════════════════════════════════════════════════════════════════════════════

class TestColumnCheatsheet(unittest.TestCase):

    def test_cheatsheet_block_present_when_tables_provided(self):
        from generation.prompt_builder import PromptBuilder
        from models.schema import ChunkType, SemanticChunk, ParsedQuery, QueryIntent
        parsed = ParsedQuery(
            original='Q', normalised='Q', intent=QueryIntent.LOOKUP,
            entities=['board', 'faculty_cache'], status_codes=[], domain_terms=[],
        )
        chunks = [
            SemanticChunk(text="board has columns ...", chunk_type=ChunkType.TABLE,
                          table_name='board', referenced_tables=['board']),
            SemanticChunk(text="faculty_cache has columns ...", chunk_type=ChunkType.TABLE,
                          table_name='faculty_cache', referenced_tables=['faculty_cache']),
        ]
        b = PromptBuilder()
        prompt = b.build(
            parsed_query=parsed, schema_chunks=chunks,
            tables=_SAMPLE_SCHEMA,
        )
        self.assertIn('=== COLUMN CHEATSHEET ===', prompt)
        # The two entity tables MUST appear in the cheatsheet
        self.assertIn('board:', prompt)
        self.assertIn('faculty_cache:', prompt)
        # And `name` should appear in faculty_cache's column list, not board's
        cheat_start = prompt.index('=== COLUMN CHEATSHEET ===')
        cheat_end   = prompt.index('=== QUESTION ===')
        cheat       = prompt[cheat_start:cheat_end]
        # Pull the board line
        for line in cheat.split('\n'):
            if line.startswith('board:'):
                self.assertNotIn('name', line.split(':',1)[1])  # board has no `name`
            if line.startswith('faculty_cache:'):
                self.assertIn('name', line.split(':',1)[1])      # faculty_cache has `name`

    def test_no_cheatsheet_when_tables_not_provided(self):
        """Backward compat: when caller doesn't pass tables=, the block isn't emitted."""
        from generation.prompt_builder import PromptBuilder
        from models.schema import ParsedQuery, QueryIntent
        parsed = ParsedQuery(
            original='Q', normalised='Q', intent=QueryIntent.LOOKUP,
            entities=['board'], status_codes=[], domain_terms=[],
        )
        b = PromptBuilder()
        prompt = b.build(parsed_query=parsed, schema_chunks=[])
        self.assertNotIn('=== COLUMN CHEATSHEET ===', prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)