"""
tests/test_batch_run.py
─────────────────────────
Unit tests for the configurable inter-query pause in batch_run.py
(BATCH_QUERY_DELAY_MS), added to reduce the risk of hitting a hosted
LLM_PROVIDER's rate limit on a long benchmark run.

RENAMED from test_batch_query_delay.py to match the source module
(batch_run.py) it tests, following the convention used across this refactor.
Content otherwise unmodified.

Deliberately dependency-free: no pipeline, no schema load, no LLM calls.
Two things are tested in isolation:

  1. Config loading — Settings().batch_query_delay_ms reads BATCH_QUERY_DELAY_MS,
     defaults to 0, milliseconds (not seconds), rejects negative values.
  2. The pacing function itself — batch_run._maybe_inter_query_delay(idx, delay_ms),
     the actual function the batch loop calls, not a re-implemented mirror of it.
     time.sleep is mocked throughout; these tests never really sleep.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings
from batch_run import _maybe_inter_query_delay


def _settings(**overrides) -> Settings:
    """Build Settings from explicit values, ignoring the developer's .env."""
    return Settings(_env_file=None, **overrides)


class TestBatchQueryDelayConfig(unittest.TestCase):
    """Configuration is correctly loaded — default, override, and validation."""

    def test_default_delay_is_zero(self):
        """No BATCH_QUERY_DELAY_MS set -> 0 -> no delay -> current behaviour."""
        self.assertEqual(_settings().batch_query_delay_ms, 0)

    def test_configured_delay_is_read_from_env_var(self):
        """BATCH_QUERY_DELAY_MS env var overrides the default correctly."""
        with patch.dict("os.environ", {"BATCH_QUERY_DELAY_MS": "250"}):
            self.assertEqual(_settings().batch_query_delay_ms, 250)

    def test_delay_is_an_integer_milliseconds_value(self):
        """The field holds milliseconds, not seconds or a float -- e.g. 1500 ms,
        not 1.5. Guards against a unit mix-up regressing silently."""
        with patch.dict("os.environ", {"BATCH_QUERY_DELAY_MS": "1500"}):
            s = _settings()
            self.assertIsInstance(s.batch_query_delay_ms, int)
            self.assertEqual(s.batch_query_delay_ms, 1500)

    def test_negative_delay_is_rejected(self):
        """A negative delay is nonsensical -- the field enforces ge=0."""
        with self.assertRaises(Exception):
            _settings(batch_query_delay_ms=-1)

    def test_explicit_zero_override_matches_default(self):
        """Explicitly setting 0 behaves identically to leaving it unset."""
        with patch.dict("os.environ", {"BATCH_QUERY_DELAY_MS": "0"}):
            self.assertEqual(_settings().batch_query_delay_ms, 0)


class TestMaybeInterQueryDelay(unittest.TestCase):
    """
    Exercises the actual function batch_run.py's loop calls
    (`_maybe_inter_query_delay`), not a re-implemented copy of its logic.
    """

    def test_default_zero_ms_never_sleeps(self):
        """delay_ms=0 (the default): sleep is never called, for any idx.
        This is the 'preserve current behaviour' requirement -- a batch run
        with no BATCH_QUERY_DELAY_MS set must behave exactly as before."""
        with patch("batch_run.time.sleep") as mock_sleep:
            for idx in range(1, 6):
                _maybe_inter_query_delay(idx, 0)
            mock_sleep.assert_not_called()

    def test_no_delay_before_first_query(self):
        """idx == 1 (the first question) never sleeps, regardless of delay_ms.
        A configured delay must not add a startup pause before any work has
        happened yet -- only between questions."""
        with patch("batch_run.time.sleep") as mock_sleep:
            _maybe_inter_query_delay(1, 500)
            mock_sleep.assert_not_called()

    def test_configured_delay_applied_from_second_query_onward(self):
        """delay_ms > 0: sleep IS called starting from idx == 2."""
        with patch("batch_run.time.sleep") as mock_sleep:
            _maybe_inter_query_delay(2, 500)
            mock_sleep.assert_called_once_with(0.5)

    def test_delay_occurs_between_queries_only(self):
        """Simulating a 5-question batch: sleep is called exactly N-1 times
        (once between each pair of consecutive questions), never on the
        first iteration -- i.e. delay count == len(questions) - 1."""
        total_questions = 5
        with patch("batch_run.time.sleep") as mock_sleep:
            for idx in range(1, total_questions + 1):
                _maybe_inter_query_delay(idx, 300)
            self.assertEqual(mock_sleep.call_count, total_questions - 1)
            # Every call used the same, correctly-converted duration.
            for call in mock_sleep.call_args_list:
                self.assertEqual(call.args[0], 0.3)

    def test_milliseconds_converted_to_seconds_for_time_sleep(self):
        """time.sleep() takes seconds; BATCH_QUERY_DELAY_MS is documented and
        configured in milliseconds. Verify the conversion is exact, not
        approximate, for a value that would reveal a units mistake."""
        with patch("batch_run.time.sleep") as mock_sleep:
            _maybe_inter_query_delay(2, 1000)   # 1000 ms -> should be 1.0 s, not 1000 s
        mock_sleep.assert_called_once_with(1.0)

    def test_single_question_run_never_sleeps(self):
        """A one-question batch (idx == 1 only) never sleeps even with a
        configured delay -- there is no 'between' for a single question."""
        with patch("batch_run.time.sleep") as mock_sleep:
            _maybe_inter_query_delay(1, 750)
            mock_sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
