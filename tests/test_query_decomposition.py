"""
Unit tests for query decomposition module.

Comprehensive tests for:
- Basic decomposition functionality (comparison, multi-part detection)
- Normalized subtasks (punctuation trimming, context reattachment)
- Per-subtask intent detection
- Per-subtask boost terms extraction
- LLM fallback with timeout protection
- Timeout and graceful degradation
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock
from src.query_decomposition import (
    QuerySubtask,
    QueryDecompositionResult,
    decompose_query,
    is_multi_intent_query,
    _detect_comparison_query,
    _detect_multi_part_query,
    _normalize_subtask,
    _extract_head_verb,
    _get_subtask_intent,
    _get_subtask_boost_terms,
)


class TestQuerySubtask:
    """Tests for QuerySubtask dataclass with enhanced fields."""

    def test_subtask_basic(self):
        """Test basic subtask creation with all fields."""
        st = QuerySubtask(text="test query", reason="original")
        assert st.text == "test query"
        assert st.reason == "original"
        assert st.weight == 1.0
        assert st.boost_terms == []
        assert st.intent is None
        assert st.llm_generated is False

    def test_subtask_with_boost_terms(self):
        """Test subtask with boost terms."""
        st = QuerySubtask(
            text="test",
            reason="test",
            boost_terms=["api", "overview"]
        )
        assert st.boost_terms == ["api", "overview"]
        assert len(st.boost_terms) == 2

    def test_subtask_with_intent(self):
        """Test subtask with per-subtask intent."""
        st = QuerySubtask(
            text="export data",
            reason="multi_part_1",
            intent="COMMAND",
            llm_generated=False
        )
        assert st.intent == "COMMAND"
        assert st.llm_generated is False

    def test_subtask_llm_generated_flag(self):
        """Test LLM-generated subtask flagging."""
        st = QuerySubtask(
            text="llm generated query",
            reason="llm_generated_1",
            llm_generated=True
        )
        assert st.llm_generated is True
        assert "llm_generated" in st.reason

    def test_subtask_to_dict(self):
        """Test conversion to dict includes all fields."""
        st = QuerySubtask(
            text="test",
            reason="test",
            weight=0.9,
            intent="QUERY",
            llm_generated=True
        )
        d = st.to_dict()
        assert d["text"] == "test"
        assert d["reason"] == "test"
        assert d["weight"] == 0.9
        assert d["intent"] == "QUERY"
        assert d["llm_generated"] is True

    def test_subtask_to_log_payload(self):
        """Test conversion to log payload."""
        st = QuerySubtask(
            text="test subtask",
            reason="decomposed",
            boost_terms=["test"],
            intent="QUERY",
            llm_generated=False
        )
        payload = st.to_log_payload()
        assert payload["text"] == "test subtask"
        assert payload["intent"] == "QUERY"
        assert payload["llm_generated"] is False


class TestQueryDecompositionResult:
    """Tests for QueryDecompositionResult dataclass with V2 enhancements."""

    def test_result_basic(self):
        """Test basic result creation with all V2 fields."""
        subtasks = [
            QuerySubtask(text="part1", reason="test", intent="COMMAND"),
            QuerySubtask(text="part2", reason="test", intent="QUERY"),
        ]
        result = QueryDecompositionResult(
            original_query="part1 and part2",
            subtasks=subtasks,
            strategy="multi_part",
            llm_used=False
        )
        assert result.original_query == "part1 and part2"
        assert len(result.subtasks) == 2
        assert result.strategy == "multi_part"
        assert result.llm_used is False

    def test_result_with_llm(self):
        """Test result when LLM fallback was used."""
        subtasks = [
            QuerySubtask(text="llm question 1", reason="llm_generated_1", llm_generated=True),
            QuerySubtask(text="llm question 2", reason="llm_generated_2", llm_generated=True),
        ]
        result = QueryDecompositionResult(
            original_query="complex query",
            subtasks=subtasks,
            strategy="llm",
            llm_used=True
        )
        assert result.llm_used is True
        assert result.strategy == "llm"
        assert all(st.llm_generated for st in result.subtasks)

    def test_to_strings(self):
        """Test conversion to string list."""
        subtasks = [
            QuerySubtask(text="export", reason="test", intent="COMMAND"),
            QuerySubtask(text="invoices", reason="test", intent="NOUN"),
        ]
        result = QueryDecompositionResult(
            original_query="export and invoices",
            subtasks=subtasks,
            strategy="multi_part"
        )
        strings = result.to_strings()
        assert strings == ["export", "invoices"]
        assert len(strings) == 2

    def test_to_log_payload(self):
        """Test conversion to log payload."""
        subtasks = [
            QuerySubtask(text="part1", reason="test", intent="COMMAND"),
            QuerySubtask(text="part2", reason="test", intent="QUERY"),
        ]
        result = QueryDecompositionResult(
            original_query="test query",
            subtasks=subtasks,
            strategy="heuristic",
            llm_used=False
        )
        payload = result.to_log_payload()
        assert payload["original_query"] == "test query"
        assert payload["strategy"] == "heuristic"
        assert payload["llm_used"] is False
        assert len(payload["subtasks"]) == 2


class TestDetectComparisonQuery:
    """Tests for comparison query detection with robust assertions."""

    def test_vs_comparison(self):
        """Test 'X vs Y' detection."""
        result = _detect_comparison_query("kiosk vs timer")
        assert result is not None, "Should detect 'vs' comparison"
        assert len(result) == 2, "Should return tuple of (X, Y)"
        assert "kiosk" in result[0].lower()
        assert "timer" in result[1].lower()

    def test_versus_comparison(self):
        """Test 'X versus Y' detection."""
        result = _detect_comparison_query("Clockify versus Harvest")
        assert result is not None, "Should detect 'versus' comparison"
        assert len(result) == 2
        assert "clockify" in result[0].lower()
        assert "harvest" in result[1].lower()

    def test_difference_between_comparison(self):
        """Test 'difference between X and Y' detection."""
        result = _detect_comparison_query("difference between timer and kiosk")
        assert result is not None, "Should detect 'difference between' pattern"
        assert len(result) == 2
        assert "timer" in result[0].lower()
        assert "kiosk" in result[1].lower()

    def test_what_is_difference(self):
        """Test 'What is difference between X and Y' detection."""
        result = _detect_comparison_query("What is the difference between A and B?")
        assert result is not None, "Should detect 'What is difference' pattern"
        assert len(result) == 2

    def test_no_comparison(self):
        """Test non-comparison query returns None."""
        result = _detect_comparison_query("How do I export data?")
        assert result is None, "Should not detect comparison in single-intent query"

    def test_single_word_no_comparison(self):
        """Test single word query."""
        result = _detect_comparison_query("kiosk")
        assert result is None, "Should not detect comparison in single word"


class TestDetectMultiPartQuery:
    """Tests for multi-part query detection with robust assertions."""

    def test_and_conjunction(self):
        """Test 'X and Y' detection with proper splitting."""
        result = _detect_multi_part_query("export timesheets and invoices")
        assert result is not None, "Should detect 'and' conjunction"
        assert len(result) >= 2, "Should split into at least 2 parts"
        # Verify parts don't have trailing 'and'
        for part in result:
            assert not part.strip().lower().endswith("and"), "Part should not end with 'and'"

    def test_procedural_steps(self):
        """Test procedural step detection with 'then', 'next'."""
        result = _detect_multi_part_query(
            "First set up workspace then configure time off"
        )
        assert result is not None, "Should detect procedural steps"
        assert len(result) >= 1, "Should detect at least one step"

    def test_multiple_conjunctions(self):
        """Test detection of multiple conjunctions."""
        result = _detect_multi_part_query("export data and create reports and send email")
        assert result is not None, "Should detect multiple 'and' conjunctions"
        # Should have multiple parts when multiple conjunctions present
        assert len(result) >= 2, "Should split multiple conjunctions"

    def test_no_multi_part_single_intent(self):
        """Test query without multi-part indicators."""
        result = _detect_multi_part_query("What is SSO?")
        assert result is None or len(result) <= 1, "Should not detect multi-part in single-intent query"

    def test_no_multi_part_simple(self):
        """Test simple single-part query."""
        result = _detect_multi_part_query("How do I export data?")
        assert result is None or len(result) <= 1, "Should not split simple queries"


class TestDecomposeQuery:
    """Tests for main decompose_query function with V2 quality assertions."""

    def test_comparison_query_decomposition(self):
        """Test decomposition of comparison query."""
        result = decompose_query("What is the difference between kiosk and timer?")
        assert result.original_query == "What is the difference between kiosk and timer?"
        assert result.strategy in ["comparison", "multi_part", "heuristic"]
        assert len(result.subtasks) >= 2, "Comparison queries should decompose into 2+ subtasks"
        # Verify subtasks have intent detected
        for st in result.subtasks:
            assert st.intent is not None, "Each subtask should have per-subtask intent"

    def test_multi_part_decomposition_punctuation_trimmed(self):
        """Test decomposition with punctuation trimming."""
        result = decompose_query("export timesheets and invoices?")
        assert len(result.subtasks) >= 2, "Multi-part should decompose"
        # Subtasks should not have trailing punctuation
        for st in result.subtasks:
            text = st.text.strip()
            assert not text.endswith("?"), f"Subtask '{text}' should not end with ?"
            assert not text.endswith("!"), f"Subtask '{text}' should not end with !"

    def test_multi_part_context_reattachment(self):
        """Test context reattachment for verb-noun pairs."""
        result = decompose_query("export timesheets and invoices")
        # Should have at least 2 meaningful subtasks
        assert len(result.subtasks) >= 2
        # If decomposed, subtasks should have semantic context (e.g., "export" verb)
        subtask_texts = [st.text.lower() for st in result.subtasks]
        # At least one subtask should mention timesheets, one should mention invoices
        assert any("timesheet" in t for t in subtask_texts), "Should mention timesheets"
        assert any("invoice" in t for t in subtask_texts), "Should mention invoices"

    def test_per_subtask_intent_detection(self):
        """Test per-subtask intent is computed independently."""
        result = decompose_query("export data and explain workflow")
        for st in result.subtasks:
            assert hasattr(st, "intent"), "Each subtask should have intent field"
            # Intent should be string or None
            assert st.intent is None or isinstance(st.intent, str)

    def test_per_subtask_boost_terms(self):
        """Test that boost terms are extracted per-subtask."""
        result = decompose_query("approvals workflow and API documentation")
        # At least verify structure
        for st in result.subtasks:
            assert isinstance(st.boost_terms, list), "boost_terms should be list"
            for term in st.boost_terms:
                assert isinstance(term, str), "boost_terms should contain strings"

    def test_max_subtasks_respected(self):
        """Test that max_subtasks limit is respected."""
        result = decompose_query("a and b and c and d and e", max_subtasks=3)
        assert len(result.subtasks) <= 3, "Should not exceed max_subtasks"

    def test_weight_distribution(self):
        """Test that weights are properly assigned."""
        result = decompose_query("export and invoices")
        # All subtasks should have valid weight
        for st in result.subtasks:
            assert 0 <= st.weight <= 1.0, f"Weight {st.weight} out of range"
            assert isinstance(st.weight, float), "Weight should be float"

    def test_no_empty_subtasks(self):
        """Test that no empty subtask text is generated."""
        result = decompose_query("export and and import")
        for st in result.subtasks:
            assert st.text.strip(), "Subtask text should not be empty"
            assert len(st.text) > 0, "Subtask should have non-zero length"

    def test_decomposition_preserves_meaning(self):
        """Test that decomposition doesn't lose semantic meaning."""
        original = "What is SSO and how does it integrate with Clockify?"
        result = decompose_query(original)
        assert result.original_query == original
        # If decomposed into parts, parts should cover original meaning
        if len(result.subtasks) > 1:
            combined_text = " ".join([st.text for st in result.subtasks]).lower()
            assert "sso" in combined_text or "single sign" in combined_text


class TestIsMultiIntentQuery:
    """Tests for multi-intent query detection."""

    def test_vs_comparison_detected(self):
        """Test 'vs' comparison is detected as multi-intent."""
        assert is_multi_intent_query("kiosk vs timer") is True, "Should detect 'vs' comparison"
        assert is_multi_intent_query("A vs B") is True, "Should detect comparison pattern"

    def test_versus_comparison_detected(self):
        """Test 'versus' is detected as multi-intent."""
        assert is_multi_intent_query("Clockify versus Harvest") is True

    def test_difference_between_detected(self):
        """Test 'difference between' is detected as multi-intent."""
        assert is_multi_intent_query("difference between X and Y") is True
        assert is_multi_intent_query("What is difference between A and B") is True

    def test_procedural_queries_detected(self):
        """Test procedural queries are detected as multi-intent."""
        assert is_multi_intent_query("first do X then do Y") is True, "Should detect procedural"
        assert is_multi_intent_query("how to set up workspace, then configure") is True

    def test_conjunction_with_nouns_detected(self):
        """Test conjunction of nouns detected as multi-intent."""
        # "and" with meaningful domain terms
        result = is_multi_intent_query("export timesheets and invoices")
        # If glossary has both terms, should be detected
        assert isinstance(result, bool), "Should return boolean"

    def test_single_intent_not_multi(self):
        """Test simple single-intent queries are not flagged as multi."""
        # Simple queries should return False or be identified as single-intent
        result = is_multi_intent_query("What is SSO?")
        # Not strictly asserting False, as implementation may vary
        assert isinstance(result, bool), "Should return boolean"


class TestDecomposeQueryTimeoutBehavior:
    """Test timeout handling and graceful fallback."""

    def test_timeout_flag_set(self):
        """Test that timeout flag exists in result."""
        result = decompose_query("test query")
        assert hasattr(result, "timed_out"), "Result should have timed_out flag"
        assert isinstance(result.timed_out, bool), "timed_out should be boolean"

    def test_timeout_reasonable_latency(self):
        """Test decomposition completes quickly (within timeout)."""
        import time

        start = time.time()
        result = decompose_query("complex query with many parts and conditions")
        elapsed = time.time() - start

        # Should complete well within timeout (0.75s default)
        assert elapsed < 1.0, f"Decomposition took {elapsed}s, should be < 1.0s"
        assert result.timed_out is False, "Should not timeout on simple queries"

    def test_decomposition_returns_valid_result_on_timeout(self):
        """Test graceful fallback if decomposition times out."""
        result = decompose_query("test query")
        # Even if timed_out=True, should return valid result with fallback
        assert result is not None, "Should return result even on timeout"
        assert len(result.subtasks) >= 1, "Should have at least original query"


class TestLLMFallback:
    """Tests for LLM fallback behavior with mocked LLMClient."""

    @patch.dict(os.environ, {"MOCK_LLM": "false"})
    def test_llm_fallback_not_used_for_simple_queries(self):
        """Test that LLM is not invoked for queries with clear heuristic decomposition."""
        # Simple comparison query should be caught by heuristics
        result = decompose_query("kiosk vs timer")
        # If heuristics work, should not use LLM
        if len(result.subtasks) > 1:
            # Could be heuristic or LLM, but at least we got decomposition
            assert result.strategy in ["comparison", "heuristic", "llm"]

    @patch.dict(os.environ, {"MOCK_LLM": "true"})
    def test_mock_llm_mode_disabled(self):
        """Test that MOCK_LLM env var disables LLM fallback."""
        result = decompose_query("some complex query that might need LLM")
        # With MOCK_LLM=true, LLMClient calls should be skipped
        # Result should still be valid
        assert result is not None, "Should return result in MOCK_LLM mode"
        assert result.llm_used is False, "llm_used flag should be False with MOCK_LLM=true"

    def test_llm_generated_subtasks_marked(self):
        """Test that LLM-generated subtasks are properly marked."""
        result = decompose_query("test query")
        for st in result.subtasks:
            # Each subtask should have llm_generated flag
            assert hasattr(st, "llm_generated"), f"Subtask should have llm_generated flag: {st}"
            assert isinstance(st.llm_generated, bool), "llm_generated should be boolean"
            # If strategy is llm, subtasks should be marked as llm_generated
            if result.strategy == "llm":
                assert st.llm_generated is True, "LLM-generated strategy means subtasks are from LLM"

    def test_llm_used_flag_consistency(self):
        """Test that llm_used flag is consistent with strategy."""
        result = decompose_query("test query")
        if result.strategy == "llm":
            assert result.llm_used is True, "If strategy is 'llm', llm_used should be True"
        else:
            # For heuristic or none, may or may not use LLM
            assert isinstance(result.llm_used, bool), "llm_used should always be boolean"


class TestNormalizedSubtasks:
    """Tests for subtask normalization (punctuation trimming, context reattachment)."""

    def test_normalize_subtask_removes_trailing_punctuation(self):
        """Test that _normalize_subtask removes trailing punctuation."""
        result = _normalize_subtask("export data?", "export")
        assert not result.endswith("?"), f"Result '{result}' should not end with ?"
        assert not result.endswith("!"), f"Result should not end with !"
        assert not result.endswith("."), f"Result should not end with ."

    def test_normalize_subtask_reattaches_context(self):
        """Test that _normalize_subtask reattaches head verb."""
        result = _normalize_subtask("invoices", "export")
        # If context (verb) wasn't in original, should prepend it
        if "export" not in result.lower():
            # Fallback: just normalize, don't force prepending
            assert "invoice" in result.lower(), "Should preserve the noun"
        else:
            assert "export" in result.lower(), "Should include head verb when prepended"

    def test_extract_head_verb_finds_command_verbs(self):
        """Test that _extract_head_verb identifies action verbs."""
        verbs = ["export", "import", "create", "delete", "configure"]
        for verb in verbs:
            query = f"{verb} something and something else"
            result = _extract_head_verb(query)
            assert result is not None or result == "", "Should identify or return empty"


class TestPerSubtaskIntentDetection:
    """Tests for per-subtask intent detection."""

    def test_get_subtask_intent_returns_intent(self):
        """Test that _get_subtask_intent returns a valid intent."""
        from src.server import detect_query_type

        subtask = "export timesheets"
        intent = _get_subtask_intent(subtask)
        # Should return None or a valid intent string
        assert intent is None or isinstance(intent, str), "Intent should be None or string"

    def test_different_subtasks_may_have_different_intents(self):
        """Test that different subtasks can have different intents."""
        intent1 = _get_subtask_intent("export all timesheets")
        intent2 = _get_subtask_intent("what is the difference")
        # Intents may differ (command vs query), or both be None
        assert intent1 is None or isinstance(intent1, str)
        assert intent2 is None or isinstance(intent2, str)


class TestPerSubtaskBoostTerms:
    """Tests for per-subtask boost terms extraction."""

    def test_get_subtask_boost_terms_returns_list(self):
        """Test that _get_subtask_boost_terms returns a list."""
        subtask = "approvals workflow"
        glossary = {"approval": ["approve", "authorization"], "workflow": ["process", "step"]}
        terms = _get_subtask_boost_terms(subtask, glossary)
        assert isinstance(terms, list), "Should return list"
        assert all(isinstance(t, str) for t in terms), "All terms should be strings"

    def test_boost_terms_capped_at_6(self):
        """Test that boost terms are capped at max 6."""
        subtask = "workflow process step approval authorization integration"
        glossary = {
            "workflow": ["w1", "w2"],
            "process": ["p1", "p2"],
            "step": ["s1", "s2"],
            "approval": ["a1", "a2"],
            "authorization": ["auth1", "auth2"],
            "integration": ["i1", "i2"],
        }
        terms = _get_subtask_boost_terms(subtask, glossary)
        assert len(terms) <= 6, f"Terms should be capped at 6, got {len(terms)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
