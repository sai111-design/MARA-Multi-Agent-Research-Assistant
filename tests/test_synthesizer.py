import unittest
from unittest.mock import patch, MagicMock
from core.state import create_initial_state
from agents.synthesizer import (
    synthesizer_agent,
    build_human_message,
    generate_fallback_report,
)


def _make_state(n_results=2):
    state = create_initial_state("What is quantum computing?")
    state["sub_questions"] = ["What are qubits?", "What are quantum gates?"]
    state["search_results"] = [
        {
            "sub_question": f"q{i}",
            "title": f"Source {i}",
            "url": f"https://example.com/{i}",
            "content": f"Content about topic {i}",
        }
        for i in range(1, n_results + 1)
    ]
    state["critique"] = "Coverage is adequate."
    return state


VALID_REPORT = (
    "# Quantum Computing Explained\n"
    "## Overview\nQuantum computing uses qubits [1].\n"
    "## Background\nDetails here [2].\n"
    "## Key Takeaways\n- Qubits are powerful [1].\n"
    "## Sources\n"
    "[1] Source 1 — https://example.com/1\n"
    "[2] Source 2 — https://example.com/2"
)

INVALID_REPORT_NO_SOURCES = (
    "Here is some text about quantum computing without proper structure."
)


class TestBuildHumanMessage(unittest.TestCase):
    def test_generates_source_index(self):
        state = _make_state(2)
        msg = build_human_message(state)

        self.assertIn("SOURCE INDEX:", msg)
        self.assertIn("[1] Source 1 | https://example.com/1", msg)
        self.assertIn("[2] Source 2 | https://example.com/2", msg)
        self.assertIn("RESEARCH RESULTS:", msg)
        self.assertIn("--- Source [1] ---", msg)

    def test_duplicate_urls_are_deduplicated(self):
        state = create_initial_state("test")
        state["search_results"] = [
            {"sub_question": "q1", "title": "Page A", "url": "https://a.com", "content": "A"},
            {"sub_question": "q2", "title": "Page B", "url": "https://b.com", "content": "B"},
            {"sub_question": "q3", "title": "Page A dup", "url": "https://a.com", "content": "A again"},
        ]
        state["critique"] = None

        msg = build_human_message(state)

        index_section = msg.split("SOURCE INDEX:\n")[1].split("\n\nRESEARCH RESULTS:")[0]
        index_lines = [l for l in index_section.strip().splitlines() if l.startswith("[")]
        self.assertEqual(len(index_lines), 2)


class TestFallbackReport(unittest.TestCase):
    def test_fallback_report_generation(self):
        state = _make_state(2)
        report = generate_fallback_report(state)

        self.assertIn("# Research Results:", report)
        self.assertIn("synthesis failed", report.lower())
        self.assertIn("Source 1", report)
        self.assertIn("Source 2", report)
        self.assertIn("https://example.com/1", report)

    def test_fallback_with_empty_results(self):
        state = create_initial_state("test")
        state["search_results"] = []
        report = generate_fallback_report(state)

        self.assertIn("No search results", report)
        self.assertTrue(len(report) > 0)


class TestSynthesizerAgent(unittest.TestCase):
    @patch("agents.synthesizer.call_llm_with_fallback")
    def test_synthesizer_success(self, mock_llm):
        mock_llm.return_value = MagicMock(content=VALID_REPORT)
        state = _make_state(2)

        result = synthesizer_agent(state)

        self.assertIn("## Sources", result["final_report"])
        self.assertIn("# Quantum Computing", result["final_report"])
        self.assertIn("[1]", result["final_report"])
        self.assertEqual(result["current_step"], "synthesizing")

    @patch("agents.synthesizer.call_llm_with_fallback")
    def test_synthesizer_fallback_on_invalid_output(self, mock_llm):
        mock_llm.return_value = MagicMock(content=INVALID_REPORT_NO_SOURCES)
        state = _make_state(2)

        result = synthesizer_agent(state)

        self.assertIn("synthesis failed", result["final_report"].lower())
        self.assertIn("Source 1", result["final_report"])
        self.assertEqual(result["current_step"], "synthesizing")

    @patch("agents.synthesizer.call_llm_with_fallback")
    def test_synthesizer_handles_llm_exception(self, mock_llm):
        mock_llm.side_effect = Exception("LLM failure")
        state = _make_state(2)

        result = synthesizer_agent(state)

        self.assertIn("synthesis failed", result["final_report"].lower())
        self.assertIn("Source 1", result["final_report"])
        self.assertIn("Source 2", result["final_report"])
        self.assertEqual(result["current_step"], "synthesizing")


if __name__ == "__main__":
    unittest.main()
