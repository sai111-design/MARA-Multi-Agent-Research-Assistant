import unittest
from unittest.mock import patch, MagicMock
from core.state import create_initial_state
from agents.critic import critic_agent


def _make_state(verdict_text, loop_count=0):
    state = create_initial_state("test query")
    state["sub_questions"] = ["q1", "q2"]
    state["search_results"] = [
        {"sub_question": "q1", "title": "T1", "url": "https://a.com", "content": "content1"},
        {"sub_question": "q2", "title": "T2", "url": "https://b.com", "content": "content2"},
    ]
    state["research_loop_count"] = loop_count
    return state


class TestCriticAgent(unittest.TestCase):
    @patch("agents.critic.call_llm_with_fallback")
    def test_sufficient_verdict(self, mock_llm_call):
        mock_llm_call.return_value = MagicMock(
            content="All sub-questions are covered.\n\nVERDICT: SUFFICIENT"
        )

        state = _make_state("sufficient")
        result = critic_agent(state)

        self.assertFalse(result["needs_more_research"])
        self.assertEqual(result["current_step"], "critiquing")
        self.assertIn("SUFFICIENT", result["critique"])
        self.assertEqual(result["research_loop_count"], 1)

    @patch("agents.critic.call_llm_with_fallback")
    def test_insufficient_verdict(self, mock_llm_call):
        mock_llm_call.return_value = MagicMock(
            content="Missing coverage on economics.\n\nVERDICT: INSUFFICIENT"
        )

        state = _make_state("insufficient", loop_count=0)
        result = critic_agent(state)

        self.assertTrue(result["needs_more_research"])
        self.assertEqual(result["research_loop_count"], 1)

    @patch("agents.critic.call_llm_with_fallback")
    def test_insufficient_capped_at_max_loops(self, mock_llm_call):
        mock_llm_call.return_value = MagicMock(
            content="Still gaps.\n\nVERDICT: INSUFFICIENT"
        )

        state = _make_state("insufficient", loop_count=2)
        result = critic_agent(state)

        self.assertFalse(result["needs_more_research"])
        self.assertEqual(result["research_loop_count"], 3)

    @patch("agents.critic.call_llm_with_fallback")
    def test_loop_count_increments(self, mock_llm_call):
        mock_llm_call.return_value = MagicMock(
            content="Looks good.\n\nVERDICT: SUFFICIENT"
        )

        state = _make_state("sufficient", loop_count=1)
        result = critic_agent(state)

        self.assertEqual(result["research_loop_count"], 2)

    @patch("agents.critic.call_llm_with_fallback")
    def test_empty_results_handled(self, mock_llm_call):
        mock_llm_call.return_value = MagicMock(
            content="No results at all.\n\nVERDICT: INSUFFICIENT"
        )

        state = create_initial_state("test")
        state["sub_questions"] = ["q1"]
        state["search_results"] = []

        result = critic_agent(state)

        self.assertTrue(result["needs_more_research"])


if __name__ == "__main__":
    unittest.main()
