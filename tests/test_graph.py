import unittest
from unittest.mock import patch, MagicMock
import pytest
from core.state import create_initial_state
from core.config import MAX_RESEARCH_LOOPS
from core.graph import route_after_critic, run_research


# ── ROUTING TESTS (no LLM needed) ────────────────────────────────────────────

class TestRouteAfterCritic(unittest.TestCase):
    def test_loops_back_when_gaps_and_under_cap(self):
        state = create_initial_state("test")
        state["needs_more_research"] = True
        state["research_loop_count"] = 0
        self.assertEqual(route_after_critic(state), "researcher")

    def test_proceeds_when_loop_cap_hit(self):
        state = create_initial_state("test")
        state["needs_more_research"] = True
        state["research_loop_count"] = 2
        self.assertEqual(route_after_critic(state), "synthesizer")

    def test_proceeds_when_sufficient(self):
        state = create_initial_state("test")
        state["needs_more_research"] = False
        state["research_loop_count"] = 0
        self.assertEqual(route_after_critic(state), "synthesizer")


# ── LOOP CAP TEST (mocked agents) ────────────────────────────────────────────

class TestLoopCap(unittest.TestCase):
    @patch("core.graph.store_research_session")
    @patch("core.graph.get_research_context", return_value=None)
    @patch("core.graph.synthesizer_agent")
    @patch("core.graph.critic_agent")
    @patch("core.graph.researcher_agent")
    @patch("core.graph.planner_agent")
    def test_researcher_called_at_most_max_plus_one(
        self, mock_planner, mock_researcher, mock_critic, mock_synth,
        mock_get_ctx, mock_store,
    ):
        mock_planner.return_value = {
            "sub_questions": ["q1", "q2", "q3"],
            "current_step": "planning",
        }
        mock_researcher.return_value = {
            "search_results": [
                {"sub_question": "q1", "title": "T", "url": "https://a.com", "content": "c"},
            ],
            "current_step": "researching",
        }

        call_count = {"n": 0}

        def critic_side_effect(state):
            call_count["n"] += 1
            return {
                "critique": "Still gaps.",
                "needs_more_research": True,
                "current_step": "critiquing",
                "research_loop_count": call_count["n"],
            }

        mock_critic.side_effect = critic_side_effect
        mock_synth.return_value = {
            "final_report": "# Report\nDone.",
            "current_step": "synthesizing",
        }

        result = run_research("test query")

        self.assertLessEqual(
            mock_researcher.call_count,
            MAX_RESEARCH_LOOPS + 1,
        )
        self.assertIsNotNone(result.get("final_report"))


# ── INTEGRATION TEST (needs real API keys) ────────────────────────────────────

@pytest.mark.integration
class TestIntegration(unittest.TestCase):
    def test_full_pipeline(self):
        result = run_research("What is photosynthesis?")
        self.assertIsNotNone(result["final_report"])
        self.assertGreaterEqual(len(result["sub_questions"]), 3)
        self.assertIn("## Sources", result["final_report"])


if __name__ == "__main__":
    unittest.main()
