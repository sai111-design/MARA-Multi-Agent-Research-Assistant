import unittest
from unittest.mock import patch, MagicMock
from core.state import create_initial_state
from agents.researcher import researcher_agent, duckduckgo_fallback, _truncate


def _make_tavily_response(n=2):
    return {
        "results": [
            {"title": f"Title {i}", "url": f"https://example.com/{i}", "content": f"Content {i}"}
            for i in range(n)
        ]
    }


class TestTruncation(unittest.TestCase):
    def test_short_text_unchanged(self):
        self.assertEqual(_truncate("hello", 500), "hello")

    def test_long_text_truncated(self):
        text = "a" * 600
        result = _truncate(text, 500)
        self.assertEqual(len(result), 503)  # 500 + "..."
        self.assertTrue(result.endswith("..."))

    def test_exact_length_unchanged(self):
        text = "a" * 500
        self.assertEqual(_truncate(text, 500), text)


class TestResearcherAgent(unittest.TestCase):
    @patch("agents.researcher.TAVILY_API_KEY", "fake-key")
    @patch("agents.researcher._search_tavily")
    def test_normalizes_tavily_results(self, mock_search):
        mock_search.return_value = [
            {"title": "T1", "url": "https://a.com", "content": "C1"},
            {"title": "T2", "url": "https://b.com", "content": "C2"},
        ]
        state = create_initial_state("test")
        state["sub_questions"] = ["q1"]

        result = researcher_agent(state)

        self.assertEqual(len(result["search_results"]), 2)
        self.assertEqual(result["search_results"][0]["sub_question"], "q1")
        self.assertEqual(result["search_results"][0]["title"], "T1")
        self.assertEqual(result["current_step"], "researching")

    @patch("agents.researcher.TAVILY_API_KEY", "fake-key")
    @patch("agents.researcher._search_tavily")
    def test_partial_failure(self, mock_search):
        mock_search.side_effect = [
            Exception("API error"),
            Exception("API error"),  # retry for q1
            [{"title": "T1", "url": "https://a.com", "content": "C1"}],
        ]
        state = create_initial_state("test")
        state["sub_questions"] = ["q1-fail", "q2-ok"]

        result = researcher_agent(state)

        self.assertEqual(len(result["search_results"]), 1)
        self.assertEqual(result["search_results"][0]["sub_question"], "q2-ok")

    @patch("agents.researcher.TAVILY_API_KEY", "fake-key")
    @patch("agents.researcher._search_tavily")
    def test_all_failures_returns_empty(self, mock_search):
        mock_search.side_effect = Exception("boom")
        state = create_initial_state("test")
        state["sub_questions"] = ["q1"]

        result = researcher_agent(state)

        self.assertEqual(result["search_results"], [])
        self.assertEqual(result["current_step"], "researching")

    @patch("agents.researcher.TAVILY_API_KEY", "")
    @patch("agents.researcher.duckduckgo_fallback")
    def test_ddg_fallback_when_no_tavily_key(self, mock_ddg):
        mock_ddg.return_value = [
            {"title": "DDG", "url": "https://ddg.com", "content": "from ddg"},
        ]
        state = create_initial_state("test")
        state["sub_questions"] = ["q1"]

        result = researcher_agent(state)

        mock_ddg.assert_called_once()
        self.assertEqual(result["search_results"][0]["title"], "DDG")

    @patch("agents.researcher.TAVILY_API_KEY", "fake-key")
    @patch("agents.researcher._search_tavily")
    def test_content_truncation(self, mock_search):
        mock_search.return_value = [
            {"title": "T", "url": "https://a.com", "content": "x" * 800},
        ]
        state = create_initial_state("test")
        state["sub_questions"] = ["q1"]

        result = researcher_agent(state)

        content = result["search_results"][0]["content"]
        self.assertEqual(len(content), 503)
        self.assertTrue(content.endswith("..."))


if __name__ == "__main__":
    unittest.main()
