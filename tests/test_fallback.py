import unittest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage
from core.config import call_llm_with_fallback


class TestCallLlmWithFallback(unittest.TestCase):
    @patch("core.config.get_gemini_llm")
    @patch("core.config.get_llm")
    def test_falls_back_to_gemini_on_rate_limit(self, mock_get_llm, mock_get_gemini):
        mock_groq = MagicMock()
        mock_groq.invoke.side_effect = Exception("rate limit 429")
        mock_get_llm.return_value = mock_groq

        mock_gemini = MagicMock()
        mock_gemini.invoke.return_value = MagicMock(content="ok")
        mock_get_gemini.return_value = mock_gemini

        result = call_llm_with_fallback(
            [HumanMessage(content="test")], temperature=0.0
        )

        self.assertEqual(result.content, "ok")
        mock_get_gemini.assert_called_once()

    @patch("core.config.time.sleep")
    @patch("core.config.get_gemini_llm")
    @patch("core.config.get_llm")
    def test_raises_after_max_attempts(self, mock_get_llm, mock_get_gemini, mock_sleep):
        mock_groq = MagicMock()
        mock_groq.invoke.side_effect = Exception("rate limit")
        mock_get_llm.return_value = mock_groq

        mock_gemini = MagicMock()
        mock_gemini.invoke.side_effect = Exception("gemini down")
        mock_get_gemini.return_value = mock_gemini

        with self.assertRaises(RuntimeError):
            call_llm_with_fallback(
                [HumanMessage(content="test")], temperature=0.0, max_attempts=2
            )

    @patch("core.config.get_gemini_llm")
    @patch("core.config.get_llm")
    def test_groq_success_skips_gemini(self, mock_get_llm, mock_get_gemini):
        mock_groq = MagicMock()
        mock_groq.invoke.return_value = MagicMock(content="groq response")
        mock_get_llm.return_value = mock_groq

        result = call_llm_with_fallback(
            [HumanMessage(content="test")], temperature=0.0
        )

        self.assertEqual(result.content, "groq response")
        mock_get_gemini.assert_not_called()


if __name__ == "__main__":
    unittest.main()
