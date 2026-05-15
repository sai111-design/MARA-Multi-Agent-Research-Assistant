import unittest
import memory.vector_store as vs


def _make_state(query="What are the latest advances in quantum computing?"):
    return {
        "query": query,
        "final_report": "# Quantum Computing\nQuantum computers use qubits [1].\n## Sources\n[1] Example — https://example.com",
        "sub_questions": ["What are qubits?", "How do quantum gates work?"],
        "search_results": [
            {"title": "Source 1", "url": "https://example.com/1", "content": "content"},
            {"title": "Source 2", "url": "https://example.com/2", "content": "content"},
        ],
    }


class TestQdrantMemory(unittest.TestCase):
    def setUp(self):
        vs._qdrant_client = None
        vs._chroma_client = None
        vs._using_chroma = False

    def test_store_returns_id(self):
        state = _make_state()
        point_id = vs.store_research_session(state)
        self.assertIsNotNone(point_id)
        self.assertIsInstance(point_id, str)

    def test_search_finds_similar(self):
        state = _make_state()
        vs.store_research_session(state)

        results = vs.search_past_research(
            "What are the latest advances in quantum computing?", limit=3
        )
        self.assertGreaterEqual(len(results), 1)
        self.assertGreaterEqual(results[0]["score"], 0.7)

    def test_search_rejects_unrelated(self):
        state = _make_state()
        vs.store_research_session(state)

        results = vs.search_past_research("recipe for chocolate cake", limit=3)
        self.assertEqual(len(results), 0)


class TestChromaFallback(unittest.TestCase):
    def setUp(self):
        vs._qdrant_client = None
        vs._chroma_client = None
        vs._using_chroma = True

    def tearDown(self):
        vs._using_chroma = False
        vs._chroma_client = None

    def test_store_via_chroma(self):
        state = _make_state()
        point_id = vs.store_research_session(state)
        self.assertIsNotNone(point_id)

    def test_search_via_chroma(self):
        state = _make_state()
        vs.store_research_session(state)

        results = vs.search_past_research(
            "quantum computing advances", limit=3
        )
        self.assertIsInstance(results, list)


if __name__ == "__main__":
    unittest.main()
