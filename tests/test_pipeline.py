import pytest
from unittest.mock import patch, MagicMock
from graph.pipeline import build_pipeline
from graph.state import MARAState

MOCK_STATE: MARAState = {
    "query": "What is quantum computing?",
    "sub_questions": [],
    "search_results": [],
    "critique": None,
    "needs_more_research": False,
    "final_report": None,
    "research_loop_count": 0,
    "current_step": "",
}

def test_pipeline_builds():
    pipeline = build_pipeline()
    assert pipeline is not None

@patch("agents.planner.ChatGroq")
@patch("agents.researcher.TavilyClient")
@patch("agents.critic.ChatGroq")
@patch("agents.synthesizer.ChatGroq")
def test_pipeline_end_to_end(mock_synth, mock_critic, mock_tavily, mock_planner):
    mock_planner.return_value.invoke = MagicMock(return_value=MagicMock(content="1. What is quantum computing?"))
    mock_tavily.return_value.search = MagicMock(return_value={"results": [{"title": "Test", "url": "http://test.com", "content": "Test content"}]})
    mock_critic.return_value.invoke = MagicMock(return_value=MagicMock(content="Research looks good.\nSUFFICIENT"))
    mock_synth.return_value.invoke = MagicMock(return_value=MagicMock(content="# Report\nQuantum computing uses qubits. [1]\n\n## Sources\n[1] http://test.com"))

    pipeline = build_pipeline()
    result = pipeline.invoke(MOCK_STATE)
    assert result["final_report"] is not None
    assert result["current_step"] == "synthesizer"
