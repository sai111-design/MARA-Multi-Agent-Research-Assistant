from core.state import create_initial_state
from agents.planner import planner_agent
from agents.researcher import researcher_agent

state = create_initial_state("What are the latest advances in quantum computing?")
state.update(planner_agent(state))
result = researcher_agent(state)
print(f"Collected {len(result['search_results'])} search results")
for r in result["search_results"]:
    print(f"  [{r['title'][:50]}] {r['url']}")
