from core.state import create_initial_state
from agents.planner import planner_agent
from agents.researcher import researcher_agent
from agents.critic import critic_agent
from agents.synthesizer import synthesizer_agent

state = create_initial_state("What is the current state of quantum computing?")
state.update(planner_agent(state))
state.update(researcher_agent(state))
state.update(critic_agent(state))
result = synthesizer_agent(state)

report = result["final_report"]
total_sources = len(state["search_results"])

print(f"Report length: {len(report)} chars")
print(f"Total sources: {total_sources}")
print(f"Has Sources section: {'## Sources' in report}")
print("\n--- FIRST 1000 CHARS ---\n")
print(report[:1000])
