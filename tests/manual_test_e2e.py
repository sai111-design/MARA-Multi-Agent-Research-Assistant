import time
from core.graph import run_research

start = time.time()
result = run_research("What are the latest breakthroughs in fusion energy?")
elapsed = time.time() - start

print(f"Completed in {elapsed:.1f}s")
print(f"Sub-questions: {len(result['sub_questions'])}")
print(f"Search results: {len(result['search_results'])}")
print(f"Research loops: {result['research_loop_count']}")
print(f"Report length: {len(result.get('final_report', ''))} chars")
print("\n--- REPORT ---\n")
print(result.get("final_report", "No report generated"))
