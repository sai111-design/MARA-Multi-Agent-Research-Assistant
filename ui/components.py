import streamlit as st

def render_progress(step: str) -> None:
    steps = ["planner", "researcher", "critic", "synthesizer"]
    cols = st.columns(len(steps))
    for i, s in enumerate(steps):
        icon = "✅" if steps.index(step) > i else ("🔄" if s == step else "⬜")
        cols[i].markdown(f"{icon} **{s.capitalize()}**")

def render_report(report: str) -> None:
    if not report:
        return
    st.divider()
    st.subheader("Research Report")
    st.markdown(report)
    st.download_button(
        label="Download Report (.md)",
        data=report,
        file_name="mara_report.md",
        mime="text/markdown",
    )

def render_sources(search_results: list[dict]) -> None:
    if not search_results:
        return
    with st.expander(f"Sources ({len(search_results)})"):
        for i, r in enumerate(search_results):
            st.markdown(f"**[{i+1}] [{r['title']}]({r['url']})**")
            st.caption(r["content"])
