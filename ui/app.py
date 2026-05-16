"""
MARA — Multi-Agent Research Assistant
Gradio UI  (replaces the Streamlit implementation)

Run locally:
    python ui/app.py
    # or
    gradio ui/app.py

Deploys identically on Hugging Face Spaces (SDK: gradio, port 7860).
"""

import os
import sys
import time
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Shim: newer huggingface_hub removed HfFolder, but some gradio builds still
# import it. Inject a minimal stand-in before gradio is loaded so the import
# resolves without error.
import huggingface_hub as _hfhub
if not hasattr(_hfhub, "HfFolder"):
    class _HfFolder:
        @staticmethod
        def get_token():
            try:
                return _hfhub.get_token()
            except Exception:
                return None
        @staticmethod
        def save_token(token): pass
        @staticmethod
        def delete_token(): pass
    _hfhub.HfFolder = _HfFolder

import gradio as gr

# Patch Gradio's OAuth setup to be non-fatal.
# huggingface_hub>=0.30 removed APIs that gradio[oauth]==5.0.0 uses at runtime,
# causing attach_oauth() to raise an exception that blocks ALL API route
# registration — producing "No API found" in the frontend.
try:
    import gradio.oauth as _go
    _orig_attach = getattr(_go, "attach_oauth", None)
    if _orig_attach:
        def _safe_attach_oauth(app):
            try:
                _orig_attach(app)
            except Exception as _e:
                logging.getLogger(__name__).warning(
                    "[MARA] OAuth setup skipped (huggingface_hub compat): %s", _e
                )
        _go.attach_oauth = _safe_attach_oauth
except Exception:
    pass

from core.graph import run_research_stream
from core.config import PRIMARY_MODEL, FALLBACK_MODEL, GROQ_API_KEY, TAVILY_API_KEY, GOOGLE_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

AGENT_META = {
    "planner": ("🧠", "Planner", "Decomposing your query into sub-questions"),
    "researcher": ("🔍", "Researcher", "Searching the web for evidence"),
    "critic": ("🔎", "Critic", "Evaluating research completeness"),
    "synthesizer": ("✍️", "Synthesizer", "Writing the final cited report"),
}

WAITING_MD = "\n".join(
    f"⬜ {icon} **{label}** — waiting"
    for icon, label, _ in AGENT_META.values()
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Global ── */
body, .gradio-container {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    background: #0f1117 !important;
    color: #e2e8f0 !important;
}

/* ── Header ── */
.mara-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.mara-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.4rem;
    letter-spacing: -0.02em;
}
.mara-subtitle {
    font-size: 1rem;
    color: #94a3b8;
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Pipeline status panel ── */
.pipeline-panel {
    background: #1e2130;
    border: 1px solid #2d3148;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    font-size: 0.95rem;
    line-height: 2;
}

/* ── Report output ── */
.report-panel {
    background: #1a1d2e;
    border: 1px solid #2d3148;
    border-radius: 12px;
    padding: 1.8rem;
    font-size: 0.95rem;
    line-height: 1.75;
    max-height: 70vh;
    overflow-y: auto;
}
.report-panel h1 { color: #e2e8f0; font-size: 1.5rem; margin-bottom: 1rem; }
.report-panel h2 { color: #c4b5fd; font-size: 1.15rem; margin-top: 1.4rem;
                   border-bottom: 1px solid #2d3148; padding-bottom: 0.3rem; }
.report-panel h3 { color: #93c5fd; font-size: 1rem; }
.report-panel a  { color: #67e8f9; text-decoration: underline; }

/* ── Metrics strip ── */
.metrics-strip {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 0.8rem;
}
.metric-card {
    flex: 1;
    min-width: 120px;
    background: #1e2130;
    border: 1px solid #2d3148;
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    text-align: center;
}
.metric-label { font-size: 0.72rem; color: #64748b; text-transform: uppercase;
                letter-spacing: 0.08em; }
.metric-value { font-size: 1.5rem; font-weight: 700; color: #a78bfa;
                margin-top: 0.15rem; }

/* ── Accordion panels ── */
.gr-accordion { background: #1e2130 !important; border: 1px solid #2d3148 !important;
                border-radius: 10px !important; margin-bottom: 0.5rem !important; }

/* ── Input + button ── */
.gr-textbox textarea, .gr-textbox input {
    background: #1e2130 !important;
    border: 1px solid #2d3148 !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
}
.gr-textbox textarea:focus, .gr-textbox input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}
.gr-button.primary {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    transition: opacity 0.2s ease !important;
}
.gr-button.primary:hover { opacity: 0.88 !important; }
.gr-button.secondary {
    background: #1e2130 !important;
    border: 1px solid #2d3148 !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
}

/* ── History table ── */
.history-table { font-size: 0.88rem; }

/* ── Sidebar info ── */
.info-card {
    background: #1e2130;
    border: 1px solid #2d3148;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.88rem;
    color: #94a3b8;
    line-height: 1.7;
}
.info-card strong { color: #c4b5fd; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f1117; }
::-webkit-scrollbar-thumb { background: #2d3148; border-radius: 4px; }
"""


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _metrics_html(elapsed: float, source_count: int, loop_count: int) -> str:
    return f"""
<div class="metrics-strip">
  <div class="metric-card">
    <div class="metric-label">⏱ Time</div>
    <div class="metric-value">{elapsed:.1f}s</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">📄 Sources</div>
    <div class="metric-value">{source_count}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">🔁 Loops</div>
    <div class="metric-value">{loop_count}</div>
  </div>
</div>
"""


def _pipeline_html(statuses: dict) -> str:
    """
    Renders the live pipeline status panel.
    statuses = {"planner": "waiting"|"running"|"done"|"error", ...}
    """
    STATE_ICON = {
        "waiting": "⬜",
        "running": "🟡",
        "done": "✅",
        "error": "❌",
    }
    STATE_COLOR = {
        "waiting": "#64748b",
        "running": "#fbbf24",
        "done": "#34d399",
        "error": "#f87171",
    }
    lines = []
    for key, (icon, label, desc) in AGENT_META.items():
        state = statuses.get(key, "waiting")
        s_icon = STATE_ICON[state]
        color = STATE_COLOR[state]
        suffix = f" — {desc}…" if state == "running" else (
            " — complete" if state == "done" else (
                " — error" if state == "error" else " — waiting"))
        lines.append(
            f'<div style="color:{color}; padding: 0.15rem 0;">'
            f'{s_icon} {icon} <strong>{label}</strong>{suffix}</div>'
        )
    return f'<div class="pipeline-panel">{"".join(lines)}</div>'


def _sources_md(search_results: list) -> str:
    seen: dict = {}
    for r in search_results:
        url = r.get("url", "")
        if url and url not in seen:
            seen[url] = r.get("title", url)
    if not seen:
        return "*No sources collected yet.*"
    lines = [f"- [{title}]({url})" for url, title in seen.items()]
    return "\n".join(lines)


def _sub_questions_md(sub_questions: list) -> str:
    if not sub_questions:
        return "*No sub-questions generated yet.*"
    return "\n".join(f"**{i}.** {q}" for i, q in enumerate(sub_questions, 1))


# ── CORE RESEARCH FUNCTION (generator — feeds Gradio streaming) ───────────────

def run_research_gradio(query: str, history: list):
    """
    Gradio generator function. Yields tuples of UI component updates
    after each agent completes, enabling live streaming in the interface.

    Yields:
        (pipeline_html, report_md, metrics_html,
         sub_q_md, sources_md, critique_md,
         updated_history, status_msg)
    """
    if not query or not query.strip():
        yield (
            _pipeline_html({}),
            "",
            "",
            "*Enter a query first.*",
            "*Enter a query first.*",
            "*Enter a query first.*",
            history,
            "⚠️ Please enter a research question.",
        )
        return

    # ── Initial state ─────────────────────────────────────────────────────────
    statuses = {k: "waiting" for k in AGENT_META}
    sub_questions = []
    search_results = []
    critique = ""
    final_report = ""
    start_time = time.time()
    error_msg = None
    loop_count = 0

    # Emit "all waiting" immediately so the UI updates before work starts
    yield (
        _pipeline_html(statuses),
        "*Research starting…*",
        "",
        "*Waiting for Planner…*",
        "*Waiting for Researcher…*",
        "*Waiting for Critic…*",
        history,
        "🔍 Research in progress…",
    )

    try:
        for event in run_research_stream(query):
            for node_name, node_output in event.items():
                if node_name not in AGENT_META:
                    continue

                # Mark this node as running
                statuses[node_name] = "running"
                yield (
                    _pipeline_html(statuses),
                    final_report or "*Generating…*",
                    "",
                    _sub_questions_md(sub_questions),
                    _sources_md(search_results),
                    critique or "*Waiting for Critic…*",
                    history,
                    f"🟡 {AGENT_META[node_name][1]} running…",
                )

                # Accumulate state from the node output
                if isinstance(node_output, dict):
                    if "sub_questions" in node_output:
                        sub_questions = node_output["sub_questions"]
                    if "search_results" in node_output:
                        for r in node_output["search_results"]:
                            if r not in search_results:
                                search_results.append(r)
                    if "critique" in node_output and node_output["critique"]:
                        critique = node_output["critique"]
                        if "INSUFFICIENT" in critique.upper():
                            loop_count += 1
                    if "final_report" in node_output and node_output["final_report"]:
                        final_report = node_output["final_report"]

                # Mark this node as done
                statuses[node_name] = "done"
                yield (
                    _pipeline_html(statuses),
                    final_report or "*Generating…*",
                    "",
                    _sub_questions_md(sub_questions),
                    _sources_md(search_results),
                    critique or "*Waiting for Critic…*",
                    history,
                    f"✅ {AGENT_META[node_name][1]} complete",
                )

    except Exception as exc:
        error_msg = str(exc)
        logger.error(f"[UI] Pipeline error: {exc}")
        for k in statuses:
            if statuses[k] == "running":
                statuses[k] = "error"

    # ── Final yield ───────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    source_count = len({r.get("url") for r in search_results if r.get("url")})

    if error_msg:
        status_msg = f"❌ Error: {error_msg}"
        final_report = final_report or (
            f"> ⚠️ Pipeline failed: {error_msg}\n\n"
            + (_sources_md(search_results) if search_results else "No results collected.")
        )
    else:
        status_msg = f"✅ Done in {elapsed:.1f}s — {source_count} sources consulted"

    # Append to history
    updated_history = history + [[
        query,
        f"{elapsed:.1f}s",
        str(source_count),
        str(loop_count),
    ]]

    yield (
        _pipeline_html(statuses),
        final_report,
        _metrics_html(elapsed, source_count, loop_count),
        _sub_questions_md(sub_questions),
        _sources_md(search_results),
        critique or "*No gaps were identified — research was sufficient.*",
        updated_history,
        status_msg,
    )


# ── CLEAR HANDLER ─────────────────────────────────────────────────────────────

def clear_all():
    return (
        "",  # query_input
        _pipeline_html({}),  # pipeline_status
        "",  # report_output
        "",  # metrics_output
        "*Enter a query above.*",  # sub_q_output
        "*Enter a query above.*",  # sources_output
        "*Enter a query above.*",  # critique_output
        "Ready.",  # status_bar
    )


# ── GRADIO BLOCKS LAYOUT ──────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(
        css=CUSTOM_CSS,
        title="MARA — Multi-Agent Research Assistant",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.violet,
            secondary_hue=gr.themes.colors.cyan,
            neutral_hue=gr.themes.colors.slate,
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
        ),
    ) as app:

        # ── Session state (persists within a browser session) ─────────────────
        history_state = gr.State([])   # list of [query, time, sources, loops]

        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="mara-header">
          <div class="mara-title">🔬 MARA</div>
          <div class="mara-subtitle">
            Multi-Agent Research Assistant — transform any research question into
            a comprehensive, cited report using a pipeline of specialised AI agents.
          </div>
        </div>
        """)

        # ── API key warnings ──────────────────────────────────────────────────
        missing_keys = []
        if not GROQ_API_KEY:
            missing_keys.append(
                "<b>GROQ_API_KEY</b> — required for the LLM pipeline "
                "(<a href='https://console.groq.com/keys' target='_blank' style='color:#67e8f9'>get one free at console.groq.com</a>)"
            )
        if not TAVILY_API_KEY:
            missing_keys.append(
                "<b>TAVILY_API_KEY</b> — optional, enables deep web search "
                "(<a href='https://app.tavily.com' target='_blank' style='color:#67e8f9'>app.tavily.com</a>); "
                "falls back to DuckDuckGo automatically"
            )
        if not GOOGLE_API_KEY:
            missing_keys.append(
                "<b>GOOGLE_API_KEY</b> — optional, enables Gemini fallback LLM "
                "(<a href='https://aistudio.google.com/app/apikey' target='_blank' style='color:#67e8f9'>aistudio.google.com</a>)"
            )

        if missing_keys:
            items_html = "".join(f"<li style='margin:0.3rem 0'>{k}</li>" for k in missing_keys)
            severity = "error" if not GROQ_API_KEY else "warning"
            border_color = "#f87171" if severity == "error" else "#fbbf24"
            gr.HTML(f"""
            <div style="background:#1e2130;border:1px solid {border_color};border-radius:10px;
                        padding:1rem 1.4rem;margin-bottom:0.5rem;font-size:0.9rem;color:#e2e8f0;">
              <strong style="color:{border_color}">
                {"❌ Required API key missing — queries will fail" if severity == "error"
                 else "⚠️ Optional keys not set"}
              </strong>
              <ul style="margin:0.5rem 0 0 1rem;padding:0">{items_html}</ul>
              <p style="margin:0.6rem 0 0;color:#94a3b8;font-size:0.82rem;">
                Add secrets in your Space →
                <b>Settings → Repository secrets</b>, then restart the Space.
              </p>
            </div>
            """)

        # ── Main layout: left sidebar + right content ─────────────────────────
        with gr.Row(equal_height=False):

            # ── LEFT SIDEBAR ─────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=260):

                gr.HTML(f"""
                <div class="info-card">
                  <strong>Agent Pipeline</strong><br>
                  🧠 Planner — decomposes your query<br>
                  🔍 Researcher — searches the web<br>
                  🔎 Critic — checks for gaps<br>
                  ✍️ Synthesizer — writes the report<br><br>
                  <strong>Primary LLM:</strong>
                  <code style="color:#67e8f9">{PRIMARY_MODEL}</code><br>
                  <strong>Fallback LLM:</strong>
                  <code style="color:#67e8f9">{FALLBACK_MODEL}</code><br>
                  <strong>Search:</strong>
                  <code style="color:#67e8f9">Tavily API</code>
                </div>
                """)

                gr.Markdown("### 📂 Session History")
                history_table = gr.Dataframe(
                    headers=["Query", "Time", "Sources", "Loops"],
                    datatype=["str", "str", "str", "str"],
                    value=[],
                    interactive=False,
                    elem_classes=["history-table"],
                )

            # ── RIGHT MAIN CONTENT ────────────────────────────────────────────
            with gr.Column(scale=3):

                # Query input row
                with gr.Row():
                    query_input = gr.Textbox(
                        placeholder="e.g. What are the environmental and economic"
                                    " impacts of electric vehicles?",
                        label="Research Question",
                        lines=1,
                        scale=5,
                    )
                    research_btn = gr.Button(
                        "🔍 Research", variant="primary", scale=1, min_width=130
                    )

                with gr.Row():
                    clear_btn = gr.Button("🗑 Clear", variant="secondary", scale=1)
                    status_bar = gr.Markdown("Ready.")

                # Pipeline status
                pipeline_status = gr.HTML(
                    value=_pipeline_html({}),
                    label="Agent Pipeline",
                )

                # Metrics strip (hidden until research completes)
                metrics_output = gr.HTML(value="")

                # Final report
                report_output = gr.Markdown(
                    value="",
                    label="📄 Research Report",
                    elem_classes=["report-panel"],
                )

                # Download buttons
                with gr.Row():
                    download_md = gr.DownloadButton(
                        "📥 Download .md", visible=False, variant="secondary"
                    )
                    download_pdf = gr.DownloadButton(
                        "📄 Download PDF", visible=False, variant="secondary"
                    )

                # Expandable detail panels
                with gr.Accordion("📋 Sub-Questions Generated", open=False,
                                  elem_classes=["gr-accordion"]):
                    sub_q_output = gr.Markdown("*Enter a query above.*")

                with gr.Accordion("📚 Sources Referenced", open=False,
                                  elem_classes=["gr-accordion"]):
                    sources_output = gr.Markdown("*Enter a query above.*")

                with gr.Accordion("🔎 Critic's Assessment", open=False,
                                  elem_classes=["gr-accordion"]):
                    critique_output = gr.Markdown("*Enter a query above.*")

        # ── WIRING ────────────────────────────────────────────────────────────

        # All outputs that the generator yields, in order
        generator_outputs = [
            pipeline_status,
            report_output,
            metrics_output,
            sub_q_output,
            sources_output,
            critique_output,
            history_state,
            status_bar,
        ]

        # Research button click
        research_btn.click(
            fn=run_research_gradio,
            inputs=[query_input, history_state],
            outputs=generator_outputs,
            show_progress=False,
        )

        # Also trigger on Enter key in the textbox
        query_input.submit(
            fn=run_research_gradio,
            inputs=[query_input, history_state],
            outputs=generator_outputs,
            show_progress=False,
        )

        # Sync history state to the visible table
        history_state.change(
            fn=lambda h: h,
            inputs=[history_state],
            outputs=[history_table],
        )

        # Show download buttons and wire files after report appears
        def _make_md_file(report: str, query: str) -> tuple:
            """Creates .md temp file and shows download buttons."""
            import tempfile
            if not report or report.startswith("*"):
                return gr.update(visible=False), gr.update(visible=False)
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".md", delete=False, encoding="utf-8"
            )
            tmp.write(f"# {query}\n\n{report}")
            tmp.close()
            return (
                gr.update(visible=True, value=tmp.name),
                gr.update(visible=True),
            )

        report_output.change(
            fn=_make_md_file,
            inputs=[report_output, query_input],
            outputs=[download_md, download_pdf],
        )

        # PDF download — calls pdf_export helper
        def _make_pdf_file(report: str, query: str):
            try:
                from ui.pdf_export import markdown_to_pdf_bytes
                import tempfile
                pdf_bytes = markdown_to_pdf_bytes(report, query)
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".pdf", delete=False
                )
                tmp.write(pdf_bytes)
                tmp.close()
                return gr.update(visible=True, value=tmp.name)
            except Exception as e:
                logger.warning(f"[UI] PDF generation failed: {e}")
                return gr.update(visible=False)

        download_pdf.click(
            fn=_make_pdf_file,
            inputs=[report_output, query_input],
            outputs=[download_pdf],
        )

        # Clear button
        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[
                query_input,
                pipeline_status,
                report_output,
                metrics_output,
                sub_q_output,
                sources_output,
                critique_output,
                status_bar,
            ],
        )

    return app


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

app = build_app()
app.launch(
    server_name="0.0.0.0",
    server_port=7860,
    show_error=True,
    share=False,
)
