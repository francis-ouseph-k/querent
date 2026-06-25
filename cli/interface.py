"""
cli/interface.py
─────────────────
Terminal-based chat interface for the NL→SQL pipeline.

Features:
  - Rich-formatted output (tables, syntax-highlighted SQL, coloured metadata)
  - Readline-style input with history (prompt_toolkit)
  - Dry-run / execute mode toggle (:dry / :exec)
  - Feedback capture (:correct feeds the Phase 2 training flywheel)
  - Debug mode (:debug shows retrieval internals + token counts)
  - Ambiguity handling — interactive clarification loop (see _handle_ambiguous)
  - Incomplete query handling — free-text value prompt for missing parameters

DISAMBIGUATION LOOP
────────────────────
When query_understanding._detect_ambiguity() flags a query, pipeline/runner.py
returns QueryResult(error="ambiguous_query", explanation=<list of options>)
without calling the LLM at all (~0ms vs ~1-2s for an LLM clarification call).

_run_query() detects this result and calls _handle_ambiguous(), which presents
one of two interactions depending on option content:

  Pattern A — numbered menu (options do NOT start with "INCOMPLETE")
    User picks a number → chosen option appended to original query as
    "— specifically: <option>" → query rerun automatically

  Pattern B — free-text prompt (at least one option starts with "INCOMPLETE")
    System explains what parameter is missing → user types a value →
    value appended to original query as "— value: <value>" → query rerun

The REFINED_MARKER ("— specifically:") and VALUE_MARKER ("— value:") are
checked by _detect_ambiguity() to prevent the clarified query from
re-triggering ambiguity detection.

Example runtime flow:
  User:    "show me students who failed in board 5"
  System:  "⚠ INCOMPLETE — pass threshold not in schema.
            Specify percentage: e.g. 'students who scored below 40%'"
  Input:   "below 40%"
  Refined: "show me students who failed in board 5 — value: below 40%"
  → SQL generated correctly with result.final_marks < total_marks * 0.4
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

# prompt_toolkit requires a real Windows Console (cmd.exe / Windows Terminal).
# In VS Code integrated terminal, PowerShell ISE, or piped stdout it raises
# NoConsoleScreenBufferError. Import so we can catch it and fall back to
# plain input() without crashing the session.
try:
    from prompt_toolkit.output.win32 import NoConsoleScreenBufferError
except ImportError:
    # Non-Windows or older prompt_toolkit versions don't have this class
    NoConsoleScreenBufferError = OSError

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from config.settings import settings
from generation.query_understanding import INCOMPLETE_PREFIX, REFINED_MARKER, VALUE_MARKER
from mcp_tools.client import call_corpus_save_correction, MCPCallError
from models.schema import QueryResult
from utils.logging_config import get_logger

logger = get_logger(__name__)

_HISTORY_FILE = ".nl_sql_history"

_PROMPT_STYLE = Style.from_dict({
    "prompt":  "ansicyan bold",
    "rprompt": "ansiblue",
})

_HELP_TEXT = """\
[bold cyan]Commands[/bold cyan]
  [cyan]:help[/cyan]        show this help
  [cyan]:dry[/cyan]         switch to dry-run mode  (validate SQL, do not execute)
  [cyan]:exec[/cyan]        switch to execute mode  (run against read-only replica)
  [cyan]:debug[/cyan]       toggle debug mode       (show full retrieval chunks)
  [cyan]:correct[/cyan]     provide a corrected SQL for the last failed query
  [cyan]:clear[/cyan]       clear the screen
  [cyan]:quit[/cyan]  / :q  exit

[bold cyan]Bare shortcuts (no colon needed)[/bold cyan]
  [cyan]cls[/cyan]          clear the screen
  [cyan]exit[/cyan] / [cyan]quit[/cyan]  exit the session

[bold cyan]Tips[/bold cyan]
  - Dry-run is ON by default - safe to experiment.
  - Start with [cyan]python main.py --debug[/cyan] to enable debug mode from launch.
  - In debug mode, every query prints Dense / BM25 / Graph / Final chunks.
  - Ambiguous queries trigger a clarification menu - pick a number or type a value.
  - Use exact status codes: FROZEN, NOT_ASSIGNED, ELIGIBLE, WITHHELD.
  - Multi-word entities: "evaluation attempt", "question paper", "rate card".
  - Use :correct after every wrong result - feeds the Phase 2 training corpus.
"""


class ChatInterface:
    """
    Terminal chat UI for the NL→SQL pipeline.

    Usage:
        ui = ChatInterface(runner=pipeline_runner)
        ui.run()
    """

    def __init__(self, runner, debug_mode: bool = False) -> None:  # runner: PipelineRunner
        self.runner     = runner
        self.console    = Console()
        self.dry_run    = settings.dry_run_default
        self.debug_mode = debug_mode or settings.debug_mode
        self.last_result: QueryResult | None = None

        try:
            self.session = PromptSession(
                history=FileHistory(_HISTORY_FILE),
                style=_PROMPT_STYLE,
            )
        except NoConsoleScreenBufferError:
            # VS Code terminal / piped stdout — fall back to plain input().
            # History and styling are lost but the CLI stays functional.
            self.session = None

    # ─────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """REPL loop — read query, dispatch to _run_query or _handle_command."""
        self._print_banner()

        while True:
            try:
                mode_tag  = "[DRY]" if self.dry_run else "[EXEC]"
                raw_input = self._prompt(f"\n{mode_tag} ❯ ")
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[dim]Goodbye.[/dim]")
                break

            text = raw_input.strip()
            if not text:
                continue

            # ── Bare convenience commands (no colon prefix needed) ─────────
            if text.lower() in ("cls", "clear"):
                self.console.clear()
                self._print_banner()
                continue
            if text.lower() in ("exit", "quit", "q"):
                self.console.print("\n[dim]Goodbye.[/dim]")
                break

            if text.startswith(":"):
                self._handle_command(text)
            else:
                self._run_query(text)

    # ─────────────────────────────────────────────────────────────────────
    # Query execution
    # ─────────────────────────────────────────────────────────────────────

    def _run_query(self, query: str) -> None:
        """
        Execute one query through the pipeline and display the result.
        If the result is ambiguous/incomplete, handle the clarification loop
        before showing any output.
        """
        with self.console.status("[bold cyan]Thinking…[/bold cyan]", spinner="dots"):
            try:
                result = self.runner.run(query, dry_run=self.dry_run)
            except Exception as exc:
                logger.exception("pipeline_exception")
                self.console.print(f"[red]Pipeline error:[/red] {exc}")
                return

        # ── Ambiguity / incomplete query — enter clarification loop ───────
        # result.explanation is a list[str] when ambiguous (set by runner.py).
        # We loop to handle cases where the refined query is itself ambiguous
        # (unlikely but possible if the user's chosen option is still vague).
        # Max 3 rounds prevents infinite loops.
        rounds = 0
        while (
            result.error == "ambiguous_query"
            and isinstance(result.explanation, list)
            and rounds < 3
        ):
            refined_result = self._handle_ambiguous(result, query)
            if refined_result is None:
                # User chose "rephrase manually" or cancelled
                self.console.print("[dim]Enter your rephrased query.[/dim]")
                return
            query  = refined_result.nl_query   # track refined query for next loop
            result = refined_result
            rounds += 1

        self.last_result = result
        self.display_result(result)

    # ─────────────────────────────────────────────────────────────────────
    # Ambiguity handling
    # ─────────────────────────────────────────────────────────────────────

    def _handle_ambiguous(
        self, result: QueryResult, original_query: str
    ) -> QueryResult | None:
        """
        Present clarification options and rerun with the user's choice.

        Two interaction modes, auto-detected from option content:

        Pattern B — INCOMPLETE (free-text value needed):
          Triggered when ANY option starts with INCOMPLETE_PREFIX.
          Shows the explanation, prompts for a free-text value.
          Appends "— value: <value>" to the original query and reruns.

        Pattern A — Menu pick (multiple valid interpretations):
          Shows numbered options. User picks one.
          Appends "— specifically: <option>" to the original query and reruns.

        Returns:
          QueryResult from the refined query, or
          None if the user cancelled or chose "rephrase manually".
        """
        options = result.explanation  # list[str] set by runner.py

        # ── Pattern B: at least one INCOMPLETE option ─────────────────────
        incomplete_options = [o for o in options if o.startswith(INCOMPLETE_PREFIX)]
        if incomplete_options:
            # Show the first INCOMPLETE message — explains what's missing
            msg = incomplete_options[0]
            # Strip "INCOMPLETE — " prefix for display
            display_msg = msg[len(INCOMPLETE_PREFIX):].lstrip(" —").strip()
            self.console.print(f"\n[yellow][!] Incomplete query:[/yellow] {display_msg}\n")

            # Also show any concrete options (non-INCOMPLETE) as hints
            concrete = [o for o in options if not o.startswith(INCOMPLETE_PREFIX)]
            if concrete:
                self.console.print("[dim]Common values:[/dim]")
                for opt in concrete:
                    self.console.print(f"  [dim]- {opt}[/dim]")
                self.console.print()

            # Prompt for free-text value
            try:
                value = self._prompt("Provide value (or press Enter to rephrase): ").strip()
            except (KeyboardInterrupt, EOFError):
                return None

            if not value:
                return None

            # Append VALUE_MARKER so _detect_ambiguity skips this refined query
            refined_query = f"{original_query} {VALUE_MARKER} {value}"
            self.console.print(f"\n[dim]Refining with: {value}[/dim]\n")

            with self.console.status("[bold cyan]Thinking…[/bold cyan]", spinner="dots"):
                return self.runner.run(refined_query, dry_run=self.dry_run)

        # ── Pattern A: numbered menu ───────────────────────────────────────
        self.console.print("\n[yellow][!] Ambiguous query - please clarify:[/yellow]\n")

        for i, opt in enumerate(options, 1):
            # Wrap long options for readability
            self.console.print(f"  [cyan]{i}.[/cyan] {opt}")

        # Always offer a manual rephrase option at the end
        rephrase_num = len(options) + 1
        self.console.print(f"  [cyan]{rephrase_num}.[/cyan] Rephrase manually\n")

        try:
            raw = self._prompt(f"Choose [1–{rephrase_num}]: ").strip()
        except (KeyboardInterrupt, EOFError):
            return None

        try:
            choice = int(raw)
        except ValueError:
            # Non-numeric input — treat as cancel
            return None

        if choice == rephrase_num:
            return None

        if not 1 <= choice <= len(options):
            self.console.print("[red]Invalid choice.[/red]")
            return None

        # Append REFINED_MARKER so _detect_ambiguity skips this refined query
        chosen        = options[choice - 1]
        refined_query = f"{original_query} {REFINED_MARKER} {chosen}"
        self.console.print(f"\n[dim]Refining: {chosen}[/dim]\n")

        with self.console.status("[bold cyan]Thinking…[/bold cyan]", spinner="dots"):
            return self.runner.run(refined_query, dry_run=self.dry_run)

    # ─────────────────────────────────────────────────────────────────────
    # Result display
    # ─────────────────────────────────────────────────────────────────────

    def display_result(self, r: QueryResult) -> None:
        """
        Render the full QueryResult to the terminal.
        Public (not _private) so main.py can call it in non-interactive mode.
        """
        # ── Meta row ──────────────────────────────────────────────────────
        conf_colour = (
            "green"  if r.confidence >= 0.7 else
            "yellow" if r.confidence >= 0.5 else
            "red"
        )
        meta_parts = [Text(f"Intent: {r.intent}", style="dim")]
        if r.confidence > 0:
            meta_parts.append(Text(f"Confidence: {r.confidence:.0%}", style=conf_colour))
        if r.tables_used:
            meta_parts.append(Text(f"Tables: {', '.join(r.tables_used)}", style="dim"))
        if r.dry_run:
            meta_parts.append(Text("DRY RUN", style="bold yellow"))
        if r.retries:
            meta_parts.append(Text(f"Retries: {r.retries}", style="yellow"))

        self.console.print(Columns(meta_parts, equal=False, expand=False))

        # ── Error ─────────────────────────────────────────────────────────
        if not r.success:
            # Don't display the raw error for ambiguous queries —
            # _handle_ambiguous already handled the interaction.
            if r.error != "ambiguous_query":
                self.console.print(
                    Panel(r.error, title="[red]Error[/red]",
                          border_style="red", expand=False)
                )
            return

        # ── Confidence warning ─────────────────────────────────────────────
        if r.confidence < settings.confidence_warn_threshold:
            self.console.print(
                f"[yellow][!] Low confidence ({r.confidence:.0%}). "
                f"Review the SQL carefully.[/yellow]"
            )

        # ── SQL ───────────────────────────────────────────────────────────
        if r.sql and settings.show_sql_in_cli:
            self.console.print(
                Panel(
                    Syntax(r.sql, "sql", theme="monokai",
                           line_numbers=False, word_wrap=True),
                    title="SQL",
                    border_style="cyan",
                    expand=False,
                )
            )

        # ── Explanation ───────────────────────────────────────────────────
        if r.explanation and isinstance(r.explanation, str) \
                and settings.show_explanation_in_cli:
            self.console.print(f"[dim]{r.explanation}[/dim]")

        # ── Results table ─────────────────────────────────────────────────
        if r.rows:
            self._display_table(r.rows, r.row_count)
        elif not r.dry_run:
            self.console.print("[dim]No rows returned.[/dim]")
        elif r.dry_run:
            self.console.print("[yellow]Dry run — SQL validated but not executed.[/yellow]")

        # ── Debug internals ───────────────────────────────────────────────
        if self.debug_mode and r.retrieval_meta:
            self._display_debug(r)

        # ── Timings ───────────────────────────────────────────────────────
        if r.latency_ms.get("total_ms"):
            token_info = ""
            if self.debug_mode and r.retrieval_meta:
                ptok = r.retrieval_meta.get("llm_prompt_tokens")
                ctok = r.retrieval_meta.get("llm_completion_tokens")
                if ptok is not None or ctok is not None:
                    token_info = f"  |  tokens: {ptok or '?'} input, {ctok or '?'} output"

            self.console.print(
                f"[dim]Latency: {r.latency_ms.get('total_ms')}ms total"
                f"  (retrieval {r.latency_ms.get('retrieval_ms', '?')}ms"
                f"  | generation {r.latency_ms.get('generation_ms', '?')}ms){token_info}[/dim]"
            )

    def _display_table(self, rows: list[dict], total: int) -> None:
        """Render list-of-dicts as a Rich table, capped at 100 display rows."""
        if not rows:
            return

        table = Table(
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold cyan",
            row_styles=["", "dim"],
            expand=False,
        )
        columns = list(rows[0].keys())
        for col in columns:
            table.add_column(col, overflow="fold", max_width=40)
        for row in rows[:100]:
            table.add_row(*[str(row.get(c, "")) for c in columns])

        self.console.print(table)

        row_line = f"{total} row{'s' if total != 1 else ''}"
        if total > 100:
            row_line += " (showing first 100)"
        self.console.print(f"[dim]{row_line}[/dim]")

    def _display_debug(self, r: QueryResult) -> None:
        """
        Full debug panel — shown after every query when debug mode is ON.

        Sections displayed:
          1. Dense (Qdrant) chunks   — raw hits before RRF, with cosine score
          2. BM25 (OpenSearch) chunks — raw hits before RRF, with BM25 score
          3. Graph / Join data        — Steiner tables + join path strings
          4. Final prompt chunks      — post-budget chunks in prompt order,
                                        with RRF score and token estimate
        """
        m = r.retrieval_meta
        if not m:
            self.console.print("[dim](no retrieval metadata)[/dim]")
            return

        RULE  = "=" * 70
        DIVID = "-" * 70

        def _section(title: str, subtitle: str = "") -> None:
            self.console.print(f"\n[bold cyan]{RULE}[/bold cyan]")
            self.console.print(f"[bold cyan]{title}[/bold cyan]")
            if subtitle:
                self.console.print(f"[dim]{subtitle}[/dim]")
            self.console.print(f"[bold cyan]{RULE}[/bold cyan]")

        def _chunk_divider(n: int) -> None:
            self.console.print(f"[dim]{DIVID}[/dim]")
            self.console.print(f"[dim]Chunk {n}[/dim]")
            self.console.print(f"[dim]{DIVID}[/dim]")

        # ── 1. Dense (Qdrant) hits ────────────────────────────────────────
        dense_hits = m.get("_debug_dense_hits", [])
        _section(
            f"DENSE VECTOR (Qdrant)  —  {len(dense_hits)} chunks",
            f"Latency: {m.get('qdrant_dense_ms', '?')}ms  |  Metric: Cosine Similarity",
        )
        if dense_hits:
            for i, hit in enumerate(dense_hits, 1):
                _chunk_divider(i)
                chunk_type  = hit.get("chunk_type", "?")
                table_name  = hit.get("table_name", "")
                chunk_id    = hit.get("chunk_id", "")
                score       = hit.get("score", 0.0)
                text        = hit.get("text", "")
                tokens      = hit.get("tokens", "?")
                self.console.print(
                    f"  [yellow]Type:[/yellow]        {chunk_type}\n"
                    f"  [yellow]Table:[/yellow]       {table_name or '(none)'}\n"
                    f"  [yellow]Chunk ID:[/yellow]    {chunk_id}\n"
                    f"  [yellow]Cosine Sim:[/yellow]  [{'green' if score >= 0.7 else 'yellow' if score >= 0.5 else 'red'}]{score:.4f}[/{'green' if score >= 0.7 else 'yellow' if score >= 0.5 else 'red'}]\n"
                    f"  [yellow]~Tokens:[/yellow]     {tokens}"
                )
                self.console.print(f"\n[dim]{text[:600]}{'…' if len(text) > 600 else ''}[/dim]")
        else:
            self.console.print("[dim](not available — run with --debug to capture)[/dim]")

        # ── 2. BM25 (OpenSearch) hits ─────────────────────────────────────
        bm25_hits = m.get("_debug_bm25_hits", [])
        _section(
            f"SPARSE / BM25 (OpenSearch)  —  {len(bm25_hits)} chunks",
            f"Latency: {m.get('opensearch_bm25_ms', '?')}ms  |  Metric: BM25 Score",
        )
        if bm25_hits:
            for i, hit in enumerate(bm25_hits, 1):
                _chunk_divider(i)
                chunk_type  = hit.get("chunk_type", "?")
                table_name  = hit.get("table_name", "")
                chunk_id    = hit.get("chunk_id", "")
                score       = hit.get("score", 0.0)
                text        = hit.get("text", "")
                tokens      = hit.get("tokens", "?")
                self.console.print(
                    f"  [yellow]Type:[/yellow]        {chunk_type}\n"
                    f"  [yellow]Table:[/yellow]       {table_name or '(none)'}\n"
                    f"  [yellow]Chunk ID:[/yellow]    {chunk_id}\n"
                    f"  [yellow]BM25 Score:[/yellow]  {score:.4f}\n"
                    f"  [yellow]~Tokens:[/yellow]     {tokens}"
                )
                self.console.print(f"\n[dim]{text[:600]}{'…' if len(text) > 600 else ''}[/dim]")
        else:
            self.console.print("[dim](not available — run with --debug to capture)[/dim]")

        # ── 3. Graph / Join data ──────────────────────────────────────────
        graph_tables = m.get("graph_tables", [])
        join_paths   = m.get("_debug_join_paths") or m.get("join_paths", [])
        _section(
            f"GRAPH  —  Steiner Tree tables: {len(graph_tables)}  |  Join paths: {len(join_paths)}",
            f"Latency: {m.get('graph_ms', '?')}ms  |  Algorithm: Steiner Tree (global minimal join path)",
        )
        if graph_tables:
            self.console.print(f"  [yellow]Steiner Tables:[/yellow]  {', '.join(graph_tables)}")
        if join_paths:
            self.console.print()
            for i, path in enumerate(join_paths, 1):
                self.console.print(f"  [cyan]{i:>2}.[/cyan] [dim]{path}[/dim]")
        if not graph_tables and not join_paths:
            self.console.print("[dim](no graph data — no entity tables detected)[/dim]")

        # ── 4. Final prompt chunks (post-budget) ──────────────────────────
        final_chunks = m.get("_debug_final_chunks", [])
        _section(
            f"FINAL PROMPT CHUNKS  —  {len(final_chunks)} chunks  |  "
            f"Tokens used: {m.get('tokens_used', '?')} / {m.get('effective_budget', '?')}",
            f"RRF k={settings.retrieval.rrf_k if hasattr(settings, 'retrieval') else 60}  |  "
            f"Reranker: {'applied' if m.get('reranker_applied') else 'not applied'}",
        )
        if final_chunks:
            for i, fc in enumerate(final_chunks, 1):
                _chunk_divider(i)
                rrf   = fc.get("rrf_score", 0.0)
                toks  = fc.get("tokens", "?")
                text  = fc.get("text", "")
                mandatory = rrf >= 999.0
                rrf_label = "[bold magenta]MANDATORY (entity seed)[/bold magenta]" if mandatory \
                            else f"[{'green' if rrf >= 0.02 else 'yellow' if rrf >= 0.01 else 'red'}]{rrf:.4f}[/{'green' if rrf >= 0.02 else 'yellow' if rrf >= 0.01 else 'red'}]"
                self.console.print(
                    f"  [yellow]Type:[/yellow]        {fc.get('chunk_type', '?')}\n"
                    f"  [yellow]Table:[/yellow]       {fc.get('table_name', '') or '(none)'}\n"
                    f"  [yellow]Chunk ID:[/yellow]    {fc.get('chunk_id', '')}\n"
                    f"  [yellow]RRF Score:[/yellow]   {rrf_label}\n"
                    f"  [yellow]~Tokens:[/yellow]     {toks}"
                )
                self.console.print(f"\n[dim]{text[:800]}{'…' if len(text) > 800 else ''}[/dim]")
        else:
            self.console.print("[dim](not available — run with --debug to capture)[/dim]")

        # ── Summary footer ────────────────────────────────────────────────
        self.console.print(f"\n[bold cyan]{RULE}[/bold cyan]")
        llm_usage_str = ""
        ptok = m.get("llm_prompt_tokens")
        ctok = m.get("llm_completion_tokens")
        if ptok is not None or ctok is not None:
            llm_usage_str = f"  ·  LLM Input: {ptok or '?'} tokens  ·  LLM Output: {ctok or '?'} tokens"

        self.console.print(
            f"[dim]DEBUG SUMMARY  |  "
            f"Dense: {m.get('qdrant_dense_hits', '?')} hits ({m.get('qdrant_dense_ms', '?')}ms)  ·  "
            f"BM25: {m.get('opensearch_bm25_hits', '?')} hits ({m.get('opensearch_bm25_ms', '?')}ms)  ·  "
            f"RRF fusion: {m.get('rrf_ms', '?')}ms  ·  "
            f"Budget: {m.get('budget_ms', '?')}ms  ·  "
            f"Total retrieval: {m.get('total_ms', '?')}ms{llm_usage_str}[/dim]"
        )
        self.console.print(f"[bold cyan]{RULE}[/bold cyan]\n")

    # ─────────────────────────────────────────────────────────────────────
    # Command handlers
    # ─────────────────────────────────────────────────────────────────────

    def _handle_command(self, text: str) -> None:
        cmd = text.lower().strip()

        if cmd in (":quit", ":q", ":exit"):
            raise EOFError

        elif cmd == ":help":
            self.console.print(Panel(_HELP_TEXT, title="Help", border_style="cyan"))

        elif cmd == ":dry":
            self.dry_run = True
            self.console.print("[yellow]Mode: DRY RUN — SQL validated, not executed.[/yellow]")

        elif cmd == ":exec":
            self.dry_run = False
            self.console.print("[green]Mode: EXECUTE — queries run on read-only replica.[/green]")

        elif cmd == ":debug":
            self.debug_mode    = not self.debug_mode
            settings.debug_mode = self.debug_mode
            self.console.print(f"[dim]Debug mode: {'ON' if self.debug_mode else 'OFF'}[/dim]")

        elif cmd == ":clear":
            self.console.clear()
            self._print_banner()

        elif cmd.startswith(":correct"):
            self._handle_correction(text)

        else:
            self.console.print(
                f"[red]Unknown command: {text}[/red]. "
                f"Type [cyan]:help[/cyan] for commands."
            )

    def _handle_correction(self, _text: str) -> None:
        """
        Capture a corrected SQL for the last query and save to corpus.

        Routes to corpus_server.py MCP (port 5013) when USE_MCP_SERVERS=true,
        falling back to local atomic file write if MCP is unreachable.
        Local path writes directly to failures/ directory (same format).
        """
        if not self.last_result:
            self.console.print("[dim]No previous query to correct.[/dim]")
            return

        self.console.print("Enter corrected SQL (blank line to finish):")
        lines = []
        try:
            while True:
                line = self._prompt("  SQL> ")
                if not line.strip():
                    break
                lines.append(line)
        except (KeyboardInterrupt, EOFError):
            return

        corrected_sql = "\n".join(lines).strip()
        if not corrected_sql:
            return

        if settings.use_mcp_servers:
            # ── MCP path: route to corpus_server.py ───────────────────────
            # entry_id is the stem of the failure filename. It is stored on
            # last_result when the runner logged the failure via corpus MCP.
            entry_id = getattr(self.last_result, "failure_entry_id", None)
            if entry_id:
                try:
                    result = call_corpus_save_correction(
                        entry_id      = entry_id,
                        corrected_sql = corrected_sql,
                        source        = "user_correction",
                    )
                    if result.get("status") == "ok":
                        self.console.print(
                            f"[green]✓ Correction saved via MCP.[/green] "
                            f"[dim](id: {entry_id})[/dim]\n"
                            f"[dim]Included in Phase 2 fine-tuning corpus.[/dim]"
                        )
                        return
                    else:
                        self.console.print(
                            f"[yellow]MCP save returned: {result}. "
                            f"Falling back to local write.[/yellow]"
                        )
                except MCPCallError as exc:
                    self.console.print(
                        f"[yellow]Corpus MCP unreachable ({exc}). "
                        f"Falling back to local write.[/yellow]"
                    )
            else:
                self.console.print(
                    "[yellow]No failure_entry_id on last result. "
                    "Falling back to local write.[/yellow]"
                )
            # Fall through to local write if MCP path failed

        # ── Local path: atomic file write ─────────────────────────────────
        failure_dir = Path(settings.failure_log_dir)
        failure_dir.mkdir(parents=True, exist_ok=True)

        uid   = str(uuid.uuid4())[:8]
        now   = datetime.now(timezone.utc)
        entry = {
            "timestamp":     now.isoformat(),
            "nl_query":      self.last_result.nl_query,
            "failed_sql":    self.last_result.sql,
            "error":         self.last_result.error,
            "corrected_sql": corrected_sql,
            "source":        "user_correction",
        }

        filename     = failure_dir / f"correction_{now.strftime('%Y%m%d_%H%M%S')}_{uid}.json"
        tmp_filename = filename.with_suffix(".tmp")
        tmp_filename.write_text(
            json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(tmp_filename, filename)   # atomic rename

        self.console.print(
            f"[green]✓ Correction saved.[/green] [dim]({filename})[/dim]\n"
            f"[dim]Included in Phase 2 fine-tuning corpus.[/dim]"
        )
        logger.info(component="cli", event="correction_saved", path=str(filename))

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _prompt(self, message: str) -> str:
        """
        Unified prompt helper — uses prompt_toolkit session when available,
        falls back to plain input() for non-console environments.
        Raises EOFError / KeyboardInterrupt on Ctrl+D / Ctrl+C (callers handle).
        """
        if self.session is not None:
            try:
                return self.session.prompt(message)
            except NoConsoleScreenBufferError:
                # Console state changed mid-session — switch permanently
                self.session = None
        return input(message)

    def _print_banner(self) -> None:
        mode  = "[yellow]DRY RUN[/yellow]" if self.dry_run else "[green]EXECUTE[/green]"
        debug = "  | [magenta]DEBUG ON[/magenta]" if self.debug_mode else ""
        self.console.print(
            Panel(
                f"[bold cyan]Digital Evaluation System - NL->SQL[/bold cyan]\n"
                f"Model: Qwen2.5-Coder 3B Q4 | Qdrant + OpenSearch | NetworkX\n"
                f"Mode: {mode}{debug} | Type [cyan]:help[/cyan] for commands",
                border_style="cyan",
                expand=False,
            )
        )