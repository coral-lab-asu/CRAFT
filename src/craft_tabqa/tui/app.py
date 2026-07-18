"""An inline, menu-driven terminal app for interactive retrieval.

Everything is chosen from menus - the user never types configuration from
scratch. The flow is:

    1. Pick a config from ``configs/`` (arrow keys).
    2. Pick the pipeline scope: Stages 1+2, or include Stage 3.
    3. Loop: pick a question from the dataset's query file (or lightly edit one),
       run retrieval, and browse the ranked tables a few at a time.

Full ranked lists are written to a human-readable results file that is opened
once, when the first query's results arrive, so the user can scroll them while
more queries stream in.

Requires the ``[tui]`` extra (``questionary`` + ``rich``).
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

from craft_tabqa.config import load_config
from craft_tabqa.loaders import load_queries
from craft_tabqa.tui.results_file import append_query_results, start_results_file

TABLES_PER_PAGE = 3


def run_tui(configs_dir: str = "configs") -> None:
    """Entry point for ``craft tui``."""
    import questionary
    from rich.console import Console

    console = Console()
    console.rule("[bold]CRAFT[/bold] · interactive retrieval")

    config_path = _pick_config(configs_dir, questionary, console)
    if not config_path:
        return
    cfg = load_config(config_path)
    _apply_hardware_env(cfg)

    scope = _pick_scope(cfg, questionary, console)
    if scope is None:
        return
    use_stage3 = scope == "stage3"

    engine = _load_engine(cfg, console)
    if engine is None:
        return

    questions = _load_question_pool(cfg, console)
    results_path = str(Path(cfg.data.output_dir) / "tui_results.txt")
    results_started = False
    top_k = cfg.stage3.top_k if use_stage3 else cfg.stage2.top_k

    while True:
        question = _pick_question(questions, questionary)
        if question is None:
            console.print("[dim]Done.[/dim]")
            break

        console.print(f"\n[bold cyan]Q:[/bold cyan] {question}")
        with console.status("Retrieving…"):
            results = engine.search(question, top_k=top_k, use_stage3=use_stage3)

        if not results_started:
            start_results_file(results_path, cfg.data.dataset, scope_label(use_stage3))
            results_started = True
            _open_file(results_path, console)
        append_query_results(results_path, question, results)

        _browse_results(results, console, questionary)
        console.print(f"[dim]Full ranking appended to {results_path}[/dim]")


# ---------------------------------------------------------------------------
# Menu steps
# ---------------------------------------------------------------------------

def _pick_config(configs_dir: str, questionary, console) -> Optional[str]:
    configs = sorted(str(p) for p in Path(configs_dir).glob("*.yaml"))
    if not configs:
        console.print(f"[red]No .yaml configs found in {configs_dir}/[/red]")
        return None
    return questionary.select("Choose a dataset config:", choices=configs).ask()


def _pick_scope(cfg, questionary, console) -> Optional[str]:
    choices = [
        questionary.Choice("Stages 1+2  (local, no API cost)", value="stage2"),
    ]
    if cfg.stage3.enabled:
        choices.append(
            questionary.Choice(
                f"Include Stage 3  ({cfg.stage3.provider} API, reranks to top {cfg.stage3.top_k})",
                value="stage3",
            )
        )
    else:
        console.print("[dim]Stage 3 is disabled in this config (no API key); offering Stages 1+2 only.[/dim]")
    return questionary.select("Pipeline scope:", choices=choices).ask()


def _pick_question(questions: List[str], questionary) -> Optional[str]:
    """Let the user pick an existing question, edit one, or quit."""
    pick = questionary.select(
        "Choose a question (or an action):",
        choices=[*questions[:50], questionary.Separator(), "✎ Edit a question", "✗ Quit"],
    ).ask()

    if pick in (None, "✗ Quit"):
        return None
    if pick == "✎ Edit a question":
        base = questionary.select("Start from which question?", choices=questions[:50]).ask()
        if base is None:
            return None
        return questionary.text("Edit the question:", default=base).ask() or None
    return pick


def _browse_results(results: List[Dict], console, questionary) -> None:
    """Show the ranked tables a few at a time, asking before showing more."""
    from rich.table import Table

    if not results:
        console.print("[yellow]No tables retrieved.[/yellow]")
        return

    shown = 0
    while shown < len(results):
        for hit in results[shown : shown + TABLES_PER_PAGE]:
            console.print(_render_hit(hit, Table))
        shown += TABLES_PER_PAGE
        if shown < len(results):
            if not questionary.confirm(
                f"Show {min(TABLES_PER_PAGE, len(results) - shown)} more? "
                f"({shown}/{len(results)} shown)",
                default=False,
            ).ask():
                break


def _render_hit(hit: Dict, Table):
    """Render one ranked table as a compact rich panel with a small preview."""
    title = hit.get("title") or hit["table_id"]
    table = Table(title=f"#{hit['rank']}  {title}   (score {hit['score']})", title_justify="left")
    for header in (hit.get("headers") or [])[:6]:
        table.add_column(str(header), overflow="fold", max_width=18)
    for row in (hit.get("rows") or [])[:3]:
        table.add_row(*[str(c) for c in row[:6]])
    return table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def scope_label(use_stage3: bool) -> str:
    return "Stages 1+2+3" if use_stage3 else "Stages 1+2"


def _load_engine(cfg, console):
    from craft_tabqa.serve.engine import RetrievalEngine

    cache_dir = Path(cfg.data.output_dir) / "cache"
    if not (cache_dir / "splade_index.pkl").exists():
        console.print(f"[red]No preprocessed cache in {cache_dir}.[/red]")
        console.print(f"Run:  craft preprocess --config <your config>")
        return None
    with console.status("Loading models and index…"):
        return RetrievalEngine(cfg, cache_dir=str(cache_dir))


def _load_question_pool(cfg, console) -> List[str]:
    try:
        queries = load_queries(cfg.data.dataset, cfg.data.queries_file)
        pool = [q["question"] for q in queries if q.get("question")]
    except Exception as e:  # noqa: BLE001 - a missing query file shouldn't crash the app
        console.print(f"[dim]Could not load queries ({e}); starting with a blank list.[/dim]")
        pool = []
    if not pool:
        pool = ["What is the capital of France?"]
    return pool


def _apply_hardware_env(cfg) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_devices
    if cfg.hf_cache:
        os.environ["HF_HOME"] = cfg.hf_cache
    if cfg.hf_token:
        os.environ["HF_TOKEN"] = cfg.hf_token


def _open_file(path: str, console) -> None:
    """Open the results file in the OS default viewer (best effort)."""
    import subprocess
    import sys

    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", path])
        elif sys.platform.startswith("linux"):
            subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        console.print(f"[dim]Opened results file: {path}[/dim]")
    except Exception:  # noqa: BLE001 - opening a viewer is a convenience, not required
        console.print(f"[dim]Results file: {path}[/dim]")
