"""Command-line interface for running, replaying, and auditing experiments.

The commands are organized around the claim this project makes: a result
should be re-runnable and checkable by someone who has only the artifact.
``run`` produces one, ``verify`` confirms it has not been altered, ``replay``
re-executes it and reports whether the numbers come back the same, and
``inspect`` shows what is inside without loading any tensors.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from transinterp.artifacts.bundle import ArtifactBundle
from transinterp.config.models import ExperimentConfig

app = typer.Typer(
    help="Run, replay, and audit reproducible interpretability experiments.",
    no_args_is_help=True,
)
config_app = typer.Typer(help="Configuration commands.", no_args_is_help=True)
app.add_typer(config_app, name="config")

console = Console()


def _load_config(path: Path) -> ExperimentConfig:
    if not path.exists():
        raise typer.BadParameter(f"Configuration does not exist: {path}")
    text = path.read_text()
    payload = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    return ExperimentConfig.model_validate(payload)


@config_app.command("validate")
def validate(config: Path) -> None:
    """Validate a config file and print its normalized form."""
    typer.echo(_load_config(config).model_dump_json(indent=2))


@app.command("run")
def run(
    config: Path = typer.Argument(..., help="Path to a YAML or JSON experiment config."),
    output: Path = typer.Option(None, "--output", "-o", help="Override the output directory."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace an existing bundle."),
) -> None:
    """Run an experiment and write an artifact bundle."""
    from transinterp.experiment import run_experiment

    experiment = _load_config(config)
    console.print(f"[bold]Running[/bold] {experiment.name} on {experiment.model.name_or_path}")
    bundle = run_experiment(experiment, output_dir=output, overwrite=overwrite)

    console.print(f"[green]Wrote[/green] {bundle.root}")
    console.print(f"  fingerprint: [cyan]{bundle.fingerprint}[/cyan]")
    for note in bundle.notes:
        console.print(f"  note: {note}")


@app.command("verify")
def verify(bundle: Path) -> None:
    """Check that a bundle's files still match its manifest digests."""
    result = ArtifactBundle.load(bundle).verify()
    if result.ok:
        console.print(f"[green]{result.summary()}[/green]")
        return
    console.print(f"[red]{result.summary()}[/red]")
    for path in result.corrupted:
        console.print(f"  modified: {path}")
    for path in result.missing:
        console.print(f"  missing:  {path}")
    for path in result.unexpected:
        console.print(f"  unlisted: {path}")
    raise typer.Exit(code=1)


@app.command("replay")
def replay(
    bundle: Path = typer.Argument(..., help="Bundle to re-run."),
    output: Path = typer.Option(None, "--output", "-o", help="Where to write the replay."),
) -> None:
    """Re-run a stored experiment and report whether it reproduces."""
    from transinterp.experiment import replay_experiment

    report = replay_experiment(bundle, output_dir=output)
    if report.reproduced:
        console.print(f"[green]{report.summary()}[/green]")
        return
    console.print(f"[yellow]{report.summary()}[/yellow]")
    raise typer.Exit(code=1)


@app.command("inspect")
def inspect(bundle: Path) -> None:
    """Show a bundle's provenance, metrics, and contents."""
    loaded = ArtifactBundle.load(bundle)

    console.print(f"[bold]{bundle}[/bold]")
    console.print(f"fingerprint: [cyan]{loaded.fingerprint}[/cyan]\n")

    provenance = loaded.provenance
    if provenance:
        table = Table(title="Provenance", show_header=False)
        table.add_row("created", provenance.created_at)
        table.add_row("python", provenance.python_version)
        table.add_row("platform", provenance.platform)
        for name, version in provenance.packages.items():
            if version:
                table.add_row(name, version)
        if provenance.git.commit:
            dirty = " (dirty)" if provenance.git.dirty else ""
            table.add_row("git", f"{provenance.git.commit[:12]}{dirty}")
        console.print(table)

    if loaded.metrics:
        table = Table(title="Metrics")
        table.add_column("name")
        table.add_column("value", overflow="fold")
        for key, value in sorted(loaded.metrics.items()):
            rendered = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            table.add_row(key, rendered[:200])
        console.print(table)

    names = loaded.tensor_names()
    if names:
        console.print(f"\n[bold]{len(names)} tensor(s)[/bold]: {', '.join(names[:8])}")
        if len(names) > 8:
            console.print(f"  ... and {len(names) - 8} more")

    for note in loaded.notes:
        console.print(f"\n[yellow]note[/yellow] {note}")


@app.command("compare")
def compare(first: Path, second: Path) -> None:
    """Diff two bundles and report what differs."""
    result = ArtifactBundle.load(first).compare(ArtifactBundle.load(second))
    if result["identical"]:
        console.print("[green]bundles are identical[/green]")
        return

    console.print("[yellow]bundles differ[/yellow]")
    for path in result["differing_files"]:
        console.print(f"  changed: {path}")
    for path in result["only_in_first"]:
        console.print(f"  only in first:  {path}")
    for path in result["only_in_second"]:
        console.print(f"  only in second: {path}")
    for key, (before, after) in result["environment_differences"].items():
        console.print(f"  env {key}: {before} -> {after}")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
