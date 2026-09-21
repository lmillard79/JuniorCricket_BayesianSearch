"""Put real player names back into a report that uses aliases.

Reports use aliases (P01, ...) so nothing sensitive enters the model files.
This writes a private copy with names for your own use. The copy is saved
next to the original as ``<name>_named<ext>`` under ``data/outputs/``
(gitignored). Do not commit or share it.

Example:
    python scripts/name_report.py data/outputs/lineup_recommendation.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from junior_cricket.playhq_scorebook import alias_names, translate_aliases

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "playhq"


def main() -> None:
    """Translate one report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("report", help="Path to a report that uses aliases")
    args = parser.parse_args()
    source = Path(args.report)
    names = alias_names(RAW_DIR / "player_map.csv", RAW_DIR / "opposition_map.csv")
    target = source.with_name(f"{source.stem}_named{source.suffix}")
    target.write_text(translate_aliases(source.read_text(encoding="utf-8"), names),
                      encoding="utf-8")
    print(f"Wrote {target} ({len(names)} aliases known). Keep it private.")


if __name__ == "__main__":
    main()
