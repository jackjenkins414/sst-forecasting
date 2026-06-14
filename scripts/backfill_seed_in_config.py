"""
Backfill `random_seed` in saved config.json files under experiments/best_*_seed*/.

Earlier runs used `--seed N` but didn't update the saved config.json (the seed
came from the HPO config and was never overridden), so configs say
`random_seed: 42` even though the run used seed N. This walks each
`best_<model>_seed<N>/config.json` and sets `random_seed = N`. Idempotent.
"""

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEST_DIR     = PROJECT_ROOT / "experiments"
PATTERN      = re.compile(r"best_(.+)_seed(\d+)$")


def main():
    fixed = 0
    skipped = 0
    for d in sorted(BEST_DIR.glob("best_*_seed*")):
        m = PATTERN.match(d.name)
        if not m:
            continue
        cfg_path = d / "config.json"
        if not cfg_path.exists():
            continue
        seed = int(m.group(2))
        cfg = json.load(open(cfg_path))
        old = cfg.get("random_seed")
        if old == seed:
            skipped += 1
            continue
        cfg["random_seed"] = seed
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  {d.name}: random_seed {old} -> {seed}")
        fixed += 1
    print(f"\nFixed {fixed} config(s), {skipped} already correct.")


if __name__ == "__main__":
    main()
