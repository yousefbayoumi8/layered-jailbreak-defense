"""
Clean a mined run folder so only canonical (completed-behavior) records remain.

Why this exists: resume in eval/remote_attacker.py is per-behavior. If a run is
killed mid-behavior and later resumed, that behavior restarts from iteration 1,
leaving the abandoned partial iterations in attacks.jsonl under an OLDER run_id.
progress.jsonl records exactly one row per COMPLETED behavior, tagged with the
run_id that finished it — so it is the source of truth for what is canonical.

This tool keeps, in attacks.jsonl and successful_attacks.jsonl, only rows whose
(behavior_id, run_id) matches a completed behavior in progress.jsonl. It drops:
  - "partial"     — behavior completed later under a DIFFERENT run_id (abandoned
                    fragments from an interrupted attempt)
  - "incomplete"  — behavior never finished, so it has no progress.jsonl row

Non-destructive by default: writes <name>.clean.jsonl beside the originals.
Use --in-place to overwrite (originals are backed up to <name>.jsonl.bak first).

Usage:
    python -m eval.clean_mined --dir eval/runs/mined/none
    python -m eval.clean_mined --dir eval/runs/mined            # all subfolders
    python -m eval.clean_mined --dir eval/runs/mined/none --in-place
"""
import argparse
import json
import shutil
from pathlib import Path


def load_jsonl(p: Path):
    out = []
    if p.exists():
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def canonical_map(progress_path: Path) -> dict:
    """behavior_id -> run_id of the completed attempt (last wins)."""
    m = {}
    for r in load_jsonl(progress_path):
        bid = r.get("behavior_id")
        if bid is not None:
            m[bid] = r.get("run_id")
    return m


def clean_file(path: Path, canonical: dict, in_place: bool):
    if not path.exists():
        return None
    rows = load_jsonl(path)
    kept, dropped_partial, dropped_incomplete = [], 0, 0
    for r in rows:
        bid = r.get("behavior_id")
        if bid not in canonical:
            dropped_incomplete += 1
        elif r.get("run_id") != canonical[bid]:
            dropped_partial += 1
        else:
            kept.append(r)

    if in_place:
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        out_path = path
    else:
        out_path = path.with_suffix(".clean" + path.suffix)
    with out_path.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "file": path.name, "total": len(rows), "kept": len(kept),
        "partial": dropped_partial, "incomplete": dropped_incomplete,
        "out": out_path.name,
    }


def clean_dir(d: Path, in_place: bool) -> bool:
    prog = d / "progress.jsonl"
    if not prog.exists():
        return False
    canonical = canonical_map(prog)
    print(f"\n[{d}]  {len(canonical)} completed behaviors")
    for name in ("attacks.jsonl", "successful_attacks.jsonl"):
        res = clean_file(d / name, canonical, in_place)
        if res is None:
            print(f"  {name:26s} (not present)")
            continue
        print(f"  {res['file']:26s} total={res['total']:6d}  kept={res['kept']:6d}"
              f"  dropped: partial={res['partial']} incomplete={res['incomplete']}"
              f"   -> {res['out']}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True,
                    help="A mined defense folder (contains progress.jsonl), or a "
                         "parent holding several such folders.")
    ap.add_argument("--in-place", action="store_true",
                    help="Overwrite originals (backed up to *.bak). Default writes "
                         "*.clean.jsonl instead.")
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.exists():
        raise SystemExit(f"No such directory: {root}")

    if clean_dir(root, args.in_place):
        print("\nDone.")
        return
    # Not a leaf run folder — try subfolders.
    any_sub = False
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and (sub / "progress.jsonl").exists():
            clean_dir(sub, args.in_place)
            any_sub = True
    if not any_sub:
        raise SystemExit(f"No progress.jsonl in {root} or its subfolders.")
    print("\nDone.")


if __name__ == "__main__":
    main()
