"""
Transfer-replay — take attacks mined on one (victim, defense) cell and replay
them ONE SHOT (no iteration) against other cells, to see if they still work.

Cheaply fills out the model x defense grid and yields the transfer matrix. A
transferred success is a legitimate dataset entry for the target cell (provenance
= "transferred"); a failure is recorded too (it's what gives the transfer rate).

Setup: only the VICTIM loads locally on the 4080 (victims are swapped one at a
time); judging is done by the remote judge_server.py — same as mining, minus the
attacker. Keep the judge server running.

Per target victim: apply the (source) defense to each source attack_prompt, run
the victim once, judge. By default each attack is replayed under its OWN source
defense (cross-model, same-defense). Pass --defenses to force a set of target
defenses instead (cross-defense grid).

Output <output>/transfer.jsonl — one row per (attack, target victim, defense):
  source_defense, source_victim, behavior_id, behavior, attack_prompt,
  target_victim, target_defense, transferred(bool), judge_label, victim_response

Usage:
    python -m eval.transfer --source-dir eval/runs/adaptive_attacker_qwen \\
        --output eval/runs/transfer --judge-url http://<judge-host>:8000 \\
        --victims llama,gemma
"""
import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model_loader import load_model, generate_response, MODELS
from eval import defenses as defenses_module
from eval.remote_attacker import RemoteJudge, iso_now


def prefer_clean(dir_path: Path, base: str) -> Path:
    clean = dir_path / base.replace(".jsonl", ".clean.jsonl")
    return clean if clean.exists() else dir_path / base


def load_source(source_dir: Path) -> List[Dict]:
    """Read successful attacks from every defense subfolder (prefer .clean)."""
    out = []
    for sub in sorted(source_dir.iterdir()):
        if not sub.is_dir():
            continue
        p = prefer_clean(sub, "successful_attacks.jsonl")
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            out.append({
                "id": r.get("id", f"{r.get('behavior_id')}__{sub.name}"),
                "behavior_id": r.get("behavior_id", ""),
                "behavior": r.get("behavior", ""),
                "behavior_category": r.get("behavior_category", ""),
                "dataset": r.get("dataset", ""),
                "attack_prompt": r.get("attack_prompt", ""),
                "source_victim": r.get("victim_model", ""),
                "source_defense": sub.name,
            })
    return out


def done_keys(path: Path) -> Set[Tuple[str, str, str]]:
    keys = set()
    if path.exists():
        for line in path.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                keys.add((r["id"], r["target_victim"], r["target_defense"]))
            except Exception:
                continue
    return keys


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-dir", required=True,
                    help="Mined run dir (reads successful_attacks from each defense subfolder).")
    ap.add_argument("--output", required=True, help="Output dir for transfer.jsonl.")
    ap.add_argument("--judge-url", required=True)
    ap.add_argument("--victims", default="llama,gemma",
                    help="Target victims to replay on (comma-separated).")
    ap.add_argument("--defenses", default="self",
                    help="'self' (default) = replay each attack under its own source "
                         "defense; or a comma-list of defenses to force a cross-defense grid.")
    ap.add_argument("--source-defenses", default=None,
                    help="Only replay attacks whose source defense is in this comma-list.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap source attacks (per source defense) — for testing.")
    ap.add_argument("--victim-max-new-tokens", type=int, default=384)
    ap.add_argument("--victim-temperature", type=float, default=0.7)
    ap.add_argument("--judge-timeout", type=int, default=60)
    ap.add_argument("--no-quantize", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    victims = [v.strip() for v in args.victims.split(",") if v.strip()]
    for v in victims:
        if v not in MODELS:
            raise SystemExit(f"Unknown victim '{v}'. Available: {list(MODELS)}")
    cross_defenses = None if args.defenses == "self" else \
        [d.strip() for d in args.defenses.split(",") if d.strip()]
    if cross_defenses:
        for d in cross_defenses:
            if d not in defenses_module.DEFENSES:
                raise SystemExit(f"Unknown defense '{d}'.")

    quantize = not args.no_quantize
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "transfer.jsonl"

    source = load_source(Path(args.source_dir))
    if args.source_defenses:
        keep = {d.strip() for d in args.source_defenses.split(",")}
        source = [s for s in source if s["source_defense"] in keep]
    if args.limit:
        by_def = defaultdict(list)
        for s in source:
            by_def[s["source_defense"]].append(s)
        source = [s for lst in by_def.values() for s in lst[:args.limit]]
    if not source:
        raise SystemExit("No source attacks found.")

    judge = RemoteJudge(args.judge_url, timeout=args.judge_timeout)
    print(f"Pinging judge {args.judge_url}...")
    judge.ping()
    print("  judge reachable.")

    done = done_keys(out_path)
    print(f"{len(source)} source attacks | victims={victims} | "
          f"defenses={'self' if cross_defenses is None else cross_defenses} | "
          f"{len(done)} replays already done")

    out_f = out_path.open("a", encoding="utf-8")
    tallies: Dict[tuple, list] = defaultdict(lambda: [0, 0])  # (src_def,tgt_v,tgt_def)->[hit,tot]

    try:
        for victim_name in victims:
            # what (attack, defense) pairs still need this victim?
            pending = []
            for s in source:
                tdefs = [s["source_defense"]] if cross_defenses is None else cross_defenses
                for td in tdefs:
                    if victim_name == s["source_victim"] and td == s["source_defense"]:
                        continue  # already mined that exact cell
                    if (s["id"], victim_name, td) in done:
                        continue
                    pending.append((s, td))
            if not pending:
                print(f"[{victim_name}] nothing pending, skipping.")
                continue

            print(f"=== Loading victim {victim_name} ({len(pending)} replays) ===")
            model, tok = load_model(victim_name, quantize=quantize)

            bar = tqdm(pending, desc=f"[{victim_name}]", unit="replay",
                       disable=args.quiet, ncols=110, smoothing=0.1)
            try:
                for s, td in bar:
                    system_prompt, messages = defenses_module.apply(td, s["attack_prompt"])
                    resp = generate_response(model, tok, system_prompt, messages,
                                             max_new_tokens=args.victim_max_new_tokens,
                                             temperature=args.victim_temperature)
                    try:
                        verdict = judge.classify(s["behavior"], resp)
                    except Exception as e:  # noqa: BLE001
                        print(f"FATAL: judge unreachable ({e}). Aborting — rerun to resume.")
                        raise
                    transferred = verdict["label"] == "complied"
                    rec = {
                        "ts": iso_now(), "id": s["id"],
                        "behavior_id": s["behavior_id"], "behavior": s["behavior"],
                        "behavior_category": s["behavior_category"], "dataset": s["dataset"],
                        "attack_prompt": s["attack_prompt"],
                        "source_victim": s["source_victim"], "source_defense": s["source_defense"],
                        "target_victim": victim_name, "target_defense": td,
                        "transferred": transferred, "judge_label": verdict["label"],
                        "judge_raw": verdict.get("raw", ""), "victim_response": resp,
                        "provenance": "transferred",
                    }
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                    t = tallies[(s["source_defense"], victim_name, td)]
                    t[1] += 1
                    if transferred:
                        t[0] += 1
                    bar.set_postfix_str(f"transfer={t[0]}/{t[1]}", refresh=False)
            finally:
                bar.close()
                del model, tok
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(3)
    finally:
        out_f.close()

    if tallies:
        print("\n=== transfer rates (source_defense -> target_victim/target_defense) ===")
        for (sd, tv, td), (hit, tot) in sorted(tallies.items()):
            print(f"  {sd:>12} -> {tv}/{td:<12}: {hit}/{tot}  ({100 * hit / tot:.1f}%)")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
