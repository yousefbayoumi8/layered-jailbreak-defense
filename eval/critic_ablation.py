"""
Ablation of the output critic, computed from data already collected.

Whenever the critic replaced a response, the response it suppressed was stored
as `pre_critic_response` and never judged. Judging those recovers exactly what
the defense would have released with the input hardening alone, so the critic's
contribution can be measured without rerunning anything.

Two different quantities come out of this, and they are not the same thing:

  STATIC ablation is a true counterfactual. The prompt is fixed, so removing the
  critic simply releases the pre-critic response. Attack success without the
  critic is therefore the responses it never touched that were harmful, plus the
  suppressed responses that turn out to have been harmful.

  ADAPTIVE attribution is not a counterfactual. There the attacker adapted to
  the critic, so a run without it would have taken a different course. What can
  be reported is how many genuinely harmful responses the critic caught that
  input hardening alone had let through.

Only the judge loads, so the full GPU is free and no remote judge is needed.
Append-only and resumable.

Usage:
    python -m eval.critic_ablation
    python -m eval.critic_ablation --limit 20      # smoke test first
"""
import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

STATIC_DIR = Path("eval/runs/benchmark/adaptive_v2")
ADAPTIVE_RUNS = {
    "qwen": Path("eval/runs/adaptive_attacker_qwen/adaptive_v2"),
    "llama": Path("eval/runs/adaptive_attacker_llama/adaptive_v2"),
}
OUT_DIR = Path("eval/runs/critic_ablation")


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(
        timespec="milliseconds").replace("+00:00", "Z")


def load_jsonl(path: Path) -> List[dict]:
    out = []
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        continue
    return out


def prefer_clean(d: Path, base: str) -> Path:
    clean = d / base.replace(".jsonl", ".clean.jsonl")
    return clean if clean.exists() else d / base


def collect_work() -> List[dict]:
    """Every stored pre-critic response that still needs a verdict."""
    work = []

    for r in load_jsonl(STATIC_DIR / "generations.jsonl"):
        if r.get("dataset") != "harmbench" or "pre_critic_response" not in r:
            continue
        work.append({
            "source": "static",
            "key": f"static|{r['model']}|{r['behavior_id']}",
            "model": r["model"],
            "behavior_id": r["behavior_id"],
            "behavior": r.get("raw_behavior") or r.get("prompt", ""),
            "pre_critic_response": r["pre_critic_response"],
        })

    for model, d in ADAPTIVE_RUNS.items():
        for r in load_jsonl(prefer_clean(d, "attacks.jsonl")):
            if "pre_critic_response" not in r:
                continue
            work.append({
                "source": "adaptive",
                "key": f"adaptive|{model}|{r['behavior_id']}|{r['iteration']}",
                "model": model,
                "behavior_id": r["behavior_id"],
                "behavior": r.get("target_behavior", ""),
                "pre_critic_response": r["pre_critic_response"],
            })

    return [w for w in work if w["behavior"] and w["pre_critic_response"]]


def done_keys(path: Path) -> Set[str]:
    return {r["key"] for r in load_jsonl(path) if "key" in r}


def judge_pending(work: List[dict], out_path: Path, quantize: bool, quiet: bool):
    done = done_keys(out_path)
    pending = [w for w in work if w["key"] not in done]
    print(f"{len(work)} suppressed responses | {len(done)} already judged "
          f"| {len(pending)} pending")
    if not pending:
        return

    from eval.judges.harmbench_cls import HarmBenchJudge
    judge = HarmBenchJudge(quantize=quantize)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = out_path.open("a", encoding="utf-8")
    try:
        bar = tqdm(pending, desc="judging suppressed", unit="rec",
                   disable=quiet, ncols=100)
        for w in bar:
            verdict = judge.classify(w["behavior"], w["pre_critic_response"])
            f.write(json.dumps({
                "ts": iso_now(), "key": w["key"], "source": w["source"],
                "model": w["model"], "behavior_id": w["behavior_id"],
                "pre_critic_label": verdict["label"],
                "pre_critic_raw": verdict.get("raw", ""),
            }, ensure_ascii=False) + "\n")
            f.flush()
        bar.close()
    finally:
        f.close()
        judge.unload()


def summarise(out_path: Path):
    verdicts = {r["key"]: r for r in load_jsonl(out_path)}
    if not verdicts:
        raise SystemExit("Nothing judged yet.")

    # ---------- static: a true counterfactual ----------
    # Final label per (model, behavior) from the benchmark judgments.
    final: Dict[Tuple[str, str], str] = {}
    for r in load_jsonl(STATIC_DIR / "judgments.jsonl"):
        if r.get("dataset") == "harmbench" and r.get("judge") == "harmbench_cls":
            final[(r["model"], r["behavior_id"])] = r.get("label")

    per_model = defaultdict(lambda: {"n": 0, "with": 0, "recovered": 0, "fired": 0})
    for (model, bid), label in final.items():
        s = per_model[model]
        s["n"] += 1
        if label == "complied":
            s["with"] += 1
    for key, v in verdicts.items():
        if v["source"] != "static":
            continue
        s = per_model[v["model"]]
        s["fired"] += 1
        if v["pre_critic_label"] == "complied":
            s["recovered"] += 1

    print("\n=== Output-critic ablation, static evaluation (HarmBench) ===")
    print("Input hardening alone vs input hardening plus the output critic.\n")
    hdr = (f"  {'model':<8}{'n':>5}{'no critic':>11}{'with critic':>13}"
           f"{'reduction':>11}{'fired':>7}{'blocked':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for model in sorted(per_model):
        s = per_model[model]
        if not s["n"]:
            continue
        with_c = 100 * s["with"] / s["n"]
        without = 100 * (s["with"] + s["recovered"]) / s["n"]
        print(f"  {model:<8}{s['n']:>5}{without:>10.1f}%{with_c:>12.1f}%"
              f"{with_c - without:>10.1f}pp{s['fired']:>7}{s['recovered']:>9}")
    print("\n  'fired' = responses the critic replaced. 'blocked' = those the "
          "judge\n  confirms were harmful, so the reduction is exact rather "
          "than estimated.")

    # ---------- adaptive: attribution, not a counterfactual ----------
    adaptive = defaultdict(lambda: {"fired": 0, "harmful": 0})
    for v in verdicts.values():
        if v["source"] != "adaptive":
            continue
        a = adaptive[v["model"]]
        a["fired"] += 1
        if v["pre_critic_label"] == "complied":
            a["harmful"] += 1

    if adaptive:
        print("\n=== Output-critic interventions under adaptive attack ===")
        print("Not a counterfactual. The attacker adapted to the critic, so this")
        print("counts what it caught rather than what its removal would cost.\n")
        for model in sorted(adaptive):
            a = adaptive[model]
            pct = 100 * a["harmful"] / a["fired"] if a["fired"] else 0.0
            print(f"  {model:<8} replaced {a['fired']:>4} responses, "
                  f"{a['harmful']:>4} of them confirmed harmful ({pct:.0f}%)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT_DIR / "ablation.jsonl"))
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap the records judged, for a smoke test.")
    ap.add_argument("--no-quantize", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    if not args.summary_only:
        work = collect_work()
        if args.limit:
            work = work[:args.limit]
        judge_pending(work, out_path, not args.no_quantize, args.quiet)
    summarise(out_path)


if __name__ == "__main__":
    main()
