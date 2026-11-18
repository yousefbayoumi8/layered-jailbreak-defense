"""
Assemble the released attack dataset.

Merges the per-run successful_attacks files into one JSONL with a single
schema, plus a datasheet describing what the records are and what they are
not. Reads the cleaned files, so run eval/clean_mined.py first on any run
that was interrupted.

Each record is one attack prompt that drew a target behavior out of a
victim model under a named defense, confirmed by the HarmBench classifier.
Records are tiered:

  defended  the attack succeeded against a model with a defense in place
  control   the attack succeeded against the undefended model

The control tier is included so that users can compare, but it is not part
of the claim that these attacks defeat defended models.

Usage:
    python -m eval.build_dataset --output release/
    python -m eval.build_dataset --output release/ --runs eval/runs/adaptive_attacker_qwen,eval/runs/adaptive_attacker_llama
"""
import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_RUNS = [
    "eval/runs/adaptive_attacker_qwen",
    "eval/runs/adaptive_attacker_llama",
]

# Human-facing names, matching the paper.
DEFENSE_LABEL = {
    "none": "none",
    "sandwich": "self-reminder",
    "spotlight": "spotlighting",
    "adaptive_v2": "layered",
}


def read_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    return out


def collect(run_dirs):
    records, skipped = [], 0
    for run_dir in run_dirs:
        run = Path(run_dir)
        if not run.exists():
            print(f"[skip] {run} does not exist")
            continue
        for sub in sorted(p for p in run.iterdir() if p.is_dir()):
            clean = sub / "successful_attacks.clean.jsonl"
            raw = sub / "successful_attacks.jsonl"
            path = clean if clean.exists() else raw
            if not path.exists():
                continue
            if not clean.exists():
                print(f"[warn] {sub.name}: using uncleaned file. Run "
                      f"eval.clean_mined --dir {run} first.")
            defense = sub.name
            for r in read_jsonl(path):
                if not r.get("attack_prompt") or not r.get("victim_response"):
                    skipped += 1
                    continue
                records.append({
                    "id": r.get("id") or f"{r.get('behavior_id')}__{defense}",
                    "behavior_id": r.get("behavior_id", ""),
                    "behavior": r.get("behavior", ""),
                    "behavior_category": r.get("behavior_category", ""),
                    "source_benchmark": r.get("dataset", "harmbench"),
                    "victim_model": r.get("victim_model", ""),
                    "defense": defense,
                    "defense_label": DEFENSE_LABEL.get(defense, defense),
                    "tier": "control" if defense == "none" else "defended",
                    "attempts_to_success": r.get("iteration"),
                    "attack_prompt": r["attack_prompt"],
                    "victim_response": r["victim_response"],
                    "judge": r.get("judge", "harmbench_mistral7b"),
                    "judge_raw": r.get("judge_raw", ""),
                    **({"critic_verdict": r["critic_verdict"]}
                       if "critic_verdict" in r else {}),
                })
    return records, skipped


DATASHEET = """# Adaptive jailbreak attacks against defended open-weight models

{n_total} attack prompts that drew a target behavior out of a locally-run
open-weight language model, each confirmed by the HarmBench classifier.
{n_defended} of them succeeded against a model with a defense in place. The
remaining {n_control} succeeded against the undefended model and are included
as a control tier.

## Why this exists

A defense cannot be measured against an adversary that adapts unless adaptive
attacks exist to measure it with. Public jailbreak collections almost all
target undefended models, so a defense evaluated against them is being tested
on attacks written without knowledge of it. These attacks were generated
against specific defenses and are labelled with the defense they defeated.

## How it was made

An attacker model was given a target behavior from HarmBench, wrote a prompt
intended to draw that behavior out of the victim, and read the victim's
defended reply. A classifier labelled whether the reply was an instance of the
behavior, and the label and reply were returned to the attacker, which
rewrote. The loop ran for at most ten attempts per behavior and stopped at the
first confirmed success.

Attacker: Mistral-7B-Instruct-v0.3. Victims: {models}. Judge: the HarmBench
Mistral-7B validation classifier, held fixed across every condition. All models
ran locally in four-bit quantisation.

## Fields

| field | meaning |
|---|---|
| `id` | unique record identifier |
| `behavior_id` | HarmBench behavior identifier |
| `behavior` | the target behavior, as stated by HarmBench |
| `behavior_category` | HarmBench functional category |
| `source_benchmark` | benchmark the behavior came from |
| `victim_model` | model the attack succeeded against |
| `defense` | internal defense key |
| `defense_label` | name used in the paper |
| `tier` | `defended` or `control` |
| `attempts_to_success` | which attempt succeeded, 1 to 10 |
| `attack_prompt` | the attacker's prompt |
| `victim_response` | what the victim returned, after any defense |
| `judge` | classifier that labelled the success |
| `judge_raw` | raw classifier output |
| `critic_verdict` | present for the layered defense only |

## Composition

{composition}

## Limitations

Success is labelled by a single classifier, so the labels inherit its error
rate. A manual audit of a random sample of the undefended tier found 84 to 96
percent genuine, and the same audit was not repeated for the defended tiers.

Attacks are specific to the victim and defense they were generated against.
Replaying them elsewhere succeeds around ten percent of the time, so this is
not a general-purpose jailbreak set.

The attacker was allowed ten attempts per behavior, so behaviors absent from
this file are not necessarily robust. They were not defeated within that
budget.

## Intended use and terms

Released for research on defending language models. The target behaviors come
unchanged from a public benchmark, so the set of harms represented here is
already documented in the literature. Do not use these prompts against systems
you do not own or have permission to test.

Generated {date}.
"""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default="release",
                    help="Directory to write the dataset and datasheet into.")
    ap.add_argument("--runs", default=",".join(DEFAULT_RUNS),
                    help="Comma-separated mined run directories.")
    args = ap.parse_args()

    run_dirs = [r.strip() for r in args.runs.split(",") if r.strip()]
    records, skipped = collect(run_dirs)
    if not records:
        raise SystemExit("No records found. Check --runs.")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "adaptive_attacks.jsonl"
    with data_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_cell = Counter((r["victim_model"], r["defense_label"]) for r in records)
    tiers = Counter(r["tier"] for r in records)
    models = sorted({r["victim_model"] for r in records if r["victim_model"]})
    behaviors = len({r["behavior_id"] for r in records})

    lines = ["| victim | defense | tier | attacks |", "|---|---|---|---|"]
    for (model, defense), n in sorted(by_cell.items()):
        tier = "control" if defense == "none" else "defended"
        lines.append(f"| {model} | {defense} | {tier} | {n} |")
    lines.append(f"| | | **total** | **{len(records)}** |")
    lines.append("")
    lines.append(f"Distinct behaviors represented: {behaviors}.")

    sheet = DATASHEET.format(
        n_total=len(records),
        n_defended=tiers["defended"],
        n_control=tiers["control"],
        models=", ".join(models) or "see records",
        composition="\n".join(lines),
        date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )
    (out_dir / "README.md").write_text(sheet, encoding="utf-8")

    print(f"Wrote {data_path}  ({len(records)} records)")
    print(f"Wrote {out_dir / 'README.md'}")
    if skipped:
        print(f"Skipped {skipped} records missing a prompt or response.")
    print()
    print(f"  defended : {tiers['defended']}")
    print(f"  control  : {tiers['control']}")
    print(f"  behaviors: {behaviors}")


if __name__ == "__main__":
    main()
