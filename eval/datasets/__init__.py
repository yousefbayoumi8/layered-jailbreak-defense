from eval.datasets import advbench, harmbench, jailbreakbench, xstest

LOADERS = {
    "advbench": advbench.load,
    "harmbench": harmbench.load,
    "jbb": jailbreakbench.load,
    # Benign/over-refusal suite. Not an attack set — used by eval/overrefusal.py,
    # not by the ASR pipeline (its prompts have no harmful behavior to judge).
    "xstest": xstest.load,
}


def load(name: str, limit: int | None = None):
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(LOADERS)}")
    items = LOADERS[name]()
    if limit is not None:
        items = items[:limit]
    return items
