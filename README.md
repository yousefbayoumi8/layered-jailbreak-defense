# Training-free jailbreak defenses under adaptive attack

This code compares training-free defenses on Qwen2.5-7B-Instruct and
Llama-3.1-8B-Instruct in two settings. In the static setting each HarmBench
prompt is tried once. In the adaptive setting a second model,
Mistral-7B-Instruct, writes its own prompts and revises them over up to ten
attempts per behavior, following PAIR. Every response is scored by the
HarmBench Mistral-7B classifier. False refusal is measured on XSTest.

## Requirements

All models load in 4-bit NF4. The adaptive attack keeps the attacker and the
victim together on one 16 GB GPU and sends each response to the classifier on
a second machine through `judge_server.py`, where about 6 GB of GPU memory is
enough. The static benchmark loads one model at a time.

```
python -m venv env
env\Scripts\activate          # Linux/macOS: source env/bin/activate
pip install -r requirements.txt
```

Llama 3.1 is gated on Hugging Face, so request access and run
`huggingface-cli login` first. The datasets (HarmBench, AdvBench,
JailbreakBench, XSTest) are downloaded from their public sources on first use.

## Defenses

Defined in `eval/defenses.py`. The output critic is in `eval/critic.py`.

| Key | Defense |
|---|---|
| `none` | Undefended baseline |
| `sandwich` | Self-reminder |
| `spotlight` | Spotlighting |
| `paraphrase` | Paraphrase |
| `shield_ignore` | Ignore-shield |
| `shield_escape` | Escape-deletion |
| `shield_fakecom` | Fake-completion |
| `shield_fakecom_t` | Fake-completion (chat template) |
| `adaptive` | Risk-gated design |
| `adaptive_v2` | Layered defense: input hardening plus an output critic |

## Static benchmark

```
python -m eval.runner --output eval/runs/benchmark --models qwen,llama \
    --datasets harmbench --defenses none,sandwich,spotlight,adaptive_v2
python -m eval.judge --runs-dir eval/runs/benchmark
python -m eval.summary --runs-dir eval/runs/benchmark
```

The paraphrase and risk-gated defenses also need `model_server.py` running on a
second machine, passed with `--paraphrase-url` and `--score-url`.

## Adaptive attack

Start the classifier on the second machine:

```
python judge_server.py
```

Then run the attacker against one victim and defense:

```
python -m eval.remote_attacker --output eval/runs/adaptive_attacker_qwen \
    --judge-url http://<judge-host>:8000 --victim qwen --defense adaptive_v2
python -m eval.clean_mined --dir eval/runs/adaptive_attacker_qwen
```

The defaults match the reported settings: at most ten attempts, the attacker
sees its last six, temperatures 0.9 for the attacker and 0.7 for the victim.
Runs are resumable, so rerunning the same command continues where it stopped.

The repeated run and the hidden-response control use a fixed sample of 100
behaviors:

```
python -m eval.remote_attacker ... --sample 100 --seed 1
python -m eval.remote_attacker ... --sample 100 --seed 1 --hide-responses
```

## Other analyses

```
# False refusal on XSTest
python -m eval.overrefusal --output eval/runs/overrefusal \
    --models qwen,llama --defenses none,sandwich,spotlight,adaptive_v2
python -m eval.overrefusal --output eval/runs/overrefusal --phase classify
python -m eval.overrefusal --output eval/runs/overrefusal --phase summary

# What the input hardening alone would have released, from the benchmark runs
python -m eval.critic_ablation

# Replaying attacks found against Qwen on another model
python -m eval.transfer --source-dir eval/runs/adaptive_attacker_qwen \
    --output eval/runs/transfer --judge-url http://<judge-host>:8000 --victims llama
```

All outputs are written under `eval/runs/`, which is not part of this
repository.

## Attack dataset

The successful attacks found by the adaptive attacker are working jailbreak
prompts. They are not included here. `eval/build_dataset.py` assembles them
from the run outputs, and the released dataset is available for research use
on request from the authors.
