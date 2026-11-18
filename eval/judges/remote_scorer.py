"""
Remote risk scorer — calls the /score endpoint on model_server.py (laptop).
Replaces PromptGuard/toxic-bert with a proper 7B instruction-tuned classifier.
"""
import requests


class RemoteScorer:
    def __init__(self, api_url: str):
        self.url = api_url.rstrip("/") + "/score"
        print(f"RemoteScorer: API at {self.url}")

    def score(self, text: str) -> tuple[str, float]:
        """Returns (label, confidence) — label is 'safe' or 'unsafe'."""
        r = requests.post(self.url, json={"text": text}, timeout=30)
        r.raise_for_status()
        data = r.json()
        label = data["label"]
        score = 1.0 if label == "unsafe" else 0.0
        return label, score

    def is_unsafe(self, text: str) -> bool:
        label, _ = self.score(text)
        return label == "unsafe"
