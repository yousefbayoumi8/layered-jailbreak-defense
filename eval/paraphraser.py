"""
Paraphraser — supports a remote API (model_server.py on the laptop) or falls
back to the local ramsrigouthamg/t5_paraphraser T5 model.

Pass api_url to use the remote server (recommended — much better quality):
    Paraphraser(api_url="http://<judge-host>:8000")

Without api_url, loads the local T5 model (lower quality, no crashes).
"""
import gc

import requests
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration


class Paraphraser:
    LOCAL_MODEL_ID = "ramsrigouthamg/t5_paraphraser"

    def __init__(self, api_url: str | None = None):
        self._api_url = (api_url.rstrip("/") + "/paraphrase") if api_url else None
        self._model = None
        self._tokenizer = None
        self.device = None

        if self._api_url:
            print(f"Paraphraser: remote API at {self._api_url}")
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading local Paraphraser ({self.LOCAL_MODEL_ID})...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.LOCAL_MODEL_ID)
            self._model = T5ForConditionalGeneration.from_pretrained(
                self.LOCAL_MODEL_ID,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ).to(self.device)
            self._model.eval()
            print("Paraphraser loaded.")

    def paraphrase(self, text: str, max_length: int = 512) -> str:
        if self._api_url:
            r = requests.post(self._api_url, json={"text": text}, timeout=120)
            r.raise_for_status()
            return r.json()["paraphrased"]

        # Local T5 fallback
        inputs = self._tokenizer(
            f"paraphrase: {text} </s>",
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                top_k=120,
                top_p=0.98,
                num_return_sequences=1,
            )
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    def unload(self):
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
