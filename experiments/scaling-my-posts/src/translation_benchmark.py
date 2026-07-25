"""Reproducible EN <-> PT-BR benchmark using this blog's parallel BERT post."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import sacrebleu
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


HISTORICAL_COMMIT = "e78386012f37512a5ebd316a1389fabf9bf3b707"
ENGLISH_NOTEBOOK = "_notebooks/2020-09-19-Distilling-BERT.ipynb"
PORTUGUESE_NOTEBOOK = "_notebooks/2020-09-19-Distilling-BERT-pt.ipynb"

# Hand-checked cell pairs. Image-only cells, navigation notes, executable code,
# and deliberately localized examples are excluded from the automatic benchmark.
ALIGNED_CELL_PAIRS = [
    (3, 3),
    (8, 8),
    (9, 9),
    (21, 21),
    (24, 24),
    (25, 25),
    (27, 27),
    (32, 32),
    (34, 34),
    (37, 37),
    (45, 46),
    (48, 49),
    (50, 51),
]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    loader: Callable[[str], "Translator"]


class Translator:
    parameter_count: int

    def translate(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
        progress: Callable[[int], None] | None = None,
    ) -> list[str]:
        raise NotImplementedError

    def close(self) -> None:
        for value in vars(self).values():
            if hasattr(value, "to"):
                try:
                    value.to("cpu")
                except Exception:
                    pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _git_json(repo: Path, commit: str, path: str) -> dict:
    raw = subprocess.check_output(
        ["git", "show", f"{commit}:{path}"], cwd=repo, text=True, encoding="utf-8"
    )
    return json.loads(raw)


def clean_markdown(value: str) -> str:
    """Reduce Markdown to the prose a translation engine would receive."""
    value = re.sub(r"{%\s*fn\s+\d+\s*%}", "", value)
    value = re.sub(
        r"{{\s*['\"]?(.*?)['\"]?\s*\|\s*fndetail:\s*\d+\s*}}",
        r"\1",
        value,
        flags=re.DOTALL,
    )
    value = re.sub(r"!\[[^]]*]\([^\n]*\)", "", value)
    value = re.sub(r"\[([^]]+)]\([^)]*\)", r"\1", value)
    value = re.sub(r"^\s{0,3}#{1,6}\s*", "", value, flags=re.MULTILINE)
    value = re.sub(r"^\s*>\s?(?:\w+:\s*)?", "", value, flags=re.MULTILINE)
    value = re.sub(r"[*_`]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def load_parallel_passages(repo: Path) -> pd.DataFrame:
    english = _git_json(repo, HISTORICAL_COMMIT, ENGLISH_NOTEBOOK)
    portuguese = _git_json(repo, HISTORICAL_COMMIT, PORTUGUESE_NOTEBOOK)
    rows = []
    for pair_id, (en_cell, pt_cell) in enumerate(ALIGNED_CELL_PAIRS, start=1):
        en = clean_markdown("".join(english["cells"][en_cell]["source"]))
        pt = clean_markdown("".join(portuguese["cells"][pt_cell]["source"]))
        rows.append(
            {
                "pair_id": pair_id,
                "english_cell": en_cell,
                "portuguese_cell": pt_cell,
                "english": en,
                "portuguese": pt,
            }
        )
    return pd.DataFrame(rows)


class MarianTranslator(Translator):
    MODELS = {
        ("en", "pt-BR"): (
            "Helsinki-NLP/opus-mt-tc-big-en-pt",
            "9f2863d807ecf91a374bdbecb8d01e402e90622e",
        ),
        ("pt-BR", "en"): (
            "Helsinki-NLP/opus-mt-ROMANCE-en",
            "e9ca9975e3972afd80732f08ce01d3a1339f47f8",
        ),
    }

    def __init__(self, device: str):
        self.device = device
        self.loaded_pair: tuple[str, str] | None = None
        self.tokenizer = None
        self.model = None
        self.parameter_count = 0

    def _load(self, pair: tuple[str, str]) -> None:
        if pair == self.loaded_pair:
            return
        if self.model is not None:
            self.model.to("cpu")
            del self.model, self.tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        model_name, revision = self.MODELS[pair]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, revision=revision
        ).to(self.device)
        self.model.eval()
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self.loaded_pair = pair

    def translate(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
        progress: Callable[[int], None] | None = None,
    ) -> list[str]:
        pair = (source_lang, target_lang)
        self._load(pair)
        prepared = [f">>pob<< {text}" for text in texts] if pair == ("en", "pt-BR") else texts
        encoded = self.tokenizer(
            prepared, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)
        source_tokens = int(encoded["attention_mask"].sum(dim=1).max().item())
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                max_new_tokens=min(512, int(source_tokens * 1.5) + 16),
            )
        results = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        if progress is not None:
            progress(len(texts))
        return results


class NllbTranslator(Translator):
    MODEL = "facebook/nllb-200-distilled-600M"
    REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
    CODES = {"en": "eng_Latn", "pt-BR": "por_Latn"}

    def __init__(self, device: str):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL, revision=self.REVISION)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.MODEL, revision=self.REVISION
        ).to(device)
        self.model.eval()
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())

    def translate(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
        progress: Callable[[int], None] | None = None,
    ) -> list[str]:
        self.tokenizer.src_lang = self.CODES[source_lang]
        target_id = self.tokenizer.convert_tokens_to_ids(self.CODES[target_lang])
        results: list[str] = []
        batch_size = 8
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            longest_source = int(encoded["attention_mask"].sum(dim=1).max().item())
            max_new_tokens = min(512, max(32, longest_source * 2 + 32))
            with torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    forced_bos_token_id=target_id,
                    num_beams=4,
                    max_new_tokens=max_new_tokens,
                )
            results.extend(
                self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
            if progress is not None:
                progress(len(batch))
        return results


class TowerTranslator(Translator):
    MODEL = "Unbabel/Tower-Plus-2B"
    REVISION = "4d779ca939174189c0677d4a75642d36d6a33b66"

    def __init__(self, device: str):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL, revision=self.REVISION)
        dtype = torch.bfloat16 if device == "cpu" else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL, revision=self.REVISION, torch_dtype=dtype
        ).to(device)
        self.model.eval()
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())

    def translate(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
        progress: Callable[[int], None] | None = None,
    ) -> list[str]:
        source_name = "English" if source_lang == "en" else "Portuguese (Brazilian)"
        target_name = "English" if target_lang == "en" else "Portuguese (Brazilian)"
        results = []
        batch_size = 4
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            source_token_ids = self.tokenizer(
                batch,
                add_special_tokens=False,
                truncation=True,
                max_length=1024,
            )["input_ids"]
            longest_source = max(len(token_ids) for token_ids in source_token_ids)
            max_new_tokens = min(512, max(32, longest_source * 2 + 32))
            prompts = []
            for text in batch:
                prompt = (
                    f"Translate the following {source_name} source text to {target_name}. "
                    "Return only the translation, without commentary.\n"
                    f"{source_name}: {text}\n{target_name}:"
                )
                prompts.append(
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                )
            encoded = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
            ).to(self.device)
            input_length = encoded["input_ids"].shape[1]
            with torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            results.extend(
                self.tokenizer.batch_decode(
                    generated[:, input_length:], skip_special_tokens=True
                )
            )
            if progress is not None:
                progress(len(batch))
        results = [result.strip() for result in results]
        return results


MODEL_SPECS = {
    "marian": ModelSpec("Marian OPUS-MT", MarianTranslator),
    "nllb": ModelSpec("NLLB-200 distilled 600M", NllbTranslator),
    "tower": ModelSpec("Tower+ 2B", TowerTranslator),
}


def score_predictions(predictions: list[str], references: list[str]) -> dict[str, float]:
    return {
        "bleu": sacrebleu.corpus_bleu(predictions, [references], tokenize="flores200").score,
        "chrf": sacrebleu.corpus_chrf(predictions, [references], word_order=2).score,
        "ter": sacrebleu.corpus_ter(predictions, [references]).score,
    }


def benchmark(
    repo: Path, model_keys: list[str], device: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    passages = load_parallel_passages(repo)
    summaries = []
    prediction_rows = []
    directions = [
        ("en", "pt-BR", "english", "portuguese"),
        ("pt-BR", "en", "portuguese", "english"),
    ]
    for model_key in model_keys:
        spec = MODEL_SPECS[model_key]
        translator = spec.loader(device)
        try:
            for source_lang, target_lang, source_column, reference_column in directions:
                sources = passages[source_column].tolist()
                references = passages[reference_column].tolist()
                started = time.perf_counter()
                predictions = translator.translate(sources, source_lang, target_lang)
                elapsed = time.perf_counter() - started
                scores = score_predictions(predictions, references)
                summaries.append(
                    {
                        "model": spec.name,
                        "direction": f"{source_lang} -> {target_lang}",
                        "passages": len(sources),
                        "parameters_millions": translator.parameter_count / 1_000_000,
                        "seconds": elapsed,
                        "passages_per_second": len(sources) / elapsed,
                        **scores,
                    }
                )
                for row, prediction, reference in zip(
                    passages.to_dict("records"), predictions, references, strict=True
                ):
                    prediction_rows.append(
                        {
                            "model": spec.name,
                            "direction": f"{source_lang} -> {target_lang}",
                            "pair_id": row["pair_id"],
                            "source": row[source_column],
                            "reference": reference,
                            "prediction": prediction,
                            "sentence_chrf": sacrebleu.sentence_chrf(
                                prediction, [reference], word_order=2
                            ).score,
                        }
                    )
        finally:
            translator.close()
            del translator
    return pd.DataFrame(summaries), pd.DataFrame(prediction_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--models", nargs="+", choices=MODEL_SPECS, default=list(MODEL_SPECS))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("posts/data"))
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, predictions = benchmark(args.repo.resolve(), args.models, args.device)
    summary_path = args.output_dir / "translation-benchmark-summary.csv"
    predictions_path = args.output_dir / "translation-benchmark-predictions.csv"
    if summary_path.exists():
        previous = pd.read_csv(summary_path)
        keys = set(zip(summary["model"], summary["direction"], strict=True))
        keep = [
            (model, direction) not in keys
            for model, direction in zip(previous["model"], previous["direction"], strict=True)
        ]
        summary = pd.concat([previous.loc[keep], summary], ignore_index=True)
    if predictions_path.exists():
        previous = pd.read_csv(predictions_path)
        keys = set(zip(predictions["model"], predictions["direction"], strict=True))
        keep = [
            (model, direction) not in keys
            for model, direction in zip(previous["model"], previous["direction"], strict=True)
        ]
        predictions = pd.concat([previous.loc[keep], predictions], ignore_index=True)
    passages = load_parallel_passages(args.repo.resolve())
    passages.to_csv(args.output_dir / "translation-benchmark-passages.csv", index=False)
    environment = {
        "platform": platform.platform(),
        "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": args.device,
        "torch_threads": torch.get_num_threads(),
        "historical_commit": HISTORICAL_COMMIT,
        "aligned_passages": len(passages),
        "english_words": int(passages["english"].str.split().str.len().sum()),
        "portuguese_words": int(passages["portuguese"].str.split().str.len().sum()),
        "model_revisions": {
            "marian_en_pt_br": MarianTranslator.MODELS[("en", "pt-BR")][1],
            "marian_pt_br_en": MarianTranslator.MODELS[("pt-BR", "en")][1],
            "nllb": NllbTranslator.REVISION,
            "tower": TowerTranslator.REVISION,
        },
    }
    (args.output_dir / "translation-benchmark-environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    summary.to_csv(summary_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
