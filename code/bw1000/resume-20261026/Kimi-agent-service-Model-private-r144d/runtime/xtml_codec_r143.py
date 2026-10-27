"""Exact K3 segment/BPE encoding without importing Transformers or Torch.

Formatting code is the pinned official encoding_k3 snapshot. The model supplies
the real tiktoken merges and added-token IDs. This is text-only P0 serving.
"""
import base64
import hashlib
import json
import pathlib

from frozen_encoding_k3_r143 import build_chat_segments

PATTERN = "|".join([
    r"[\p{Han}]+",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"\p{N}{1,3}", r" ?[^\s\p{L}\p{N}]+[\r\n]*", r"\s*[\r\n]+", r"\s+(?!\S)", r"\s+",
])


class KimiXTMLCodec:
    def __init__(self, model_dir):
        import tiktoken
        self.path = pathlib.Path(model_dir)
        config_path = self.path / "tokenizer_config.json"
        merges_path = self.path / "tiktoken.model"
        config = json.loads(config_path.read_bytes())
        merges = {}
        with merges_path.open("rb") as f:
            for line in f:
                token, rank = line.split()
                merges[base64.b64decode(token)] = int(rank)
        assert len(merges) == 163584, ("unexpected K3 BPE vocabulary", len(merges))
        added = config["added_tokens_decoder"]
        special = {added.get(str(i), {}).get("content", "<|reserved_token_%d|>" % i): i
                   for i in range(len(merges), len(merges) + 256)}
        for token, expected in {"[BOS]": 163584, "[EOS]": 163585,
                                "<|end_of_msg|>": 163586, "<|open|>": 163587,
                                "<|close|>": 163588, "<|sep|>": 163589}.items():
            assert special[token] == expected, token
        self.encoding = tiktoken.Encoding(name="KimiK3-P0-pinned-merges", pat_str=PATTERN,
                                           mergeable_ranks=merges, special_tokens=special)
        self.eos_token_id = 163585
        self.eos_token_ids = {163585, 163586}
        self.model_max_length = 1024
        self.identity = {
            "model_directory": str(self.path),
            "tokenizer_config_SHA256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "tiktoken_model_SHA256": hashlib.sha256(merges_path.read_bytes()).hexdigest(),
            "base_tokens": len(merges), "special_tokens": len(special),
            "Torch_imported": False, "Transformers_imported": False,
            "algorithm": "official K3 segment-aware structural-special vs ordinary text BPE",
        }

    def encode_chat(self, messages, chat_template_kwargs=None, add_generation_prompt=True):
        options = dict(chat_template_kwargs or {})
        if "enable_thinking" in options:
            options["thinking"] = options.pop("enable_thinking")
        if "reasoning_effort" in options:
            options["thinking_effort"] = options.pop("reasoning_effort")
        options.setdefault("thinking", True)
        if options["thinking"]:
            options.setdefault("thinking_effort", "max")
        segments = build_chat_segments(messages, add_generation_prompt=add_generation_prompt, **options)
        ids = []
        for segment in segments:
            ids.extend(self.encoding.encode(segment.text, allowed_special="all")
                       if segment.allow_special else
                       self.encoding.encode(segment.text, disallowed_special=()))
        return "".join(segment.text for segment in segments), ids

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=True, **kwargs):
        text, ids = self.encode_chat(conversation, kwargs, add_generation_prompt)
        return ids if tokenize else text

    def decode(self, ids, skip_special_tokens=False, **kwargs):
        values = [int(x) for x in ids]
        if skip_special_tokens:
            values = [x for x in values if x not in {163584, 163585, 163586}]
        return self.encoding.decode(values)

    def encode(self, text, add_special_tokens=False, **kwargs):
        assert not add_special_tokens, "K3 encoder adds no implicit BOS"
        return self.encoding.encode(text, allowed_special="all")
