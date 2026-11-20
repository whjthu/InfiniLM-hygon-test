"""Build a private single-file provider overlay. Refuse unknown original bytes."""

import argparse
import ast
import hashlib
from pathlib import Path


SOURCE_SHA256 = "f39b3621fea3f757a01d2e00ebd3a2be3516b7df3048d862a365988aba9fa994"
LAYER_SOURCE_SHA256 = "f56c853f48f1d9a86026aa0f88ed1c33e629483e4a84b933cd4dddbcb00e1500"
NAME = "modeling_qwen3_vl.py"
CALL = """        from qwen_paired_layers import try_paired_layers
        paired_applied, hidden_states = try_paired_layers(
            self, hidden_states, attention_mask, text_position_ids, position_embeddings,
            past_key_values, use_cache, visual_pos_masks, deepstack_visual_embeds, kwargs,
            decoder_class=Qwen3VLTextDecoderLayer, checkpoint_base=GradientCheckpointingLayer,
        )
        if not paired_applied:
"""


def render(source, layer_source):
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("unexpected Qwen provider bytes")
    if hashlib.sha256(layer_source).hexdigest() != LAYER_SOURCE_SHA256:
        raise ValueError("unexpected gradient checkpoint provider bytes")
    text = source.decode("utf-8")
    tree = ast.parse(text)
    cls, = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen3VLTextModel"]
    forward, = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward"]
    loop, = [node for node in forward.body if isinstance(node, ast.For)
             and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name)
             and node.iter.func.id == "enumerate" and not node.iter.keywords
             and len(node.iter.args) == 1 and isinstance(node.iter.args[0], ast.Attribute)
             and node.iter.args[0].attr == "layers" and isinstance(node.iter.args[0].value, ast.Name)
             and node.iter.args[0].value.id == "self"]
    begin, end = loop.lineno - 1, loop.end_lineno
    lines = text.splitlines(keepends=True)
    fallback = "".join("    " + line if line.strip() else line for line in lines[begin:end])
    candidate = "".join(lines[:begin]) + CALL + fallback + "".join(lines[end:])
    ast.parse(candidate)
    return candidate.encode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("original", type=Path)
    parser.add_argument("layer_provider", type=Path)
    parser.add_argument("private_output", type=Path)
    args = parser.parse_args()
    if args.private_output.name != NAME:
        parser.error(f"private output must be named {NAME}")
    overlay = render(args.original.read_bytes(), args.layer_provider.read_bytes())
    with args.private_output.open("xb") as stream:
        stream.write(overlay)
    print(hashlib.sha256(overlay).hexdigest())


if __name__ == "__main__":
    main()
