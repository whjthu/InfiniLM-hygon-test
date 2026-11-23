"""Derive a fixed same-input baseline control and a key/value-only attention candidate."""

from hashlib import sha256
import ast


PRIOR_OBSERVER_SHA = "520eb4d0f6457176ed4e4eb98ff0a2bcaea1284f2ad273f3efe894ffd7085e37"


def patch(source, before, after):
    if source.count(before) != 1:
        raise ValueError("The qualified native candidate source anchor differs")
    return source.replace(before, after, 1)


def backend_renderer(previous):
    modified = patch(previous.decode(), "from .hunyuanvideo_native_prefix import",
                     "from .hunyuanvideo_native_kv_prefix import")
    ast.parse(modified)
    return modified.encode()


def observer(previous):
    if sha256(previous).hexdigest() != PRIOR_OBSERVER_SHA:
        raise ValueError("The consumed Hygon native pair source differs")
    src = previous.decode()
    replacements = (
        ('ENV = "HUNYUAN_T09_NATIVE_PREFIX"', 'ENV = "HUNYUAN_T09_NATIVE_KV_PREFIX"'),
        (
            "        inputs = (t[..., :layout.valid, :] for t in (query, key, value))\n"
            "        compact = functional.scaled_dot_product_attention(\n"
            "            *inputs, attn_mask=None, dropout_p=0.0, is_causal=False)\n"
            "        return functional.pad(compact, (0, 0, 0, layout.length - layout.valid))\n",
            "        return functional.scaled_dot_product_attention(\n"
            "            query, key[..., :layout.valid, :], value[..., :layout.valid, :],\n"
            "            attn_mask=None, dropout_p=0.0, is_causal=False)\n",
        ),
        ("    def check_pair(self, base, prefix, initial):",
         '    def check_pair(self, base, prefix, initial, *, receipt="first-update.json"):'),
        ('        accepted = all(row["passed"] for row in groups.values())',
         '        control = receipt == "baseline-control.json"\n'
         '        thresholds = (0.0001, 0.001) if control else (RELATIVE_L2_LIMIT, MAX_TENSOR_RELATIVE_L2_LIMIT)\n'
         '        for row in groups.values():\n'
         '            row["passed"] = row["relative_l2"] <= thresholds[0] and row["maximum_per_tensor_relative_l2"] <= thresholds[1]\n'
         '        accepted = all(row["passed"] for row in groups.values())'),
        ('        save(self.receipts / "first-update.json", {','        save(self.receipts / receipt, {'),
        ('            "tolerance": {"relative_l2": RELATIVE_L2_LIMIT, "max_tensor_relative_l2": MAX_TENSOR_RELATIVE_L2_LIMIT},',
         '            "tolerance": {"relative_l2": thresholds[0], "max_tensor_relative_l2": thresholds[1]},'),
        ('            raise ValueError("Native same-input prefix first update did not pass")',
         '            raise ValueError("Native same-input baseline control did not pass" if control\n'
         '                             else "Native same-input key/value prefix first update did not pass")'),
        ('                self.restore(initial)\n        self.mode, self.first = "prefix", {} if step == 1 else None',
         '                self.restore(initial)\n'
         '            self.mode, self.first = "baseline", {}\n'
         '            try:\n'
         '                train_one(step)\n'
         '                control = self.first\n'
         '            finally:\n'
         '                self.restore(initial)\n'
         '            self.check_pair(baseline, control, initial["parameters"], receipt="baseline-control.json")\n'
         '        self.mode, self.first = "prefix", {} if step == 1 else None'),
    )
    for before, after in replacements:
        src = patch(src, before, after)
    ast.parse(src)
    return src.encode()
