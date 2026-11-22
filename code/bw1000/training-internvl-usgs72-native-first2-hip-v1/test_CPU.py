import unittest

from domain_checks import AFTER, BEFORE, after_sample, before_sample, signatures, verify_updates


class TestDomain(unittest.TestCase):
    def test_exact_roster_and_post_forward_labels(self):
        rows, source = [], []
        for i in range(1, 73):
            row = {"ordinal": i, "row_SHA256": str(i), "source_ID_SHA256": str(i)}
            source.append(row)
            rows.append({"ordinal": i, "source_row_SHA256": str(i), "source_ID_SHA256": str(i),
                         "label_and_logit_mask_native_semantics_checked": True,
                         "finite_CPU_input_fields": True,
                         "before_fields": {k: {"SHA256": f"{i}:{k}"} for k in BEFORE},
                         "after_fields": {k: {"SHA256": f"{i}:{k}"} for k in AFTER}})
        domain = signatures(rows, source)
        sha = lambda value: {"SHA256": value}
        data = {k: f"1:{k}" for k in set(BEFORE) | set(AFTER)}
        member = before_sample(data, domain, sha)
        self.assertEqual(after_sample(data, member, sha), {"ordinal": 1, "row_SHA256": "1"})
        with self.assertRaises(ValueError):
            before_sample({**data, "pixel_values": "outside-domain"}, domain, sha)
        with self.assertRaises(ValueError):
            after_sample({**data, "labels": "changed"}, member, sha)
        rows[-1]["before_fields"] = rows[0]["before_fields"]
        with self.assertRaises(ValueError):
            signatures(rows, source)

    def test_zero_lr_warmup_and_first_parameter_change(self):
        def step(params, optimizer, rates):
            return {"parameters": params, "optimizer": optimizer, "learning_rates": rates}
        a, b, c, d = step("A", "N", [0.0]), step("A", "B", [0.0]), step("A", "B", [1e-5]), step("D", "C", [1e-5])
        verify_updates(a, b, c, d)
        for broken in (step("A", "C", [1e-5]), step("D", "B", [1e-5])):
            with self.assertRaises(ValueError):
                verify_updates(a, b, c, broken)


if __name__ == "__main__":
    unittest.main()
