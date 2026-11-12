import unittest
from vlm.runtime.native_caller_budget import describe_capture, plan_capture


class CaptureBudgetTests(unittest.TestCase):
    def rows(self):
        return [dict(ordinal=i,sequence_length=3363,selected_logit_rows=21,
                     collated_serialized_bytes=32*1024**2,image_tokens=3072) for i in range(1,65)]

    def plan(self, rows):
        return plan_capture(rows,parameters=43646976,tensors=504,hidden_size=4096,vocab_size=151936)

    def test_complete_f32_conservative_plan_fits_unchanged_limits(self):
        p=self.plan(self.rows())
        self.assertLess(p['per_rank_total_upper_bytes'],p['per_rank_total_limit'])
        self.assertLess(p['maximum_artifact_upper_bytes'],p['per_artifact_limit'])
        self.assertEqual(p['full_parameter_and_AdamW_payload_bytes'],15*43646976*4)
        self.assertFalse(p['actual_GPU_serialized_artifact_bytes_qualified'])

    def test_all_rows_are_required_in_order(self):
        for rows in (self.rows()[:-1],self.rows()[::-1]):
            with self.assertRaises(ValueError):self.plan(rows)

    def test_full_logits_fail_instead_of_truncating(self):
        rows=self.rows()
        for row in rows:row['selected_logit_rows']=3363
        with self.assertRaises(ValueError):self.plan(rows)

    def test_oversized_full_payload_report_is_not_an_admission_plan(self):
        rows=self.rows()
        for row in rows:row['selected_logit_rows']=3363
        description=describe_capture(rows,parameters=43646976,tensors=504,
                                     hidden_size=4096,vocab_size=151936)
        self.assertFalse(description['within_recorder_limits'])
        self.assertFalse(description['analytical_sufficiency'])
        self.assertEqual(description['per_rank_excess_upper_bytes'],
                         description['per_rank_total_upper_bytes']-8*1024**3)
        self.assertFalse(description['input_truncation_allowed'])
        self.assertFalse(description['output_truncation_allowed'])
        self.assertEqual(rows[0]['selected_logit_rows'],3363)
        with self.assertRaises(ValueError):self.plan(rows)

    def test_invalid_native_counts_fail(self):
        for key,value in [('sequence_length',4097),('selected_logit_rows',True),
                          ('selected_logit_rows',4000),('image_tokens',0),
                          ('collated_serialized_bytes',2**31)]:
            rows=self.rows();rows[3][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):self.plan(rows)


if __name__=='__main__':unittest.main()
