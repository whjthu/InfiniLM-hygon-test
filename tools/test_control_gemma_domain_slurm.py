import unittest

from control_gemma_domain_slurm import arm_timeline, remaining, srun_command


def fixture(gpu_terminal=True, gpu_bound=True, gpu_passed=True, pre_passed=True, post_passed=True):
    calls, seals = [], []

    def step(phase):
        calls.append(phase)
        if phase == "gpu":
            return dict(row=["gpu"], passed=gpu_passed, terminal_visible=gpu_terminal,
                        original_step_bound=gpu_bound)
        return dict(row=[phase], passed=pre_passed if phase == "pre" else post_passed,
                    terminal_visible=True, original_step_bound=True)

    return step, calls, seals


class GemmaSlurmController(unittest.TestCase):
    def test_preflight_failure_stops_before_acquiring_eight_cards(self):
        step, calls, seals = fixture(pre_passed=False)
        self.assertEqual(set(arm_timeline(step, seals.append)), {"pre"})
        self.assertEqual(calls, ["pre"])
        self.assertEqual(seals, [])

    def test_live_or_unbound_gpu_step_retains_reservation(self):
        for change in (dict(gpu_terminal=False), dict(gpu_bound=False)):
            with self.subTest(change=change):
                step, calls, seals = fixture(**change)
                arm_timeline(step, seals.append)
                self.assertEqual(calls, ["pre", "gpu"])
                self.assertEqual(seals, [])

    def test_known_gpu_terminal_failure_still_captures_independent_post(self):
        step, calls, seals = fixture(gpu_passed=False, post_passed=False)
        stages = arm_timeline(step, seals.append)
        self.assertEqual(calls, ["pre", "gpu", "post"])
        self.assertFalse(stages["gpu"]["passed"])
        self.assertFalse(stages["post"]["passed"])
        self.assertEqual(seals, [{"pre": ["pre"], "gpu": ["gpu"], "post": ["post"]}])

    def test_srun_requests_only_the_frozen_parent_node_and_phase_resources(self):
        for phase, cards in (("pre", "none"), ("gpu", "dcu:8"), ("post", "none")):
            argv = srun_command(phase, ["fixture.py"], 75)
            self.assertEqual(argv[0], "srun")
            self.assertIn("--jobid=1103606", argv)
            self.assertIn("--nodelist=f13r1n12", argv)
            self.assertIn("--gres=" + cards, argv)
            self.assertIn("--time=00:01:15", argv)
            self.assertEqual(argv[-1], "fixture.py")

    def test_original_deadline_shrinks_and_preserves_post_budget(self):
        self.assertEqual(remaining(100, 20, 100, 15), 65)
        self.assertEqual(remaining(100, 20, 33, 15), 33)
        with self.assertRaises(TimeoutError):
            remaining(100, 85, 150, 15)
        with self.assertRaises(ValueError):
            remaining(100, float("nan"), 150, 15)


if __name__ == "__main__":
    unittest.main()
