"""Exercise real stdlib spawn cleanup and receipt boundaries without Torch or Slurm."""
import errno
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest

import probe as p


class FakeGroup:
    def __init__(self,rank,fail_destroy=False):
        self.rank=rank
        self.active=False
        self.fail_destroy=fail_destroy

    def is_gloo_available(self):
        return True

    def is_initialized(self):
        return self.active

    def get_rank(self):
        return self.rank

    def get_world_size(self):
        return p.WORLD

    def init_process_group(self,backend,*,init_method,rank,world_size,timeout):
        assert backend=="gloo" and init_method.startswith("file://") and init_method.endswith("/group-store")
        assert rank==self.rank and world_size==p.WORLD and 0<timeout.total_seconds()<=50
        self.active=True

    def destroy_process_group(self):
        if self.fail_destroy:
            raise OSError(errno.EPERM,"private detail to hide","/never/expose/raw/file")
        self.active=False


def provider_for(rank,*,fail_destroy=False):
    group=FakeGroup(rank,fail_destroy)
    class FakeShard:
        def __init__(self,**kwargs):
            self.__dict__.update(p.FIELDS,rank=rank)

        def __len__(self):
            return 8

        def __iter__(self):
            return iter([[index] for index in range(rank,64,8)])

    torch=SimpleNamespace(distributed=group,cuda=SimpleNamespace(is_initialized=lambda:False),
                          get_rng_state=lambda:b"fixed",equal=lambda a,b:a==b)
    return torch,FakeShard


def positive_worker(here,rank,deadline):
    p.worker(here,rank,deadline,load_provider=lambda:provider_for(rank))


def mixed_worker(here,rank,deadline):
    def provider():
        if rank==0:
            time.sleep(0.3)
            raise RuntimeError("failure contains /private/service/filename")
        if rank==7:
            time.sleep(20)
        return provider_for(rank)
    p.worker(here,rank,deadline,load_provider=provider)


class StagedFixtures(unittest.TestCase):
    def test_full_exit0_and_exact_0_through_63(self):
        with TemporaryDirectory() as temporary:
            result=p.run(Path(temporary).resolve(),4,target=positive_worker)
        self.assertTrue(result["native_logical_permutation_verified"])
        self.assertEqual((result["started_ranks"],result["normal_ranks"],result["reaped_ranks"]),(8,8,8))
        self.assertEqual(result["cleanup"],[])
        self.assertEqual({i for rank in result["rank_logical_indices"] for i in rank},set(range(64)))

    def test_rank0_failure_and_other_rank_deadline_survive_collection(self):
        with TemporaryDirectory() as temporary:
            result=p.run(Path(temporary).resolve(),3.3,target=mixed_worker)
        self.assertFalse(result["native_logical_permutation_verified"])
        self.assertEqual((result["started_ranks"],result["normal_ranks"],result["reaped_ranks"]),(8,6,8))
        zero, *_, delayed = result["after_cleanup_rank_reports"]
        self.assertEqual(zero["exit_code"],2)
        self.assertEqual(zero["errors"][0]["phase"],"provider")
        self.assertTrue(any(item["function"]=="provider" for item in zero["errors"][0]["stack"]))
        self.assertEqual(delayed["entered"],["start","provider"])
        self.assertEqual(result["cleanup"],[{"rank":7,"action":"terminate"}])
        self.assertEqual(result["survivor_ranks"],[])
        self.assertNotIn("/private/service",json.dumps(result))

    def test_draw_then_destroy_failure_is_a_rank_failure_with_errno(self):
        with TemporaryDirectory() as temporary:
            here=Path(temporary).resolve()
            with self.assertRaises(SystemExit) as exited:
                p.worker(here,0,time.monotonic()+3,load_provider=lambda:provider_for(0,fail_destroy=True))
            report,indexes=p.rank_evidence(here,0,os.getpid(),exited.exception.code)
            self.assertTrue((here/"rank-0-draw-returned.json").is_file())
            self.assertFalse((here/"rank-0-terminal-normal.json").exists())
        self.assertIn("draw",report["returned"])
        self.assertIsNone(indexes)
        self.assertEqual(report["errors"][0]["errno"],errno.EPERM)
        self.assertEqual(report["errors"][0]["phase"],"destroy")
        self.assertNotIn("/never/expose",json.dumps(report))

    def test_full_exit_guard_rejects_duplicate_union(self):
        rows=[list(range(rank,64,8)) for rank in range(8)]
        rows[0][-1]=rows[1][-1]
        with self.assertRaisesRegex(ValueError,"EXACT_FULL_LOGICAL_PERMUTATION"):
            p.combine(rows)


if __name__=="__main__":
    unittest.main()
