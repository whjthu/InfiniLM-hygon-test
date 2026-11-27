"""Verify projection and fail-closed geometry without launching Torch or Slurm."""
import unittest
from types import SimpleNamespace

import probe as p


class ShardFixture:
    def __init__(self, **kwargs):
        assert kwargs == dict(total_samples=64,batch_size=1,shuffle=True,drop_last=False,
                              data_seed=42,tp_size=1,group_by_length=False,lengths=None)
        self.__dict__.update(p.FIELDS,rank=2)

    def __len__(self):
        return 8

    def __iter__(self):
        return iter([[index] for index in range(2,64,8)])


class GuardTests(unittest.TestCase):
    def test_disjoint_actual_output_requires_complete_logical_geometry(self):
        rows=[list(range(rank,64,8)) for rank in range(8)]
        result=p.combine(rows)
        self.assertEqual(result["rank_slots"],[8]*8)
        self.assertTrue(result["matches_unshuffled_interleave"])
        self.assertFalse(result["actual_loader_invoked"] or result["actual_arrow_mapping_verified"] or result["training_ready"])
        rows[0],rows[1]=rows[1],rows[0]
        self.assertFalse(p.combine(rows)["matches_unshuffled_interleave"])
        rows[0][0]=rows[1][0]
        with self.assertRaisesRegex(ValueError,"FULL_LOGICAL_PERMUTATION"):
            p.combine(rows)
        with self.assertRaisesRegex(ValueError,"EXACT_EIGHT_RANKS"):
            p.combine(rows[:7])

    def test_injected_shape_rng_and_rank_boundaries(self):
        states=iter([b"same",b"same"])
        torch=SimpleNamespace(distributed=SimpleNamespace(get_rank=lambda:2,get_world_size=lambda:8),
                              get_rng_state=lambda:next(states),equal=lambda a,b:a==b,
                              cuda=SimpleNamespace(is_initialized=lambda:False))
        self.assertEqual(p.rank_draw(torch,ShardFixture,2),list(range(2,64,8)))
        with self.assertRaisesRegex(ValueError,"CPU_PROCESS_GROUP"):
            p.rank_draw(torch,ShardFixture,3)
        states=iter([b"before",b"after"])
        with self.assertRaisesRegex(ValueError,"CPU_RNG"):
            p.rank_draw(torch,ShardFixture,2)


if __name__=="__main__":
    unittest.main()
