"""Test the replacement occurrence guards and rendered native callback anchors."""
import ast
from copy import deepcopy
import json
import unittest

import live_guard as g
import render


def fixture():
    full=[f"row-{i}".encode() for i in range(1,65)]
    original=b"\n".join(full)+b"\n"
    source=list(range(1,57))+list(range(1,9))
    stored=[full[i-1] for i in source]
    raw=b"\n".join(stored)+b"\n"
    proposed=[dict(source_ordinal=i,row_SHA256=g.digest(data)) for i,data in zip(source,stored,strict=True)]
    proof={row["source_ordinal"]:row["row_SHA256"] for row in proposed}
    first=[0,*range(2,32),56]
    second=[i for i in range(64) if i not in first]
    order=first+second
    matrix=[[] for _ in range(8)]
    for flat,physical in enumerate(order):
        rank=flat%8
        matrix[rank].append(dict(rank=rank,update=flat//32+1,accumulation=flat%32//8+1,logical=63-physical,
                                 physical=physical,**proposed[physical]))
    return raw,original,proposed,proof,matrix


class NativeProof(unittest.TestCase):
    def test_64_occurrences_allow_same_source_in_each_window(self):
        raw,original,proposed,proof,matrix=fixture()
        ordered,order=g.bind_matrix(raw,original,proposed,proof,matrix)
        self.assertEqual(order,list(range(63,-1,-1)))
        for step in (1,2):
            subset=[item for rank in ordered for i,item in enumerate(rank) if i//4==step-1]
            self.assertEqual(len(subset),32)
            self.assertEqual(len({item["ordinal"] for item in subset}),31)
        self.assertEqual(sum(len(row) for row in ordered),64)
        corrupted=deepcopy(matrix)
        corrupted[0][1]["physical"]=corrupted[0][0]["physical"]
        with self.assertRaisesRegex(ValueError,"NO_REUSED_PHYSICAL"):
            g.bind_matrix(raw,original,proposed,proof,corrupted)
        corrupted[0][1]=deepcopy(matrix[0][1])
        corrupted[0][1]["accumulation"]=4
        with self.assertRaisesRegex(ValueError,"RANK_ACCUMULATION"):
            g.bind_matrix(raw,original,proposed,proof,corrupted)
        with self.assertRaisesRegex(ValueError,"PHYSICAL_EXPOSURE"):
            g.bind_matrix(raw.replace(b"row-1\n",b"row-64\n",1),original,proposed,proof,matrix)

    def test_live_native_generator_and_both_occurrence_hooks(self):
        plan,order=g.bind_matrix(*fixture())
        correct=plan[0]
        cursor=g.LiveCursor(0,correct,order)
        with self.assertRaisesRegex(ValueError,"PREPARE_CONSUMES_EMITTED_SLOT"):
            cursor.prepare(0,correct[0])
        # Simulate the real loader emitting the first microbatch then prefetching.
        batches=[[row["logical"]] for row in correct]
        self.assertEqual(list(cursor.witness(iter(batches))),batches)
        for i,row in enumerate(correct):
            source=dict(ordinal=row["ordinal"],row_SHA256=row["row_SHA256"],after={})
            prepared=cursor.prepare(i,source)
            if i==0:
                with self.assertRaisesRegex(ValueError,"FORWARD_MATCHES_PREPARED"):
                    cursor.forward(i,dict(prepared,physical=56))
            landed=cursor.forward(i,prepared)
            self.assertEqual(landed,row)
            if i==3:
                cursor.confirmed(4)
        cursor.confirmed(8)
        other=g.LiveCursor(0,correct,order)
        with self.assertRaisesRegex(ValueError,"NATIVE_SAMPLER_LOGICAL_AND_PHYSICAL"):
            list(other.witness(iter([[63-56],*batches[1:]])))

    def test_rendered_hooks_and_source_manifest(self):
        files=render.render()
        self.assertEqual(render.render(),files)
        callback=files["callback.py"].decode()
        tree=ast.parse(callback)
        attrs={node.attr for node in ast.walk(tree) if isinstance(node,ast.Attribute)}
        self.assertIn("attach",attrs)
        self.assertNotIn("sampler_identity",attrs)
        self.assertIn("should_save",attrs)
        self.assertIn("self.live_cursor.prepare(self.inputs,observed)",callback)
        self.assertIn("self.live_cursor.forward(self.inputs,observed)",callback)
        self.assertIn("cap.live_cursor.confirmed(cap.inputs)",callback)
        self.assertIn("physical-rank-matrix.private.json",files["node.py"].decode())
        claim=json.loads(files["CLAIM.json"])
        self.assertFalse(claim["GPU_GO"])
        self.assertEqual((claim["production_workers"],claim["placement"]["port"]),(4,29985))
        with self.assertRaisesRegex(SystemExit,"DISABLED_UNTIL_FRESH_OWNER"):
            exec(compile(files["control_once.py"],"control_once.py","exec"),{})


if __name__=="__main__":
    unittest.main()
