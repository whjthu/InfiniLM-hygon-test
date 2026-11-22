"""Small fixtures for exact seals and original-inode-preserving assembly."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import assemble as model


class View(unittest.TestCase):
    def setUp(self):
        raw=(model.HERE/"ASSETS.json").read_bytes()
        self.assets=json.loads(raw)
        self.assertEqual(model.digest(raw),model.ASSETS_SHA)
        self.shard=self.assets["public_shards"]["3"]["spec"]

    def test_exact17_and_unadopted_release(self):
        self.assertEqual(len(self.assets["old14"]),14)
        self.assertEqual(sum(x["bytes"] for x in self.assets["old14"].values()),4922323586)
        self.assertEqual(sum(x["spec"]["bytes"] for x in self.assets["public_shards"].values()),12150388664)
        claim=json.loads((model.HERE/"CLAIM.json").read_bytes())
        if not claim["CPU_GO"]:
            self.assertIsNone(claim["adopted_seals_SHA256"])
        self.assertFalse(claim["GPU_GO"])
        self.assertRegex(claim["request"],r"^[0-9a-f]{32}$")

    def test_sealed_checkpoint_and_tampered_ctime_or_range(self):
        spec=self.shard
        original=dict(st_dev=8,st_ino=70,st_size=spec["bytes"],st_mtime_ns=1100,st_ctime_ns=1200)
        segments=[dict(offset=offset,bytes=min(spec["segment_bytes"],spec["bytes"]-offset),SHA256="a"*64)
                  for offset in range(0,spec["bytes"],spec["segment_bytes"])]
        state=dict(spec=spec,sealed=True,confirmed_bytes=spec["bytes"],file=[8,70],
                   sealed_stat=[1100,1200],segments=segments)
        model.verify_checkpoint(state,spec,original)
        with self.assertRaisesRegex(ValueError,"inode differs"):
            model.verify_checkpoint(state,spec,dict(original,st_ctime_ns=1300))
        with self.assertRaisesRegex(ValueError,"missing exact whole seal"):
            model.verify_checkpoint(dict(state,sealed=False),spec,original)
        bad=deepcopy(state)
        bad["segments"][-1]["offset"]+=1
        with self.assertRaisesRegex(ValueError,"range|denominator"):
            model.verify_checkpoint(bad,spec,original)

    def test_missing_shard02_stops_before_metadata_open(self):
        authority=dict(schema="INTERNVL_PUBLIC_INDEPENDENT_THREE_SEALS_V1",request=model.REQUEST,
                       seals={"3":{},"4":{}})
        with patch.object(model,"fd_info") as observed:
            with self.assertRaisesRegex(ValueError,"Three individually"):
                model.source_receipts(self.assets,authority)
        observed.assert_not_called()

    def test_only_changed_old_member_requires_targeted_rehash(self):
        originals={name:dict.fromkeys(model.STAT_FIELDS,11) for name in self.assets["old14"]}
        reverse={x["source"]:name for name,x in self.assets["old14"].items()}
        changed=next(iter(originals))
        def fake(path,**keywords):
            name=reverse[path]
            result=dict(originals[name])
            if name==changed:
                result["st_ctime_ns"]+=1
            return result,None
        with patch.object(model,"source_receipts",return_value=dict.fromkeys(("A","B","C"))), \
             patch.object(model,"old_rows",return_value=originals), patch.object(model,"fd_info",side_effect=fake):
            result=model.build_plan(self.assets,{},Path("unused"),{})
        self.assertFalse(result["prepare_allowed"])
        self.assertEqual(result["matched_old14"],13)
        [target]=result["required_targeted_rehash"]
        self.assertEqual((target["name"],target["differing_fields"]),(changed,["st_ctime_ns"]))

    def test_preparation_uses_new_regular_stubs_and_never_rewrites_sources(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            source=root/"original"
            source.write_bytes(b"unmodified")
            before=source.stat()
            target=root/"exclusive-v3"
            plan=dict(prepare_allowed=True,request=model.REQUEST,modelroot=str(target/"modelroot"),
                      members={f"shard-{i:02}.safetensors":{"source":str(source)} for i in range(17)})
            with patch.object(model,"verify_originals") as verify:
                receipt=model.prepare(plan,target)
                verify.assert_called_once_with(plan)
            leaf=target/"modelroot/shard-00.safetensors"
            self.assertTrue(leaf.is_file() and not leaf.is_symlink())
            self.assertEqual(leaf.stat().st_size,0)
            self.assertEqual(source.read_bytes(),b"unmodified")
            self.assertEqual(source.stat().st_ctime_ns,before.st_ctime_ns)
            self.assertEqual(len(receipt["binds"]),18)
            self.assertTrue(receipt["binds"][1].endswith(":/internvl-model/shard-00.safetensors:ro"))
            with self.assertRaisesRegex(ValueError,"unique"):
                model.prepare(plan,target)

    def test_mounted_path_must_be_readonly_and_match_host_stamp(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            paths=[root/f"shard-{i}" for i in range(17)]
            for path in paths:
                path.write_bytes(b"fixture")
            mounts={str(path) for path in paths}
            plan=dict(state="HOST_PLACEHOLDERS_PREPARED_CONTAINER_UNVERIFIED",request=model.REQUEST,
                      members={path.name:{"stamp":{"st_ino":3}} for path in paths})
            with patch.object(model,"fd_info",return_value=({"st_ino":3},None)):
                model.verify_mounted(plan,mounts,root)
            with self.assertRaisesRegex(ValueError,"readonly"):
                model.verify_mounted(plan,set(),root)
            with patch.object(model,"fd_info",return_value=({"st_ino":4},None)):
                with self.assertRaisesRegex(ValueError,"HOST"):
                    model.verify_mounted(plan,mounts,root)
        self.assertEqual(model.readonly_mounts(
            "11 1 0:7 /source /path ro - nfs host rw\n"
            "12 1 0:7 /other /path rw - nfs host rw\n"
            "13 1 0:7 /good /good ro - nfs host rw\n"),{"/good"})


if __name__=="__main__":
    unittest.main()
