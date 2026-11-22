from contextlib import ExitStack
from hashlib import sha256
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import preflight


REPO = Path(__file__).resolve().parents[3]
LOCAL_STAGE = REPO/"local-only/training-vggt-three-recording-native-input-v1"


class PreflightTest(unittest.TestCase):
    def test_actual_local_stage_has_exact_assets_and_overlay(self):
        data=preflight.dataset(LOCAL_STAGE/"data")
        self.assertEqual(len(data),74)
        self.assertEqual(sum(map(len,data.values())),12766333)
        self.assertEqual(preflight.digest(data["bundle.json"]),preflight.ASSETS_MANIFEST)
        self.assertEqual(len(preflight.overlay(LOCAL_STAGE/"gencuda")),6)

    def test_wrong_member_paths_and_pin_reject(self):
        for value in ("./bundle.json","a/../bundle.json","../data","e/b/","a\\b"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError,"member name"):
                preflight.relative(value)
        with self.assertRaisesRegex(ValueError,"frozen file differs"):
            preflight.read(LOCAL_STAGE/"data/bundle.json","0"*64,2048)
        with self.assertRaisesRegex(ValueError,"frozen file differs"):
            preflight.read(LOCAL_STAGE/"data/bundle.json",preflight.ASSETS_MANIFEST,100)

    def test_install_isolated_candidate_and_common_drift_before_any_write(self):
        with TemporaryDirectory() as tmp, ExitStack() as patches:
            base=Path(tmp).resolve()
            framework,upstream,weights=(base/name for name in ("framework","upstream","weights"))
            for root in (framework,upstream,weights):
                root.mkdir()
            asset={"bundle.json":b"asset"}
            overlay={name:(b"# candidate\\n"+name.encode()) for name in preflight.SOURCES}
            old={"component/"+str(i)+".py":str(i).encode() for i in range(392)}
            backend=(preflight.BASE/preflight.BACKEND).as_posix()
            old[backend]=b"old backend"
            for name,raw in old.items():
                preflight.save(framework/name,raw)
            manifest=preflight.document({"files":{
                name:{"bytes":len(raw),"SHA256":sha256(raw).hexdigest()} for name,raw in old.items()}})
            common=base/"manifest.json"
            common.write_bytes(manifest)
            up_raw=b"source"
            preflight.save(upstream/"x.py",up_raw)
            source=base/"upstream.json"
            source.write_bytes(preflight.document({"files":[{"name":"x.py","SHA256":sha256(up_raw).hexdigest()}]*69}))
            model=weights/"model.pt"
            model.write_bytes(b"weights fixture")
            identity=model.stat()
            for field,value in dict(FRAMEWORK=sha256(manifest).hexdigest(),
                                    UPSTREAM=sha256(source.read_bytes()).hexdigest(),OLD_BACKEND=sha256(old[backend]).hexdigest(),
                                    WEIGHT_STAT=(identity.st_size,identity.st_ino,identity.st_mtime_ns)).items():
                patches.enter_context(patch.object(preflight,field,value))
            patches.enter_context(patch.object(preflight,"dataset",return_value=asset))
            patches.enter_context(patch.object(preflight,"overlay",return_value=overlay))
            args=(base/"unused",framework,common,upstream,source,weights)
            candidate=base/"candidate"
            result=preflight.install(*args,candidate)
            self.assertEqual(result["final_framework_members"],398)
            self.assertEqual((candidate/"framework"/backend).read_bytes(),overlay[preflight.BACKEND])
            self.assertEqual((framework/backend).read_bytes(),b"old backend")
            self.assertEqual((candidate/"FRAMEWORK_FILES.json").read_bytes(),manifest)
            self.assertFalse(result["training_READY"])
            sha=sha256((candidate/"install.json").read_bytes()).hexdigest()
            with patch.dict(os.environ,{"SLURM_JOB_ID":"0"},clear=True):
                with self.assertRaisesRegex(ValueError,"fixed CPU-only"):
                    preflight.native(candidate,sha)
            with self.assertRaises(FileExistsError):
                preflight.install(*args,candidate)
            (framework/backend).write_bytes(b"wrong")
            other=base/"different"
            with self.assertRaisesRegex(ValueError,"frozen file differs"):
                preflight.install(*args,other)
            self.assertFalse(other.exists())

    def test_reserved_vault_and_unauthorized_target_source_cannot_pass_assets(self):
        with TemporaryDirectory() as tmp:
            src=Path(tmp).resolve()
            (src/"data").mkdir()
            (src/"data/bundle.json").write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError,"frozen file differs"):
                preflight.dataset(src/"data")
            (src/"gencuda/backends").mkdir(parents=True)
            (src/"gencuda/backends/vggt_training.py").write_bytes(b"# replacement")
            with self.assertRaisesRegex(ValueError,"frozen file differs"):
                preflight.overlay(src/"gencuda")


if __name__=="__main__":
    unittest.main()
