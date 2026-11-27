"""Test v5 mount ownership and same-FileStore receipts without Torch or Slurm."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest

import probe as p


EXPECTED=dict(dev=12,inode=34,uid=56,mount_id=78,store_path_SHA256="a"*64)
VALID="78 2 0:12 / /dev/shm rw,nosuid,nodev - tmpfs shm rw,size=16m"


def fake_worker(here,rank,deadline,store,expected):
    for phase in p.PHASES:
        file_id=(dict(store=expected) if phase=="init" else {})
        p.mark(here,rank,phase,"entered",**file_id)
        details={
            "init":{"elapsed_ns":100},
            "destroy":{"performed":True},
            "draw":{"indexes":list(range(rank,64,8))},
        }.get(phase,{})
        p.mark(here,rank,phase,"returned",**details)
    p.mark(here,rank,"terminal","normal",slots=8)


def mismatched_worker(here,rank,deadline,store,expected):
    if rank==3:
        expected={**expected,"inode":expected["inode"]+1}
    fake_worker(here,rank,deadline,store,expected)


class StoreTests(unittest.TestCase):
    def test_root_and_effective_owner_preserve_sticky_shared_protection(self):
        p.require_shm_protection(0,0o1777,1234)
        p.require_shm_protection(1234,0o1777,1234)
        with self.assertRaisesRegex(ValueError,"EXACT_SHM_OWNER_NOT_ROOT_OR_EFFECTIVE_UID"):
            p.require_shm_protection(9999,0o1777,1234)
        for mode in (0o0777,0o1775,0o1776):
            with self.subTest(mode=mode),self.assertRaisesRegex(ValueError,"EXACT_SHM_STICKY_SHARED_PROTECTION"):
                p.require_shm_protection(1234,mode,1234)

    def test_effective_tmpfs_and_mount_mismatches(self):
        p.require_mount(VALID,78,"0:12",False)
        invalid=[(VALID,77,"0:12",False),(VALID,78,"0:13",False),(VALID,78,"0:12",True),
                 (VALID.replace("tmpfs","nfs4"),78,"0:12",False),
                 (VALID.replace(" / /dev/shm "," /shared /dev/shm "),78,"0:12",False),
                 (VALID.replace("rw,nosuid","ro,nosuid"),78,"0:12",False),
                 (VALID+"\n"+VALID,78,"0:12",False)]
        for args in invalid:
            with self.subTest(args=args),self.assertRaises(ValueError):
                p.require_mount(*args)

    def test_real_spawn_reaped_before_retire_and_one_mismatched_rank_fails(self):
        for actor in (fake_worker,mismatched_worker):
            with TemporaryDirectory() as temporary:
                here=Path(temporary).resolve()
                independent=here/"shared"
                independent.mkdir()
                fake_local=here/"shm"
                fake_local.mkdir()
                retires=[]
                def retire(store,expected):
                    self.assertEqual((store,expected),(fake_local,EXPECTED))
                    self.assertEqual(len(list(independent.glob("rank-*-terminal-normal.json"))),8)
                    retires.append(True)
                    return dict(state="RETIRED")
                result=p.run(independent,4,target=actor,allocate=lambda:(fake_local,EXPECTED),retire=retire)
                self.assertEqual((result["started_ranks"],result["reaped_ranks"],retires),(8,8,[True]))
                if actor==fake_worker:
                    self.assertTrue(result["native_logical_permutation_verified"])
                    self.assertEqual([r["local_filestore"] for r in result["after_cleanup_rank_reports"]],
                                     [EXPECTED]*8)
                    self.assertEqual([r["gloo_init_elapsed_ns"] for r in result["after_cleanup_rank_reports"]],
                                     [100]*8)
                else:
                    self.assertFalse(result["native_logical_permutation_verified"])
                    self.assertEqual(result["normal_ranks"],7)
                    errors=result["after_cleanup_rank_reports"][3]["errors"]
                    self.assertEqual(errors[0]["error_code"],"EXACT_SAME_LOCAL_FILESTORE_DIRECTORY")

    def test_worker_mismatched_directory_refuses_before_init_call(self):
        with TemporaryDirectory() as temporary:
            here=Path(temporary).resolve()
            touched=[]
            dist=SimpleNamespace(init_process_group=lambda *a,**k:touched.append(True),is_initialized=lambda:False)
            def verify(path,expected):
                p.require_same_store({**expected,"dev":13},expected)
            with self.assertRaises(SystemExit):
                p.worker(here,0,time.monotonic()+3,here,EXPECTED,
                         load_provider=lambda:(SimpleNamespace(distributed=dist),object()),verify_store=verify)
            self.assertTrue((here/"rank-0-init-error.json").is_file())
            self.assertEqual(touched,[])

    @unittest.skipUnless(os.name=="posix" and Path("/dev/shm").is_dir(),"requires Linux /dev/shm")
    def test_real_private_tmpfs_and_preserve_unknown(self):
        local,expected=p.create_store()
        unknown=local/"unknown"
        try:
            self.assertEqual(p.check_store(local,expected),expected)
            unknown.write_bytes(b"retain")
            with self.assertRaisesRegex(ValueError,"EXACT_NO_UNKNOWN_FILESTORE_RESIDUE"):
                p.cleanup_store(local,expected)
            self.assertEqual(unknown.read_bytes(),b"retain")
            unknown.unlink()
            (local/p.STORE).write_bytes(b"known")
            self.assertTrue(p.cleanup_store(local,expected)["known_store_removed"])
        finally:
            if unknown.exists():
                unknown.unlink()
            if local.exists():
                p.cleanup_store(local,expected)


if __name__=="__main__":
    unittest.main()
