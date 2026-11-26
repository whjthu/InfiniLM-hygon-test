"""Exercise the independent payload reader and nine-frame order using synthetic fixtures."""

import json
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import contract as c
from worker import decode_frames
from readback import review


REQ="0"*32
NOW=1795737604


def reject(fn,wanted):
    try:
        fn()
    except ValueError as ex:
        assert str(ex)==wanted, str(ex)
    else:
        raise AssertionError(wanted)


def check():
    auth=dict(request=REQ,label=c.LABEL,mode="inspect",recipe=c.RECIPE,root_authorized=True,
              network_GO=False,training_GO=False,text_GO=False,zero_GRES_GO=True,GPU_GO=False,
              weight_payload_GO=False,allocatable_DCU=0,epoch=NOW-4,deadline=NOW+30,
              media_receipt_SHA256=c.RENDER_RECEIPT_SHA,upstream_binding_SHA256=c.SOURCE_BINDING_SHA,
              vae_metadata_SHA256=c.VAE_READBACK_SHA,original_full_weight_SHA256=c.WEIGHTS_PIN["SHA256"])
    assert c.permit(c.encoded(auth),REQ,"inspect",NOW)==auth
    reject(lambda:c.permit(c.encoded(auth),REQ,"encode",NOW),"EXACT_MODE_SPECIFIC_ROOT_PERMISSION")
    reject(lambda:c.permit(c.encoded(auth|{"GPU_GO":True}),REQ,"inspect",NOW),"EXACT_MODE_SPECIFIC_ROOT_PERMISSION")
    reject(lambda:c.permit(c.encoded(auth),REQ,"inspect",NOW+40),"EXACT_MODE_SPECIFIC_ROOT_PERMISSION")
    good=struct.pack("<ff",0.1,-0.4)*(c.BYTES//8)
    assert c.payload_summary(good,c.SHAPE)["elements"]==152640
    reject(lambda:c.payload_summary(good,[1,32,3,53,30]),"EXACT_LATENT_SHAPE_AND_SIZE")
    reject(lambda:c.payload_summary(b"\0"*c.BYTES,c.SHAPE),"LATENT_CONSTANT_OR_EMPTY")
    reject(lambda:c.payload_summary(struct.pack("<f",float("nan"))+good[4:],c.SHAPE),"LATENT_NONFINITE")
    with TemporaryDirectory() as t:
        folder=Path(t)
        rows=[]
        pins=[]
        for i in range(9):
            image=Image.new("RGB",(848,480),(i,0,0))
            name=f"frame-{i:02d}.private.png"
            image.save(folder/name)
            pins.append(c.pin(image.tobytes())["SHA256"])
            rows.append(dict(name=name,index=i,azimuth_degrees=i*15,pin=c.pin((folder/name).read_bytes())))
        ledger=b"".join(c.encoded(row) for row in rows)
        (folder/"frames.jsonl").write_bytes(ledger)
        raw=c.encoded(dict(device="CPU",frame_count=9,ledger_SHA256=c.pin(ledger)["SHA256"]))
        (folder/"render.json").write_bytes(raw)
        with patch.object(c,"RENDER_RECEIPT_SHA",c.pin(raw)["SHA256"]),patch.object(c,"RGB",pins):
            frames,meta=decode_frames(folder)
            assert frames.shape==(9,480,848,3) and list(frames[:,0,0,0])==list(range(9))
            assert meta["decoded_RGB_SHA256"]==pins
            (folder/"frames.jsonl").write_bytes(b"".join(c.encoded(row) for row in rows[::-1]))
            reject(lambda:decode_frames(folder),"EXACT_RENDER_LEDGER")

        worker=dict(schema="HUNYUAN_ABO_VAE_CANDIDATE_WORKER_V1",request=REQ,mode="encode",actual_step="1.2",
            recipe=c.RECIPE,passed=True,phase="sealed",epoch=NOW-4,deadline=NOW+30,node="f13r3n13",
            planned_frames=9,validated_frames=9,actual_candidate_latents=1,training_rows=0,optimizer_updates=0,
            actual_native_encode_calls=1,native_encode_attempts=1,actual_GPU=1,
            text_embeddings=0,old_cache_equivalence_verified=False,media={"decoded_RGB_SHA256":list(c.RGB)},
            source={"active_class_source":c.SOURCE_PIN},latent=c.payload_summary(good,c.SHAPE),
            asset=dict(config=c.CONFIG_PIN,header=c.HEADER_PIN,weight=c.WEIGHTS_PIN,current_weight_bytes=c.WEIGHTS_PIN["bytes"],
                actual_whole_weight_SHA256=c.WEIGHTS_PIN["SHA256"]),
            runtime=dict(actual_native_encode_calls=1,device="cuda:0",architecture="gfx936",config_scale=1.03682,
                         config_shift=0.0,posterior="mode"))
        worker_pin=c.write(folder/"worker.json",c.encoded(worker))
        c.write(folder/c.LATENT,good)
        terminal=dict(request=REQ,TZ="UTC",independent=True,worker_SHA256=worker_pin["SHA256"],
            accounting=["1.2","fixture","COMPLETED","0:0","2","f13r3n13","cpu=8,gres/dcu=1,mem=64G,node=1",
                        "2026-11-27T00:00:01","2026-11-27T00:00:03"])
        term_pin=c.write(folder/"terminal.json",c.encoded(terminal))
        expected=review(folder,REQ,worker_pin["SHA256"],folder/"terminal.json",term_pin["SHA256"])
        actual=subprocess.check_output([sys.executable,"-I","-S","-B",str(HERE/"readback.py"),"--request",REQ,
                 "--worker-dir",str(folder),"--worker-sha256",worker_pin["SHA256"],"--terminal",str(folder/"terminal.json"),
                 "--terminal-sha256",term_pin["SHA256"],"--out",str(folder/"review.json")])
        assert json.loads(actual)["native_encode_calls"]==1 and json.loads((folder/"review.json").read_bytes())==expected
        wrong=terminal | {"accounting":terminal["accounting"][:6]+["cpu=8,mem=64G,node=1"]+terminal["accounting"][7:]}
        tpin=c.write(folder/"wrong.json",c.encoded(wrong))
        reject(lambda:review(folder,REQ,worker_pin["SHA256"],folder/"wrong.json",tpin["SHA256"]),"EXACT_WORKER_TRES")
        (folder/c.LATENT).write_bytes(good[4:]+good[:4])
        reject(lambda:review(folder,REQ,worker_pin["SHA256"],folder/"terminal.json",term_pin["SHA256"]),"EXACT_FILE_EXTENT_OR_SHA")
    print("PASS synthetic nine frame order, mode flags, payload numeric and fresh independent reader; no weight or model access")


if __name__=="__main__":
    check()
