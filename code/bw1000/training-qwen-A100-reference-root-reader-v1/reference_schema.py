"""Declared denominators for the two-update, eight-rank original reference."""
def expected_files():
    return {"initial.pt",*[f"forward-{i:02d}-{part}.pt" for i in range(1,9) for part in ("input","output")],
            *[f"update-{i:02d}-{part}.pt" for i in (1,2) for part in ("preclip","postclip","before-optimizer","after-optimizer")]}
def validate_capture(receipt,rank):
    assert receipt["state"]=="TWO_NATIVE_OPTIMIZER_UPDATES_CAPTURED"
    assert receipt["rank"]==rank and receipt["world_size"]==8 and receipt["completed_global_step"]==2
    assert receipt["micro_forwards"]==8 and receipt["optimizer_updates"]==receipt["clip_calls"]==2
    rows=receipt["files"];assert len(rows)==25
    assert {x["name"] for x in rows}==expected_files()
    for row in rows:
        assert 0<row["bytes"]<2*1024**3
        assert len(row["SHA256"])==64 and all(x in "0123456789abcdef" for x in row["SHA256"])
    assert sum(x["bytes"] for x in rows)<=8*1024**3
def validate_native25(row,rank):
    assert row["rank"]==rank and row["world_size"]==8 and row["global_step"]==25
    assert row["effective_batch"]==32 and row["checkpoint_steps"]==[10,20,25]
    assert row["trainable_parameters"]==43646976
