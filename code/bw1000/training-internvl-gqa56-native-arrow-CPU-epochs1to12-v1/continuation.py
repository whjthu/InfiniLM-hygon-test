"""Bind the actual CP3 cursor and project epochs 1–12 into steps 4–25."""
import json
from pathlib import Path

from lineage import check_matrix, check_permutation, combine, digest, encode, need


ROOT=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
EPOCH1_REQUEST="c6d2ed29e9454b72a3ba57be5a6c3f1a"
EPOCH1_ROOT=ROOT/f"incoming/internvl-gqa56-native-arrow-epoch1-{EPOCH1_REQUEST}/output"
EPOCH1_RESULT="0edb15c8762aac40a9fab7954cac0a62bf898a5ab263a04b58342d8b9dc5be3a"
EPOCH1_MATRIX="30f9d0c16edd2926d51a0df98847414a42a4376e77343605f1eda630881943cf"
EPOCH1_ARCHIVE="49f2fb9a42de79269ce723d9c9df2e0c44c8e1e8d3a81e8b7f856465c2702730"
CP3_REQUEST="4c23112f1a064af6bfd27d23e55ef89e"
CP3_NATIVE="932a8579bb727ae8f44f78ccccafbdb6b28aa35b18dfa54bd16c70b4e8d8b345"
CP3_INDEPENDENT="0a2e760e8ca017b773112a66109e4d700c4c31b67879b376c8b1fbaa06b32494"
ARROW_ORDER="5d7ec6e740dfca0cf3004fb0d979fd37ce0d8878ef9a6af1e46b24a89ee33f4e"
EPOCHS=tuple(range(1,13))
STEPS=tuple(range(4,26))


def restored_cursor(receipt):
    need(receipt["rc"]==0 and receipt["actual_step"]=="1103606.4728" and
         receipt["original_authoritative_request"]==CP3_REQUEST and
         (receipt["global_step"],receipt["trainer_epoch"],receipt["planned_steps"])==(3,1.5,25)
         and (receipt["ranks"],receipt["accepted_forward_slots"])==(8,32)
         and receipt["matched_all_final_parameter_hashes"] is True,
         "EXACT_INDEPENDENT_4728_STEP3_EPOCH1_5")
    return dict(global_step=3,epoch=1.5,max_steps=25)


def epoch1_reference(pin):
    result=json.loads(pin(EPOCH1_ROOT/"result.json",6740,EPOCH1_RESULT))
    matrix=pin(EPOCH1_ROOT/"physical-rank-matrix.private.json",10610,EPOCH1_MATRIX)
    need((result["request"],result["actual_step"],result["epoch_index"],result["data_seed"],
          result["expected_sampler_seed"])==(EPOCH1_REQUEST,"1103606.4726",1,42,43) and
         result["actual_arrow_mapping_verified"] is True and
         result["actual_arrow_logical_to_physical_SHA256"]==ARROW_ORDER and
         result["actual_rank_occurrence_SHA256"]==EPOCH1_MATRIX,
         "EXACT_ORIGINAL_4726_REFERENCE_RESULT")
    return matrix


def assemble(reports, expected, reference_epoch1, epoch1_logical):
    need(len(reports)==8,"EXACT_EIGHT_EPOCH_RANKS")
    order=None
    for rank, report in enumerate(reports):
        current=check_permutation(report["order"])
        need((order is None or current==order) and digest(encode(current))==ARROW_ORDER and
             report["rank"]==rank and report["physical_values_verified"]==64 and
             len(report["epochs"])==12,"EXACT_SAME_ARROW_AND_ALL_TWELVE_EPOCHS")
        order=current
    source_ordinals={value[1]["source_ordinal"] for value in expected.values()}
    steps={step:[] for step in STEPS}
    digests=[]
    for index,epoch in enumerate(EPOCHS):
        matrix=[]
        for rank,report in enumerate(reports):
            observation=report["epochs"][index]
            need((observation["epoch_index"],observation["data_seed"],observation["sampler_seed"])==
                 (epoch,42,42+epoch) and len(observation["rows"])==8,"ACTUAL_EPOCH_INDEX_AND_SEED")
            verified=[]
            for offset,cell in enumerate(observation["rows"]):
                logical=cell["logical"]
                need(type(logical) is int and 0<=logical<64,"EXACT_BOUNDED_LOGICAL_INDEX")
                expected_row=dict(rank=rank,update=offset//4+1,accumulation=offset%4+1,logical=logical,
                                  **expected[order[logical]][1])
                need(cell==expected_row,"RANK_EPOCH_PHYSICAL_SOURCE_LINEAGE")
                verified.append(cell)
                global_step=epoch*2+offset//4+1
                if global_step in steps:
                    steps[global_step].append(dict(cell,epoch_index=epoch,rank_offset0=offset,update=global_step))
            matrix.append(verified)
        logical=check_matrix([[cell["logical"] for cell in rank] for rank in matrix])
        raw=encode(matrix)
        if epoch==1:
            need(raw==reference_epoch1 and logical==epoch1_logical,"BYTE_EXACT_EPOCH1_ORIGINAL_4726")
        stats=combine(matrix,source_ordinals)
        digests.append(dict(epoch_index=epoch,sampler_seed=42+epoch,rank_slots=stats["physical_occurrences"],
                            unique_source_rows=stats["unique_source_rows"],internal_source_rows=0,
                            rank_matrix_SHA256=digest(raw)))
    updates=[]
    for step,rows in steps.items():
        rows.sort(key=lambda cell:(cell["accumulation"],cell["rank"]))
        need([(cell["accumulation"],cell["rank"]) for cell in rows]==
             [(micro,rank) for micro in range(1,5) for rank in range(8)],"EXACT_GLOBAL_UPDATE_32_POSITIONS")
        updates.append(dict(global_step=step,slots=rows))
    need(sum(len(x["slots"]) for x in updates)==704,"EXACT_704_CONTINUATION_SLOTS")
    return dict(start_global_step=3,start_epoch=1.5,max_steps=25,updates=updates),digests
