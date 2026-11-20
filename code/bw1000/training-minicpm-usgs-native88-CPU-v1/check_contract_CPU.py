"""Synthetic mask/EOS boundary checks only; not real MiniCPM input qualification."""
from input_contract import check_record,check_encoded


def check():
    row=dict(id="synthetic-only",messages=[dict(role="user",content="<image>\nsynthetic"),
               dict(role="assistant",content="FINAL: 1")],images=["synthetic.png"])
    check_record(row)
    # MiniCPM ChatML can end with EOS followed by a real separator/newline.
    assert check_encoded([1,2,3,9,10],[-100,-100,3,9,10],9,[9,10])==dict(length=5,supervised_tokens=3,shifted_supervised_tokens=3)
    rejected=0
    bad=((check_record,(dict(row,messages=row["messages"][:1]),)),
         (check_encoded,([1,2,9,10,77],[-100,2,9,10,77],9,[9,10])),
         (check_encoded,([1,2,9,11],[-100,2,9,11],9,[9,10])),
         (check_encoded,([1,2,9,10],[-100,2,-100,10],9,[9,10])),
         (check_encoded,([1]*4096+[9,10],[-100]*4096+[9,10],9,[9,10])))
    for function,args in bad:
        try:
            function(*args)
        except ValueError:
            rejected+=1
    assert rejected==5
    return dict(passed=True,synthetic_valid_contracts=2,rejected=5,planned_rejections=5,
                legal_EOS_then_native_separator_accepted=True,arbitrary_post_suffix_rejected=True,
                actual_native_encodes=0,actual_collator_calls=0,actual_training_updates=0,actual_GPU=0)
