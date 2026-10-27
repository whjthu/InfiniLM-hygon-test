import pathlib,json
def load():
 q=json.loads(pathlib.Path(__file__).with_name("case_plan_r125.json").read_bytes());expected=[("A0-M15OFF",False),("B0-M15ON",True),("B1-M15ON",True),("A1-M15OFF",False)]
 assert q["max_live"]==16 and q["warm_prompt_tokens"]==40 and q["warm_output_cap"]==2 and len(q["episodes"])==4
 for a,(label,on)in zip(q["episodes"],expected):assert a["label"]==label and type(a["m15_oproj"])is bool and a["m15_oproj"]is on and a["chunked"]is False and a["down_row2"]is True
 return q
