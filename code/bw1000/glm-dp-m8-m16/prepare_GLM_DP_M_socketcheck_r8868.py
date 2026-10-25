import pathlib
r=pathlib.Path(__file__).resolve().parent
s=(r/'check_GLM_DP_protocol_r8864.py').read_text().replace('r8864','r8868').replace("r/'GLM-exact-budget-Model-private-r8855'","r/'GLM-DP-Mmatrix-private-r8868'")
s=s.replace('for phase in PHASES:\n   ticket',"for phase in PHASES:\n   micro=16 if phase=='jointM16'else 8;cohorts=[list(range(x,min(x+micro,16)))for x in range(0,16,micro)]\n   ticket")
s=s.replace('cohorts=[list(range(8)),list(range(8,16))]','cohorts=cohorts').replace("if phase=='joint':joint_emitters.wait()","joint_emitters.wait()")
s=s.replace('for cohort in range(2):','for cohort in range(len(cohorts)):').replace('rows=list(range(8*cohort,8*cohort+8));','rows=cohorts[cohort];').replace("'ids':[2+step]*8","'ids':[2+step]*len(rows)").replace("['ids']==[2+step]*8","['ids']==[2+step]*len(cohorts[cohort])")
s=s.replace('len(co.records)==24','len(co.records)==20').replace("x['phase']=='joint'","x['phase']=='jointM16'").replace("'isoA'","'jointM8'").replace("'isolatedA_isolatedB_joint':True","'jointM8_jointM16_noisolated':True")
(r/'check_GLM_DP_protocol_r8868.py').write_bytes(s.encode())
