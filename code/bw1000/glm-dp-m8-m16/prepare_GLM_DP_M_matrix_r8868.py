import pathlib,json,hashlib,ast
r=pathlib.Path(__file__).resolve().parent;old=r/'GLM-DP-Model-private-r8864';native=r/'GLM-M16-Model-private-r8866';d=r/'GLM-DP-Mmatrix-private-r8868';d.mkdir(exist_ok=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();write=lambda p,x:p.write_bytes((json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode());parent=json.loads((old/'GLM-CROSSREQUEST-Model-source-manifest-r8864.json').read_bytes());nm=json.loads((native/'GLM-CROSSREQUEST-Model-source-manifest-r8866.json').read_bytes());oldmh=sha(old/'GLM-CROSSREQUEST-Model-source-manifest-r8864.json')
for n in list(parent['source_SHA256'])+['dispatch_GLM_resident_exact_Model_r8864.py']:
 raw=(old/n).read_bytes()
 if 'r8864'in n:raw=raw.replace(b'r8864',b'r8868')
 raw=raw.replace(b'candidates/glm-exact-budget-r8854/libglm_exact_budget_r8854.so',b'candidates/glm-M16-r8861/libglm_M16_r8861.so').replace(parent['model_SHA256'].encode(),nm['model_SHA256'].encode())
 (d/n.replace('r8864','r8868')).write_bytes(raw)
for stem in('glm_crossrequest_control','glm_crossrequest_resident_base','glm_model_library_metadata','GLM_transient_preflight'):
 (d/(stem+'_r8868.py')).write_bytes((native/(stem+'_r8866.py')).read_bytes().replace(b'r8866',b'r8868'))
def edit(name,fn):
 p=d/name;p.write_bytes(fn(p.read_text()).encode())
def probe(s):
 s=s.replace('prefill_micro in(4,8)and micro==8','prefill_micro==8 and micro in(8,16)').replace("len(e['rows'])==8)","len(e['rows'])in(8,16))")
 s=s.replace("labels=[f'perf-I512-DP-{replica}-{p}-N-r0'for p in('iso'+replica,'joint')]","labels=[f'perf-I512-DP-{replica}-{p}-N-r0'for p in('jointM8','jointM16')]")
 s=s.replace("for dp_phase in('isoA','isoB','joint'):","for dp_phase in('jointM8','jointM16'):").replace("generate(label,i512_inputs,8,True,128", "generate(label,i512_inputs,16 if dp_phase=='jointM16'else 8,True,128")
 s=s.replace("report['performance_adoption']=False;report['ordinary_profileOFF']=True","report['performance_adoption']=False;report['ordinary_profileOFF']=True;protocol_success=False")
 needle="            feedback.request('dp_done','DP-control',phase=dp_phase,healthy=True)\n    except CandidateRejected";assert needle in s;s=s.replace(needle,"            feedback.request('dp_done','DP-control',phase=dp_phase,healthy=True)\n        protocol_success=True\n    except CandidateRejected")
 s=s.replace("'0'if rejected else'1'","'1'if protocol_success else'0'").replace("'SKIPPED_REJECTED_CANDIDATE'if rejected else'EXECUTED_REQUIRED'","'EXECUTED_REQUIRED'if protocol_success else('SKIPPED_REJECTED_CANDIDATE'if rejected else'SKIPPED_PROTOCOL_INCOMPLETE')").replace("if not rejected:assert chat_gate('after_select')","if protocol_success:assert chat_gate('after_select')").replace("report['resident_candidate_quality_passed']=not rejected;report['cohort_phases_completed']=not rejected","report['resident_candidate_quality_passed']=protocol_success;report['cohort_phases_completed']=protocol_success")
 s=s.replace("report['planned_phase_accounting_completed']=len(report['multi_batch_cases'])+len(report['skipped_rejected_candidate_phases'])==len(labels)","report['planned_phase_accounting_completed']=(protocol_success or rejected) and len(report['multi_batch_cases'])+len(report['skipped_rejected_candidate_phases'])==len(labels)")
 s=s.replace("tmp.write_text(json.dumps(report, indent=2)); tmp.replace(report_path)","with tmp.open('wb')as f:f.write((json.dumps(report,ensure_ascii=False,indent=2)+'\\n').encode());f.flush();os.fsync(f.fileno())\n        tmp.replace(report_path)")
 s=s.replace('        resident.finish(label,resident_before,events,prompts)',"        report['last_completed_generation']=row;save()\n        resident.finish(label,resident_before,events,prompts)")
 s=s.replace('ownisolated thenjoint','jointM8 thenjointM16').replace('B16M8 mapped disjoint32globalrequests','B16M8/M16 mapped disjoint32globalrequests')
 return s
edit('glm_prefill_micro_probe_r8868.py',probe)
edit('glm_DP_common_clock_r8868.py',lambda s:s.replace("PHASES=('isoA','isoB','joint')","PHASES=('jointM8','jointM16')").replace("return ('A','B')if p=='joint'else(p[-1],)","return ('A','B')").replace("if phase=='joint':self.finished=True","if phase==PHASES[-1]:self.finished=True").replace('len(rows)<=8','len(rows)<=16'))
def metrics(s):
 s=s.replace("dict.fromkeys(('isoA','isoB','joint'),32)","dict.fromkeys(('jointM8','jointM16'),32)").replace("for phase in ('isoA','isoB','joint'):","for phase in ('jointM8','jointM16'):").replace("replicas=('A','B')if phase=='joint'else(phase[-1],)","replicas=('A','B')").replace("case['micro']==8","case['micro']==(16 if phase=='jointM16'else 8)")
 start=s.index(" result['joint']['actual_joint_overlap_proved']")
 return s[:start]+" for phase in('jointM8','jointM16'):result[phase]['actual_joint_overlap_proved']=result[phase]['actual_decode_overlap_s']>0\n result['comparison']={'total_requests_fixed':32,'M16_over_M8_decode_ratio':result['jointM16']['decode_useful_TPS']/result['jointM8']['decode_useful_TPS'],'scope':'Matched actualjoint/commonclock;same32distinct;twoB16replicas,notB32x2'}\n return result\n"
edit('glm_DP_clock_metrics_r8868.py',metrics)
def audit(s):
 s=s.replace("row['micro']==8","row['micro']in(8,16)").replace('flat in(4,8)','flat in(4,8,16)').replace("e['flat_tokens']in(4,8)","e['flat_tokens']in(4,8,16)").replace("e['round']>0 and len(e['rows'])==8","e['round']>0 and len(e['rows'])in(8,16)").replace("e['round']>0 and len(e['rows'])<8","e['round']>0 and len(e['rows'])not in(8,16)")
 s=s.replace("metrics['joint']['actual_joint_overlap_proved']","all(metrics[p]['actual_joint_overlap_proved']for p in('jointM8','jointM16'))")
 return s
edit('audit_GLM_prefill_Model_r8868.py',audit)
m=json.loads(json.dumps(parent).replace('r8864','r8868'))
for k in('DSO_SHA256','model_SHA256','cpu_receipt_path','budget_cpu_receipt_path','budget_parent_DSO_SHA256','CPU_native_receipt','CPU_raw_receipt_SHA256','compiled_units','CPU_contracts'):m[k]=nm[k]
m.update(DP_phases=['jointM8','jointM16'],performance_micro=[8,16],decode_micro_candidates=[8,16],scope='CPU_READY_ONLY:twoindependentTP8PP2/B16replicas;fixedtotal32distinct477I512O128;oneweightsloadperreplica,jointM8 thenjointM16 full128;original567native68bdbd/vector/prefix/exact160banks/PVtilequantreuseOFF;6208MiB/.5GiBguards;32normal/audit/finallypost;910371n03/n07/n15/n16/32DCU/60min3500 proposed;noGPU GO.',math=nm['math'],root_GO='PENDING_FINAL_PACKET_ROOT_GPU_GO',diagnostic_budget='60minhard3500inner proposed;504actual2032s included3M8phasewalls431s;now2jointwalls~143+M16B16UNKNOWN;startup+pairload~1500s observed;QA/cleanupbudget retained;storagevarianceUNKNOWN;noextraarms/retry/extension.',scope_hit='A477global0..15/B16..31;bothphases same32distinct512inputs/full128EOSQA;M16B16 source supported,actualwholeModelquality/perfstillrequiresqualification.',mechanical_revision={'parent':'Qualified504 commoncollector/independentcomms/namespace,qualified567native+actualfinishrepair','current':'Onlytwojointphases8/16;noisolatedarms;noB32x2/no64requests;boundedcollectoraccept16/nativeCPU210same;0nativebuild'},perf_labels={a:[f'perf-I512-DP-{a}-{p}-N-r0'for p in('jointM8','jointM16')]for a in('A','B')})
m.pop('decode_micro_fixed',None)
side=d/'GLM-transient-Model-observer-manifest-r8868.json';s=json.loads(side.read_bytes());s['source_SHA256']={n.replace('r8864','r8868'):sha(d/n.replace('r8864','r8868'))for n in json.loads((old/'GLM-transient-Model-observer-manifest-r8864.json').read_bytes())['source_SHA256']};write(side,s);m['observer_manifest_SHA256']=sha(side);m['source_SHA256']={n.replace('r8864','r8868'):sha(d/n.replace('r8864','r8868'))for n in parent['source_SHA256']};mp=d/'GLM-CROSSREQUEST-Model-source-manifest-r8868.json';write(mp,m);ep=d/'dispatch_GLM_resident_exact_Model_r8868.py';ep.write_bytes(ep.read_bytes().replace(oldmh.encode(),sha(mp).encode()))
for n in m['source_SHA256']:
 if n.endswith('.py'):ast.parse((d/n).read_bytes(),filename=n)
print(json.dumps({'CPU_only':True,'GPU':False,'nativebuild':False,'manifest_SHA256':sha(mp),'entry_SHA256':sha(ep),'DSO_SHA256':m['model_SHA256'],'phases':m['DP_phases'],'total_requests':32}))
