"""Read-only CPU metadata checker. Output only paths/SHA/counts/bools/scalars.
Never return source text, build argv, service env or credentials."""
import argparse,pathlib,json,hashlib
D=pathlib.Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--root',default='/refactor');a.add_argument('--output',required=True);x=a.parse_args();r=pathlib.Path(x.root);wanted=json.loads((D/'original-source-pins-r179.json').read_bytes())['files'];roots=[r/'incoming/Kimi-speculative-batch-Model-r9122c',r/'candidates/kimi-compact-cache-r4974',r/'candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc',r/'candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/csrc'];names={pathlib.Path(n).name for n in wanted};found={name:[]for name in names}
 for base in roots:
  for p in base.rglob('*'):
   if p.is_file()and p.name in names:
    b=p.read_bytes();found[p.name].append({'path':str(p),'SHA256':sha(b),'LF_SHA256':sha(b.replace(b'\r\n',b'\n')),'bytes':len(b)})
 rows=[]
 for n,pin in wanted.items():
  matches=[q for q in found[pathlib.Path(n).name]if q['LF_SHA256']==pin['LF_SHA256']];rows.append({'relative':n,'expected_LF_SHA256':pin['LF_SHA256'],'match_count':len(matches),'matches':matches,'other_metadata':[]if matches else found[pathlib.Path(n).name]})
 contexts=[]
 for b in (r/'incoming').glob('Kimi-ctx2048*'):
  for p in b.glob('per-TU-original-context*.json'):
   q=json.loads(p.read_bytes());contexts.append({'path':str(p),'SHA256':sha(p.read_bytes()),'TU_labels':sorted(q),'contains_expected6':all(k in q for k in ['pair_probe','worker','delta','trace','infer_engine','allocate']),'original_context_only_no_argv_exported':True})
 out={'CPU_only':True,'GPU':False,'source_text_exported':False,'rows':rows,'all_required_source_matches':all(q['match_count']for q in rows),'original_compile_context_metadata':contexts,'Native_reference_SHA256':'3057262312679f86bcdb181ce198b9b1f1d2fccc59fc204b4adbc197132b60dd'}
 with pathlib.Path(x.output).open('x')as f:json.dump(out,f,indent=2)
 print(json.dumps(out))
if __name__=='__main__':main()
