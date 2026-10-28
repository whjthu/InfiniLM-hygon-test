"""CPU: prepare the frozen actual regular-agent datasets with ordinary BPE.
No synthetic assistant turns are generated; real workflows later create them.
"""
import argparse,pathlib,sys,hashlib,importlib.util,json
D=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(D/'runtime'))
from long_codec_r179 import KimiLongCodec
CLIENT_SHA='51218ed03190ea87b63cb06cf6c5883aae5d327a7db878109fd34c5ce4bde194'
def main():
 a=argparse.ArgumentParser();a.add_argument('--model',default='/shared/Kimi-K3');a.add_argument('--output-dir',required=True);x=a.parse_args();codec=KimiLongCodec(x.model);p=D/'client/client_regular_agent.py';assert hashlib.sha256(p.read_bytes()).hexdigest()==CLIENT_SHA
 spec=importlib.util.spec_from_file_location('frozen_regular_agent',p);client=importlib.util.module_from_spec(spec);spec.loader.exec_module(client)
 output=pathlib.Path(x.output_dir);profiles=client.prepare_profiles(lambda text:len(codec.encoding.encode(text,disallowed_special=())),output)
 assert all(q['output_budget']==(8192 if q['thinking']else 4096)for q in profiles)
 result={'CPU_only':True,'GPU':False,'weights_read':False,'model_outputs_generated':False,'official_ordinary_BPE_count':True,'client_SHA256':CLIENT_SHA,'tokenizer':codec.identity,'profiles':profiles,'max_context':32768,'ordinary_default_output':4096,'thinking_output':8192,'thinking_default_effort':'high','actual_model_eight_step_history':'NOT_RUN','GPU_GO':False}
 (output/'CPU-profile-receipt-r179.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
 assert 'torch'not in sys.modules and 'transformers'not in sys.modules
if __name__=='__main__':main()
