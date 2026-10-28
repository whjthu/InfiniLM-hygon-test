"""Genuine official tokenizer sizing, no model/Native/GPU acceptance."""
import os,pathlib,json,hashlib
assert os.environ['USE_TORCH']=='0'and os.environ['USE_TF']=='0'
assert all(os.environ[n]=='-1'for n in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
D=pathlib.Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((D/'source-manifest-r178a.json').read_bytes())
for n,h in m['files'].items():assert sha(D/n)==h,n
from CPU_wire_contract_r178a import validate
wire=validate()
from transformers import PreTrainedTokenizerFast
from client_regular_agent import prepare_profiles
M=pathlib.Path('/old/models/ZhipuAI-GLM-5.3');tok=PreTrainedTokenizerFast.from_pretrained(M,local_files_only=True)
template=hashlib.sha256(tok.get_chat_template().encode()).hexdigest();assert template=='3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c'
step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=pathlib.Path('/refactor/results')/('GLM-regular-fixture-r178a-'+step)
profiles=prepare_profiles(lambda text:len(tok.encode(text,add_special_tokens=False)),out)
assert [p['output_budget']for p in profiles]==[4096,4096,8192]
for p in profiles:
 assert p['CPU_raw_text_tokens']>=p['minimum_prompt_tokens']
 assert p['CPU_raw_text_tokens']+p['output_budget']<32768
 p['reasoning_policy']='high'if p['thinking']else'native_default_omitted'
 # Raw tool-result sizing only; actual prompt overhead and accumulated generated
 # reasoning/history must be measured and admitted in each live request.
q={'completed':True,'CPU_only':True,'GPU_initialized':False,'Native_loaded':False,'weights_read':False,'official_template_SHA256':template,'config_SHA256':sha(M/'config.json'),'regular_client_SHA256':sha(D/'client_regular_agent.py'),'WireClient_SHA256':sha(D/'client_GLM_regular_wire_r178a.py'),'source_manifest_SHA256':sha(D/'source-manifest-r178a.json'),'wire_mock_contract':wire,'profiles':profiles,'profiles_json':str(out/'profiles.json'),'profiles_SHA256':sha(out/'profiles.json'),'actual_prompt_tokens_NOT_YET_MEASURED':True,'actual_eight_tools_NOT_RUN':True,'normal_agent_PASS':False,'scope':'Real complete input files sized by official tokenizer; actual live native-default/high reasoning,8 generated/executed tool steps,full history usage/cap/quality remain unqualified'}
dest=out/'CPU-sizing.json';dest.write_text(json.dumps(q,indent=2));print(json.dumps({'completed':True,'CPU_only':True,'path':str(dest),'SHA256':sha(dest),'profiles':profiles,'normal_agent_PASS':False}))
