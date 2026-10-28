"""Optional exact1206 thinking/request009 tokenizer-only counterfactual."""
import copy,hashlib,importlib.util,json,os,pathlib,sys
from tokenizer_guard_r205 import install_backend_guard,no_backend,deny_weights,BACKEND_METADATA_PROBES,BLOCKED_BACKEND_ATTEMPTS
D=pathlib.Path(__file__).resolve().parent;R=pathlib.Path('/refactor');OLD=R/'incoming/Deep-agent-regular32K-r196';ROOT=R/'results/Deep-agent-regular32K-r196-910374.1206/regular';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main():
 no_backend();assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']);install_backend_guard();sys.addaudithook(deny_weights)
 pins=json.loads((D/'review-manifest-r205.json').read_bytes())
 for p,h in pins['source_SHA256'].items():assert sha(D/p)==h
 assert sha(OLD/'manifest-r196.json')=='6c449a639bff0debd57cf61c80dcf4edb3b1657e3cae45f2195f22038cbac32a';oldmf=json.loads((OLD/'manifest-r196.json').read_bytes())
 for p in ['chat_prepare_r173.py','encoding/encoding_dsv4.py','overlay/python/infinilm/server/agent_protocol.py']:assert sha(OLD/p)==oldmf['source_SHA256'][p]
 assert sha(ROOT/'result.json')=='e45158f2f75a25ef86fee207668a22ec159fc69c5411bd21b5eeb13f5af76fff';result=json.loads((ROOT/'result.json').read_bytes());thinking=next(c for c in result['cases']if c['name']=='thinking');assert result['passed']and thinking['passed']and result['actual_HTTP_requests']==30
 p=ROOT/'thinking/request-009.request.json';raw=p.read_bytes();body=json.loads(raw);assert body['tool_choice']=='none'and body['max_tokens']==8192 and body['chat_template_kwargs']['enable_thinking']is True
 protocol=load('cpu205_protocol',OLD/'overlay/python/infinilm/server/agent_protocol.py');data,schemas,choice=protocol.prepare_agent_request(body,'deepseek_v4');data['_agent_schemas']=schemas;data['_agent_choice']=choice;assert not schemas and data['chat_template_kwargs']['tools']is None
 os.environ.update(USE_TORCH='0',USE_TF='0',USE_FLAX='0')
 from transformers import PreTrainedTokenizerFast
 no_backend();model=pathlib.Path('/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8');assert sha(model/'tokenizer.json')=='8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf';tok=PreTrainedTokenizerFast.from_pretrained(str(model),local_files_only=True);no_backend()
 old=load('cpu205_old_prepare',OLD/'chat_prepare_r173.py');new=load('cpu205_new_prepare',D/'chat_prepare_r173.py');messages=data['messages'];before=copy.deepcopy(messages);kwargs=data['chat_template_kwargs'];thoughts=[m['reasoning_content']for m in messages if m.get('reasoning_content')];assert thoughts
 oldprompt,oldids=old.prepare(messages,data,True,kwargs,tok,32768,8192);newprompt,newids=new.prepare(messages,data,True,kwargs,tok,32768,8192);assert messages==before and len(oldids)==thinking['actual_prompt_lengths'][-1]
 assert all(rc in newprompt for rc in thoughts)and newprompt.endswith(new.encoder.ASSISTANT_SP_TOKEN+'<think>')
 # Same actual saved history, only hypothetical current generation set to chat.
 chat=copy.deepcopy(kwargs);chat['enable_thinking']=False;chat.pop('thinking_mode',None);cp,cids=new.prepare(messages,data,True,chat,tok,32768,8192);assert cp.endswith(new.encoder.ASSISTANT_SP_TOKEN+'</think>')and all(rc in cp for rc in thoughts)
 def fp(prompt,ids):return {'chars':len(prompt),'tokens':len(ids),'prompt_SHA256':hashlib.sha256(prompt.encode()).hexdigest(),'ids_SHA256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),'reasoning_presence':[rc in prompt for rc in thoughts]}
 no_backend();report={'completed':True,'CPU_only':True,'Model_called':False,'new_Model_PASS':False,'scope':'One actual1206 saved final thinking request official-HF tokenizer counterfactual, plus current-chat hypothetical; no response generation','candidate_review_manifest_SHA256':sha(D/'review-manifest-r205.json'),'frozen1206_regular_PASS_unchanged':True,'frozen_regular_result_SHA256':sha(ROOT/'result.json'),'request_SHA256':hashlib.sha256(raw).hexdigest(),'tokenizer_json_SHA256':sha(model/'tokenizer.json'),'input_reasoning_fields':len(thoughts),'old':fp(oldprompt,oldids),'candidate':fp(newprompt,newids),'candidate_current_chat_hypothetical':fp(cp,cids),'current_thinking_prefix_preserved':True,'current_chat_prefix_closed':True,'history_and_reasoning_input_unchanged':messages==before,'output_budget':8192,'context':32768,'strict_schemas_unchanged':not schemas,'Torch_imported':False,'Native_loaded':False,'weights_opened':False,'backend_metadata_probes':BACKEND_METADATA_PROBES,'blocked_backend_attempts':BLOCKED_BACKEND_ATTEMPTS};print(json.dumps(report,separators=(',',':')))
if __name__=='__main__':
 try:main()
 except BaseException as e:print(json.dumps({'completed':False,'CPU_only':True,'error_class':type(e).__name__,'error_SHA256':hashlib.sha256(str(e).encode()).hexdigest(),'Torch_imported':'torch'in sys.modules,'Native_loaded':'_infinilm'in sys.modules,'Model_called':False,'blocked_backend_attempts':BLOCKED_BACKEND_ATTEMPTS}));raise SystemExit(1)
