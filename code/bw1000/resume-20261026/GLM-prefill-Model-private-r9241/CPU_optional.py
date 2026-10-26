import pathlib,ast,types,json,copy
import torch
assert not torch.cuda.is_initialized()
p=pathlib.Path(__file__).parent;tree=ast.parse((p/'glm_prefill_probe_r9241.py').read_text());outer=next(v for v in tree.body if isinstance(v,ast.FunctionDef)and v.name=='run');branch=next(v for v in ast.walk(outer)if isinstance(v,ast.If)and ast.unparse(v.test)=="cohort.get('optional64')")
loop=ast.For(target=ast.Name(id='cohort',ctx=ast.Store()),iter=ast.List(elts=[ast.Dict(keys=[ast.Constant('optional64')],values=[ast.Constant(True)])],ctx=ast.Load()),body=[branch],orelse=[]);code=compile(ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[])),'<actualoptionalAST>','exec')
class CPU:
 @staticmethod
 def zeros(*a,**k):k.pop('device',None);return torch.zeros(*a,**k)
 @staticmethod
 def tensor(*a,**k):k.pop('device',None);return torch.tensor(*a,**k)
 float32=torch.float32
checks=[]
for faster,quality,allow in [(True,True,True),(True,False,True),(False,True,True),(True,True,False)]:
 primary=[{'C':32,'optional64':False,'prefill_deferred':v,'prefill_completion_wall_seconds':10 if v and faster else 20 if v else 15,'decode_common_window_useful_tok_s':20,'task_scores':[{'flexible_correct':quality if v else True}for _ in range(32)]}for v in [False,True,True,False]]
 env={'report':{'service':primary},'allservice':[{'input_ids':[1]*100}for _ in range(64)],'optional_admitted_r9241':None,'winner_mode_r9241':False,'budget':lambda n:allow,'torch':CPU,'is_emitter':True,'control':lambda x:None,'save':lambda:None};exec(code,env);assert env['optional_admitted_r9241']is allow and env['winner_mode_r9241']is(faster and quality);checks.append({'faster':faster,'quality_no_loss':quality,'budget':allow,'passed':True})
print(json.dumps({'passed':True,'cases':checks,'actual_optional_AST':True,'CPU_Torch_only':True,'noGPUCCLqualification':True}))
