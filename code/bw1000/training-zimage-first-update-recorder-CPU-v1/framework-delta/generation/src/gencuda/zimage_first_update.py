"""Private native Z-Image first-update recording; no synthetic training inputs.

The original loss AST is executed with observation calls at its actual sampled
timestep/noise/target and returned prediction/loss boundaries. Removing those
calls must reproduce the complete original function AST. No loss expression,
random call, optimizer expression or activation value is replaced.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import textwrap

from .vggt_first_update import FirstUpdateRecorder
from . import hy_resume_checkpoint as common


def instrument_loss(original, observer):
    """Build the same exact source function with two non-mutating observers."""
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError('expected one original FlowMatchSFTLoss function')
    function = tree.body[0]
    if function.name != 'FlowMatchSFTLoss' or function.decorator_list:
        raise ValueError('unexpected native loss identity or decorators')
    original_ast = ast.dump(tree, include_attributes=False)
    inserted = []
    body = []
    for node in function.body:
        body.append(node)
        targets = {t.id for t in node.targets if isinstance(t, ast.Name)} if isinstance(node, ast.Assign) else set()
        if 'training_target' in targets:
            new = ast.parse('_zimage_observer.prepared(inputs, noise, timestep_id, timestep, training_target)').body[0]
            body.append(new);inserted.append(new)
        if isinstance(node, ast.Return):
            body.pop()
            new = ast.parse('_zimage_observer.forward_value(noise_pred, loss)').body[0]
            body.extend([new,node]);inserted.append(new)
    if len(inserted) != 2:
        raise ValueError('original sampled target/returned loss boundaries differ')
    function.body = body
    stripped = ast.parse(ast.unparse(tree))
    stripped.body[0].body = [n for n in stripped.body[0].body if not (
        isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
        and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='_zimage_observer')]
    if ast.dump(stripped,include_attributes=False) != original_ast:
        raise ValueError('observer changed original native loss AST')
    scope = dict(original.__globals__)
    scope['_zimage_observer'] = observer
    ast.fix_missing_locations(tree)
    exec(compile(tree,inspect.getsourcefile(original) or '<native_zimage_loss>','exec'),scope)
    value = scope[function.name]
    value.__module__ = original.__module__
    return value


@contextmanager
def observe_native_loss(native_training_module, recorder):
    """Only the first real training forward is observed; always restore globals."""
    original = native_training_module.FlowMatchSFTLoss
    if getattr(original,'_zimage_first_update_observer',False):
        raise RuntimeError('native loss observer already installed')
    value = instrument_loss(original,recorder)
    value._zimage_first_update_observer = True
    native_training_module.FlowMatchSFTLoss = value
    try:
        yield
    finally:
        native_training_module.FlowMatchSFTLoss = original


def safe_input(value):
    """Keep real tensors and ordinary values; bind a PIL input by its bytes."""
    import torch
    if isinstance(value,torch.Tensor) or value is None or type(value) in (bool,int,float,str,bytes):
        return value
    if isinstance(value,dict):return {k:safe_input(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(safe_input(v) for v in value)
    from PIL import Image
    if isinstance(value,Image.Image):
        return dict(kind='PIL.Image',mode=value.mode,size=list(value.size),decoded_pixels_SHA256=hashlib.sha256(value.tobytes()).hexdigest())
    raise TypeError('unsupported native Z-Image input '+type(value).__name__)


class ZImageFirstUpdateRecorder(FirstUpdateRecorder):
    """Reuse bounded atomic CPU serialization; compact full tensor groups."""

    def progress(self):
        value = dict(case='T07.sft',planned_global_steps=25,planned_recorded_updates=1,
                     planned_tensor_groups=['initial','preclip','postclip','updated','optimizer'],
                     planned_fixed_files=['initial_optimizer.pt','input.pt','prepared.pt','target.pt','forward.pt','clip_norm.pt','optimizer_groups.pt'],
                     trainable_tensors=len(self.named),trainable_elements=sum(p.numel() for _,p in self.named),
                     completed_global_steps=self.completed_steps,current_global_step=self.current_step,
                     first_update_complete=self.first_complete,epoch=self.epoch,deadline=self.deadline,
                     phase=self.phase,failed=self.failed,files=self.files,artifact_bytes=self.total,
                     diagnostic_seconds=self.diagnostic_seconds)
        tmp=self.root/'progress.tmp';tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(self.root/'progress.json')

    def _group(self, stage, gradients=False):
        values={}
        for name,parameter in self.named:
            value=parameter.grad if gradients else parameter
            if value is None:raise ValueError('missing native Z-Image gradient '+name)
            values[name]=value
        self._save(stage+'.pt',values)

    def enter(self, step, batch):
        super().enter(step,safe_input(batch))

    def prepared(self, inputs, noise, timestep_id, timestep, target):
        if self.current_step != 1:return
        if self.phase != 'input':raise ValueError('native noise/target capture order differs')
        actual=safe_input(inputs)
        self.prepared_identity=common.tree_sha256(actual)
        self.prepared_inputs=actual
        self._save('prepared.pt',dict(inputs=actual,noise=noise,timestep_id=timestep_id,timestep=timestep))
        self._save('target.pt',dict(training_target=target))
        self.phase='target';self.progress()

    def forward_value(self, prediction, loss):
        if self.current_step != 1:return
        if self.phase != 'target' or common.tree_sha256(self.batch)!=self.input_SHA256:
            raise ValueError('native first forward order or raw input changed')
        if common.tree_sha256(self.prepared_inputs)!=self.prepared_identity:
            raise ValueError('native model forward mutated its prepared input')
        self._save('forward.pt',dict(noise_prediction=prediction,loss=loss))
        del self.batch,self.prepared_inputs
        self.phase='forward';self.progress()

    def updated(self):
        if self.current_step==1:
            if self.phase!='postclip':raise ValueError('native update capture order differs')
            self._group('updated')
            states={}
            for name,parameter in self.named:
                state=self.optimizer.state.get(parameter)
                if not state or not {'step','exp_avg','exp_avg_sq'}<=set(state) or float(state['step'])!=1:
                    raise ValueError('missing first native AdamW state '+name)
                states[name]=state
            self._save('optimizer.pt',states)
            self._save('optimizer_groups.pt',self.optimizer.state_dict()['param_groups'])
            self.first_complete=True
        self.completed_steps=self.current_step;self.phase='updated';self.progress()

    def finish(self, global_step, frozen_parameters_unchanged):
        self.check()
        if global_step!=25 or self.completed_steps!=25 or not self.first_complete or not frozen_parameters_unchanged:
            raise ValueError('native25 or original frozen boundary did not complete')
        self.phase='NATIVE25_RETURNED';self.progress()


def recorded_native_update(native_update, native_training_module, recorder, step, model, optimizer, data, seed, max_grad_norm):
    """One native update with parameter/gradient callbacks and exact math body."""
    if recorder is None:
        return native_update(model,optimizer,data,seed,max_grad_norm)
    # The caller supplies the instrumented native function, whose stripped AST is
    # checked against the original backend. The ordinary disabled path is direct.
    def run():
        recorder.enter(step,data)
        if step==1:
            with observe_native_loss(native_training_module,recorder):
                return native_update(model,optimizer,data,seed,max_grad_norm,observer=recorder)
        return native_update(model,optimizer,data,seed,max_grad_norm,observer=recorder)
    return recorder.guard(run)


def instrument_update(original):
    """Insert callbacks around original backward/clip/update without math edits."""
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):raise ValueError('one native update required')
    fn=tree.body[0];baseline=ast.dump(tree,include_attributes=False)
    if fn.args.kwonlyargs:raise ValueError('unexpected original update signature')
    fn.args.kwonlyargs.append(ast.arg(arg='observer'));fn.args.kw_defaults.append(ast.Constant(value=None))
    body=[];count=0
    for node in fn.body:
        body.append(node)
        if not isinstance(node,(ast.Expr,ast.Assign)):continue
        call=node.value
        if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Attribute):continue
        attr=call.func.attr
        if attr=='backward':extra="if observer is not None: observer.gradients('preclip')"
        elif attr=='clip_grad_norm_':extra="if observer is not None: observer.gradients('postclip', grad_norm)"
        elif attr=='step' and isinstance(call.func.value,ast.Name) and call.func.value.id=='optimizer':extra='if observer is not None: observer.updated()'
        else:continue
        body.extend(ast.parse(extra).body);count+=1
    if count!=3:raise ValueError('native backward/clip/update boundaries differ')
    fn.body=body
    stripped=ast.parse(ast.unparse(tree));ss=stripped.body[0]
    ss.args.kwonlyargs=[];ss.args.kw_defaults=[]
    ss.body=[n for n in ss.body if not (isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Name) and n.test.left.id=='observer')]
    if ast.dump(stripped,include_attributes=False)!=baseline:raise ValueError('observer changed original update AST')
    scope=dict(original.__globals__);ast.fix_missing_locations(tree);exec(compile(tree,inspect.getsourcefile(original) or '<native_update>','exec'),scope)
    return scope[fn.name]
