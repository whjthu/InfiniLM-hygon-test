import hashlib,json,os,sys
from pathlib import Path
from swift.callbacks import TrainerCallback, callbacks_map
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from vlm.runtime.lora_dropout_session import LoRADropoutSession, input_identity
from vlm.runtime.minicpm_recipe_contract import validate_native_recipe
from vlm.runtime.initial_trainables import restore_initial_trainables,validate_initialization_callback_order
KEY='native_lora_dropout_alignment'


class NativeLoRADropoutAlignment(TrainerCallback):
    """Diagnostic-only MiniCPM callback; native recipe and timing status stay explicit."""
    def __init__(self, swift_args, trainer):
        super().__init__(swift_args, trainer)
        self.trainer=trainer; self.session=None; self.handles=[]; self.ordinal=0; self.receipt=None

    def on_train_begin(self,args,state,control,**kwargs):
        if state.global_step != 0:raise ValueError('mask diagnostic starts at original update zero')
        path=Path(os.environ['NATIVE_LORA_DROPOUT_CONFIG']);raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=os.environ['NATIVE_LORA_DROPOUT_CONFIG_SHA256']:
            raise ValueError('mask diagnostic configuration identity differs')
        config=json.loads(raw)
        if (config['case']!='T05.sft' or config['recipe_SHA256']!='015401ce4791ec978f645afd8eea7724a57f075b6d06a6eac474123418a78789'
                or config['initial_trainable_SHA256']!='9ac0f645b35f75f38aa29770c26f9298e765a4ee4190c86c9481a34bea6bac16'
                or config['world_size']!=8 or config['forwards']!=8 or config['accumulation']!=4
                or config['expected_dropout_modules']!=186 or not 0<config['mask_byte_budget']<=1024**3):
            raise ValueError('expected frozen original MiniCPM first-two-update diagnostic')
        rank=int(os.environ['RANK'])
        if int(os.environ['WORLD_SIZE'])!=8 or not 0<=rank<8:raise ValueError('expected eight real ranks')
        model=kwargs.get('model',self.trainer.model)
        named,_=validate_native_recipe(model,args,10822656)
        initial_receipt=None
        if 'initial_state' in config:
            validate_initialization_callback_order(self.trainer.callback_handler.callbacks,self)
            initial=config['initial_state']
            if (config['mode']!='replay' or initial['bytes']!=43466515
                    or initial['SHA256']!='d3824dc5a788a0238861ae86dec0374dac7b5bf7e6e945e163c32877d6b86d83'):
                raise ValueError('expected original MiniCPM initialization for replay only')
            initial_receipt=restore_initial_trainables(named,self.trainer.optimizer,global_step=state.global_step,
                path=initial['path'],file_bytes=initial['bytes'],file_SHA256=initial['SHA256'],
                named_SHA256=config['initial_trainable_SHA256'])
        h=hashlib.sha256();features={}
        for name,p in named:
            p=p.detach().cpu().contiguous()
            h.update(name.encode());h.update(str(tuple(p.shape)).encode());h.update(p.view(__import__('torch').uint8).numpy().tobytes())
            if '.lora_A.default.weight' in name:
                features[name.replace('.lora_A.default.weight','.lora_dropout.default')]=p.shape[-1]
        if h.hexdigest()!=config['initial_trainable_SHA256'] or len(features)!=186:
            raise ValueError('actual initial parameters or native adapter module inventory differs')
        binding={k:config[k] for k in ('case','recipe_SHA256','initial_trainable_SHA256','expected_dropout_modules')}
        output=Path(config['output'])/f'rank-{rank:02d}'
        mode=config['mode'];reference=None;pin=None
        if mode=='replay':
            reference=Path(config['reference'])/f'rank-{rank:02d}'
            pin=config['reference_manifest_SHA256_by_rank'][str(rank)]
        self.session=LoRADropoutSession(model,output,mode=mode,binding=binding,rank=rank,world_size=8,
                    reference=reference,reference_manifest_SHA256=pin,expected_input_features=features,
                    forwards=8,accumulation=4,byte_budget=config['mask_byte_budget'])
        if initial_receipt is not None:
            with (output/'initial-parameters.json').open('x') as stream:
                json.dump(initial_receipt,stream,indent=2)
        def before(module,argv,kw):
            self.ordinal+=1
            self.session.begin_forward(global_step=self.trainer.state.global_step,micro_ordinal=self.ordinal,
                                       input_SHA256=input_identity(argv,kw))
        def after(module,argv,kw,result):
            if result is None:
                self.session.abort(RuntimeError('root model forward failed'));self._remove_hooks();return
            try:
                active=self.session.active
                if len(active['observed'])!=186 or {x['module'] for x in active['observed']}!=set(features):
                    raise ValueError('expected all 186 actual LoRA dropout calls exactly once per micro forward')
                self.session.end_forward()
                if self.ordinal==8:
                    self.receipt=self.session.close();self._remove_hooks()
            except BaseException as error:
                if not self.session.closed:self.session.abort(error)
                self._remove_hooks();raise
        self.handles=[model.register_forward_pre_hook(before,with_kwargs=True),
                      model.register_forward_hook(after,with_kwargs=True,always_call=True)]
        return control

    def _remove_hooks(self):
        for handle in self.handles:handle.remove()
        self.handles.clear()

    def on_step_end(self,args,state,control,**kwargs):
        if state.global_step==2 and (self.ordinal!=8 or self.receipt is None):
            raise ValueError('incomplete mask packet capture at the second real optimizer update')
        return control

    def on_train_end(self,args,state,control,**kwargs):
        self._remove_hooks()
        if state.global_step!=25 or self.ordinal!=8 or self.receipt is None:
            if self.session is not None and not self.session.closed:self.session.abort(ValueError('native training incomplete'))
            raise ValueError('mask diagnostic does not replace the native 25-update run')
        return control

if KEY in callbacks_map:raise ValueError('duplicate native mask diagnostic callback')
callbacks_map[KEY]=NativeLoRADropoutAlignment
