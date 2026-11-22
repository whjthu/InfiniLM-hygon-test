"""Additive observers after the exact original Gemma native resume callback."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import time

from contract import ROOT,ASSET,ARM,Clock,need,stable_json
from common_rng_tap import CommonRNGTap
from native_bridge import NativeBackend
from light_trace import install_if_enabled

provider=ASSET/'framework/vlm/plugins/gemma_domain_resume.py'
need(hashlib.sha256(provider.read_bytes()).hexdigest()=='188e07812f9b3c9f5916da520f05785e6e7c6c34b95225b75440e78ce777fa01',
     'original Gemma resume callback source changed')
original=importlib.import_module('vlm.plugins.gemma_domain_resume')
need(Path(original.__file__).resolve(strict=True)==provider.resolve(strict=True),'loaded original callback differs')
need(original.callbacks_map.get(original.KEY) is original.GemmaDomainResume,'unexpected original callback registry')


class GemmaPairedDiagnostic(original.GemmaDomainResume):
    def on_train_begin(self,args,state,control,**kwargs):
        result=super().on_train_begin(args,state,control,**kwargs)
        cfg,pin=stable_json(ROOT/'diagnostic-config.json')
        need(pin['SHA256']==os.environ['GEMMA_DIAGNOSTIC_CONFIG_SHA256'] and cfg['arm']==ARM,
             'exact diagnostic scope config differs')
        rank=int(os.environ['RANK']);model=kwargs.get('model',self.trainer.model)
        backend=NativeBackend(cfg['trace'],model,self.trainer,rank,self.witness.check)
        self.common_tap=CommonRNGTap(backend,ROOT/'common-rng',rank).attach(model)
        self.light_trace=install_if_enabled(cfg['trace'],rank=rank,backend_factory=lambda:backend)
        if self.light_trace is not None:self.light_trace.attach(model)
        return result

    def on_train_end(self,args,state,control,**kwargs):
        result=super().on_train_end(args,state,control,**kwargs)
        common=self.common_tap.finish()
        full=self.light_trace.finish() if self.light_trace is not None else None
        with (ROOT/'diagnostic-hooks'/('rank-%02d.json'%int(os.environ['RANK']))).open('x') as f:
            json.dump(dict(common_complete=common['complete'],full_trace_enabled=self.light_trace is not None,
                full_trace_complete=full['complete'] if full else None,arm=ARM),f,sort_keys=True)
        return result


original.callbacks_map[original.KEY]=GemmaPairedDiagnostic
