"""Real PEFT/AdamW CPU checks for safe original trainable initialization."""
import copy,hashlib
from pathlib import Path
import sys,tempfile,unittest
import torch
from peft import LoraConfig,get_peft_model
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from vlm.runtime.initial_trainables import named_parameter_sha256,restore_initial_trainables,validate_initialization_callback_order


def model(seed):
    torch.manual_seed(seed)
    return get_peft_model(torch.nn.Sequential(torch.nn.Linear(7,9),torch.nn.Tanh(),torch.nn.Linear(9,4)),
            LoraConfig(r=2,lora_alpha=4,lora_dropout=.05,target_modules=['0','2']))

def named(m):return sorted((n,p) for n,p in m.named_parameters() if p.requires_grad)

def source(m):
    return dict(parameters={n:p.detach().clone() for n,p in named(m)},
                manifest=[dict(name=n,shape=list(p.shape),dtype=str(p.dtype),numel=p.numel()) for n,p in named(m)],
                global_step=0,RNG={'torch_CPU':torch.get_rng_state().clone()})


class InitialTrainablesTest(unittest.TestCase):
    def test_initialization_precedes_the_actual_first_update_recorder(self):
        class FirstUpdates:pass
        current=object();recorder=FirstUpdates()
        validate_initialization_callback_order([object(),current,recorder],current)
        for callbacks in ([recorder,current],[current],[recorder],[current,recorder,FirstUpdates()]):
            with self.subTest(callbacks=callbacks),self.assertRaises(ValueError):
                validate_initialization_callback_order(callbacks,current)

    def fixture(self,root,mutation=None):
        reference=model(41);target=model(876);value=source(reference)
        if mutation:mutation(value)
        path=root/'initial.pt';torch.save(value,path)
        options=dict(global_step=0,path=path,file_bytes=path.stat().st_size,
                     file_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     named_SHA256=named_parameter_sha256(named(reference)))
        optimizer=torch.optim.AdamW([p for _,p in named(target)],lr=.001)
        return reference,target,optimizer,options

    def test_real_peft_exact_trainables_frozen_base_rng_and_fresh_optimizer(self):
        with tempfile.TemporaryDirectory() as d:
            reference,target,opt,kw=self.fixture(Path(d))
            frozen={n:p.detach().clone() for n,p in target.named_parameters() if not p.requires_grad}
            # Source file stores a different RNG; the loader must not apply it.
            torch.manual_seed(999);rng=torch.get_rng_state().clone()
            proof=restore_initial_trainables(named(target),opt,**kw)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertFalse(proof['RNG_restored'])
            self.assertEqual(len(opt.state),0)
            for n,p in named(target):self.assertTrue(torch.equal(p,dict(named(reference))[n]))
            for n,p in target.named_parameters():
                if n in frozen:self.assertTrue(torch.equal(p,frozen[n]))
            # A genuine subsequent backward/update remains wired to these same parameters.
            target(torch.arange(21).float().reshape(3,7)/20).square().mean().backward();opt.step()
            self.assertTrue(opt.state)

    def test_all_bad_artifact_fields_rejected_before_any_parameter_write(self):
        def rename(v):v['parameters']['foreign']=v['parameters'].pop(next(iter(v['parameters'])))
        def shape(v):
            n=next(iter(v['parameters']));v['parameters'][n]=v['parameters'][n].flatten()
        def dtype(v):
            n=next(iter(v['parameters']));v['parameters'][n]=v['parameters'][n].double()
        def nonfinite(v):v['parameters'][next(iter(v['parameters']))].flatten()[0]=float('nan')
        for change in (rename,shape,dtype,nonfinite,lambda v:v['manifest'].reverse(),
                       lambda v:v.update(global_step=1),lambda v:v.update(global_step=False),
                       lambda v:v.update(extra='foreign')):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                _,target,opt,kw=self.fixture(Path(d),change)
                before={n:p.detach().clone() for n,p in target.named_parameters()}
                rng=torch.get_rng_state().clone()
                with self.assertRaises(ValueError):restore_initial_trainables(named(target),opt,**kw)
                for n,p in target.named_parameters():self.assertTrue(torch.equal(p,before[n]))
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_bad_digest_length_step_gradient_and_optimizer_ownership_rejected(self):
        for bad in ('file_sha','length','named_sha','step','gradient','foreign_optimizer','used_optimizer','duplicate'):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as d:
                _,target,opt,kw=self.fixture(Path(d));selected=named(target)
                if bad=='file_sha':kw['file_SHA256']='0'*64
                if bad=='length':kw['file_bytes']+=1
                if bad=='named_sha':kw['named_SHA256']='0'*64
                if bad=='step':kw['global_step']=1
                if bad=='gradient':selected[0][1].grad=torch.zeros_like(selected[0][1])
                if bad=='foreign_optimizer':opt=torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))])
                if bad=='used_optimizer':
                    target(torch.ones(2,7)).sum().backward();opt.step();opt.zero_grad(set_to_none=True)
                if bad=='duplicate':selected.append(selected[0])
                before={n:p.detach().clone() for n,p in target.named_parameters()}
                with self.assertRaises(ValueError):restore_initial_trainables(selected,opt,**kw)
                for n,p in target.named_parameters():self.assertTrue(torch.equal(p,before[n]))


if __name__=='__main__':unittest.main()
