"""Independent Torch F32/BF16 activation and NumPy exact QuantI8-0 reference.

CPU libmath can differ at one BF16 rounding boundary. That is recorded, never
inherited as a device permission: same-GU original GPU SDK must pass first.
"""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
sha=lambda b:hashlib.sha256(b).hexdigest()

def quant_reference(activation):
 a=np.asarray(activation,dtype=np.float32);assert np.isfinite(a).all()
 maximum=np.max(np.abs(a),axis=1,keepdims=True)
 scale=np.maximum(maximum/np.float32(127),np.float32(1e-8)).astype(np.float32)
 v=a*(np.float32(1)/scale);m=np.floor(np.abs(v)+np.float32(.5))
 return np.clip(np.where(v>=0,m,-m),-127,127).astype(np.int8),scale

def run(folder,executable):
 import torch
 folder=Path(folder);folder.mkdir(exist_ok=False);report=folder/'summary.json';q=dict(completed=False,passed=False,cases=[],GPU=False,Torch_CPU_reference=True,CXX_actual=True,nonfinite_quantization='NOT_QUALIFIED: original SDK casts nonfinite float to int with no defined portable numeric result')
 def save():report.write_text(json.dumps(q,indent=2)+'\n')
 rng=np.random.default_rng(438);cases=[]
 for rows in [1,6]:cases.append(('ordinary_rows'+str(rows),rng.normal(0,2,(rows,512)).astype(np.float32),False))
 x=np.zeros((1,512),np.float32);cases.append(('zeros',x.copy(),False))
 x[0,:256]=-0.;x[0,256:]=1.;cases.append(('signed_zero',x.copy(),False))
 values=np.array([-100.,-20.,-10.0625,-10.,-9.9375,-.5,0.,.5,9.9375,10.,10.0625,100.],np.float32)
 x=np.resize(values,(6,512));cases.append(('clamp_edges',x.copy(),False))
 x=np.full((1,512),10.,np.float32);x[0,256:]=0.;x[0,256]=6.34375;x[0,257]=.0250244140625;x[0,258]=-.0250244140625;cases.append(('half_away_ties',x.copy(),False))
 x=np.zeros((1,512),np.float32);x[0,:256]=1.;x[0,256:]=np.float32(1e-10);cases.append(('scale_epsilon',x.copy(),False))
 x=np.zeros((1,512),np.float32);x[0,:256]=np.linspace(-110,-80,256,dtype=np.float32);x[0,256:]=1.;cases.append(('tiny_silu',x.copy(),False))
 # Largest *BF16* finite value: F32 max would round to BF16 infinity.
 largest=np.array([0x7f7f0000],dtype=np.uint32).view(np.float32)[0]
 x=np.full((6,512),largest,np.float32);x[:,::2]*=-1;cases.append(('large_finite_BF16',x.copy(),False))
 x=rng.normal(size=(1,512)).astype(np.float32);x[0,[0,1,2,256,257]]=[np.nan,np.inf,-np.inf,np.inf,np.nan];cases.append(('nonfinite_activation_classes_only',x.copy(),True))
 save()
 try:
  for name,source,classes_only in cases:
   q['active_case']=name;save();gu=torch.from_numpy(source.copy()).to(torch.bfloat16);g=gu[:,:256].float().clamp(max=10).to(torch.bfloat16).float();u=gu[:,256:].float().clamp(min=-10,max=10).to(torch.bfloat16).float()
   sigmoid=torch.reciprocal(torch.add(torch.exp(-g),1.));wanted=((g*sigmoid)*u).to(torch.bfloat16)
   raw=gu.view(torch.uint16).numpy().tobytes();inp=folder/(name+'.input.raw');out=folder/(name+'.output.raw');inp.write_bytes(raw)
   z=subprocess.run([str(executable),'--activation'if classes_only else'--row',str(inp),str(out)],capture_output=True,text=True,timeout=20);assert z.returncode==0,(name,z.stderr)
   data=out.read_bytes();rows=source.shape[0];width=512 if classes_only else 772;assert len(data)==rows*width
   actual_bits=np.stack([np.frombuffer(data[r*width:r*width+512],dtype=np.uint16)for r in range(rows)])
   actual=torch.from_numpy(actual_bits.copy()).view(torch.bfloat16).float().numpy();want_bits=wanted.view(torch.uint16).numpy();expected=wanted.float().numpy()
   assert np.array_equal(np.isnan(actual),np.isnan(expected))and np.array_equal(np.isposinf(actual),np.isposinf(expected))and np.array_equal(np.isneginf(actual),np.isneginf(expected))
   finite=np.isfinite(expected);nonzero=finite&(expected!=0);zeros=finite&(expected==0);assert np.array_equal(actual_bits[zeros],want_bits[zeros])
   ulps=np.abs(actual_bits[nonzero].astype(np.int32)-want_bits[nonzero].astype(np.int32));max_ulp=int(ulps.max(initial=0));assert max_ulp<=1,(name,max_ulp)
   row=dict(name=name,passed=False,input_SHA256=sha(raw),output_SHA256=sha(data),rows=rows,classes_only=classes_only,activation_exact_bits=bool(np.array_equal(actual_bits[finite],want_bits[finite])),CPU_activation_max_BF16_ULP=max_ulp)
   if not classes_only:
    packed=np.stack([np.frombuffer(data[r*772+512:r*772+768],dtype=np.int8)for r in range(rows)]);scales=np.stack([np.frombuffer(data[r*772+768:r*772+772],dtype=np.float32)for r in range(rows)])
    p,s=quant_reference(actual);assert np.array_equal(packed,p)and np.array_equal(scales.view(np.uint32),s.view(np.uint32)),(name,'independent quant on CXX BF16 activation')
    if name=='half_away_ties':assert packed[0,1]==1 and packed[0,2]==-1,'Half-away differs from banker rounding'
    row.update(independent_quant_I8_exact=True,independent_quant_F32_scale_bits_exact=True)
   row['passed']=True;q['cases'].append(row);save()
  q.update(passed=True,CPU_activation_gate='Classification/signedzero strict; at most one BF16 ULP across CPU libmath. Actual GPU same-GU baseline/candidate not inherited.',device_qualified=False,torch_cuda_initialized=bool(torch.cuda.is_initialized()));assert not q['torch_cuda_initialized']
 except BaseException as error:q.update(passed=False,error=repr(error));raise
 finally:q['completed']=True;save()
 return q
