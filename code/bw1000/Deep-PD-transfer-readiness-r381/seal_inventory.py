"""Compact the one CPU inventory; capability advertisement is not a measured transfer."""
from pathlib import Path
import json,hashlib,gzip,base64,re
G=Path(__file__).resolve().parents[3];L=G.parent;source=L/'Deep381-fresh-two-node-CPU-inventory.json';b=source.read_bytes();raw=json.loads(b)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
for r in sorted(raw['node_probe']['records'],key=lambda x:x['node']):
 assert r['KFD_available']and not r['kfd_pids']and not r['GPU_compute_processes']and not r['GPU_initialized']and not r['transfer_started']
 gpu=[p for p in r['GPU_PCI_sysfs']if p['driver']in ['hycu','amdgpu']];assert len(gpu)==8
 addresses=json.loads(r['network']['addresses']['stdout']);ucx=r.get('UCX_devices',{}).get('stdout','')
 rows.append(dict(node=r['node'],CPU_step=r['actual_step'],KFD=[],compute_processes=[],GPU_driver_PCI=gpu,
  ib_IPv4={n['ifname']:[v['local']for v in n.get('addr_info',[])if v['family']=='inet']for n in addresses if n['ifname'].startswith('ib')},
  IB_ports=r['RDMA_sysfs'],candidate_ports=r['candidate_transfer_ports_bindable'],memlock=r['memlock'],tools=r['tools'],
  perftest_ROCm_option_advertised=r['perftest_ROCm_option_advertised'],UCX_ROCm_registration_advertised='rocm (reg,cache)'in ucx,
  UCX_ROCm_memory_domain_present='# Memory domain: rocm'in ucx,UCX_ROCm_lines=[s for s in ucx.splitlines()if'rocm'in s.lower()],
  interfaces_route=json.loads(r['network']['routes']['stdout'])))
result=dict(schema=1,GPU=0,model_loaded=False,transfer_started=False,Git_requested=False,observed_UTC=raw['observed_UTC'],
 CPU_probe=dict(actual='910376.832',nodes=['a02r4n12','a02r4n13'],CPUs_per_node=1,host_mem_MiB_per_node=1024,srun_returncode=raw['node_probe']['returncode'],bounded_seconds=120,read_only=True),
 allocations=raw['allocations'],active_steps_before=raw['active_steps'],nodes=rows,
 recommended_minimum=dict(parent=910376,P=dict(node='a02r4n12',GPU_cards=8,TP=8,PP=1),D=dict(node='a02r4n13',GPU_cards=8,TP=8,PP=1),total_GPU_cards=16,not_claimed_for_GPU=True,fresh_recheck_required=True,unprobed_fallback_allocation=910380),
 transport=dict(known_available_baseline='GPU D2H -> pinnedhost -> TCP over IPoIB ib0 -> pinnedhost H2D -> GPU; no deviceRDMA claim',
 host_RDMA='libibverbs/rdmacm and host perftest available, current pair throughput not measured',
 device_RDMA='UNKNOWN: all4 UCX shca memorydomains advertise rocm(reg,cache) but no GPU pointer registration or current container UCX integration tested; perftest doesnotadvertiseROCm; do not infer impossible or qualified',
 IPoIB_route=dict(P='12.1.2.72',D='12.1.2.73',rail='ib0/shca_0',optional_other_rails='13/14/15.1.2.72->.73 source-selected; no multirail claim yet'),
 old_host_RDMA_result='results/bw1000/2026-10-26/host-rdma3483-actual-multirail-bandwidth.json is othernodes/hostmemory only; cannot inherit currentpair/device bandwidth'),
 interpretation=['Original GPU_PCI_sysfs is intentionally broad vendorinventory including Hygon chipset. SummaryGPU count filters actualdriver hycu/amdgpu only; no GPU contextcreated.',
 'Nominal4x400Gb/s links are link capability, not measured aggregate application bandwidth.',
 'CPUprobe was one actual910376.832. Prior two preprocessing attempts submitted0 nodeCPU/0GPU: unsupported squeue step%T, then parent batch/extern overbroad skip. Both preserved; neither proves resourcesbusy.',
 'Do not subtract clocks acrossnodes. Future transfer timer must sender-local roundtrip/fenceACK or receiver-local completedstate fence; all8P+8D cardseconds included.'],
 original=dict(path=str(source),SHA256=sha(source),bytes=len(b),actual_node_source_SHA256=sha(Path(__file__).with_name('fresh_probe.py')),orchestrator_SHA256=sha(L/'remote381-fresh-allocation-once.py')))
out=G/'results/bw1000/2026-11-02';out.mkdir(parents=True,exist_ok=True)
seal=out/'Deep381-CPU832-original-inventory-sealed.json';seal.write_text(json.dumps(dict(schema=1,content='CPU sysfs/process/IBtool scalarinventory, no model/key/activation',uncompressed_SHA256=sha(source),uncompressed_bytes=len(b),encoding='base64(gzip)',payload=base64.b64encode(gzip.compress(b,mtime=0)).decode()),indent=2)+'\n',encoding='utf8')
result['sealed']=dict(path=str(seal.relative_to(G)),SHA256=sha(seal))
p=out/'Deep381-CPU832-resource-and-interconnect-readiness.json';p.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(completed=True,GPU=0,result=str(p),SHA256=sha(p),sealed_bytes=seal.stat().st_size)))
