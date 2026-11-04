"""Small hardware/network contracts; no import-time GPU actions."""
import os,re
def validate_properties(p):
 arch=str(getattr(p,'gcnArchName',''))
 assert arch.split(':')[0]=='gfx936',('GPU architecture',arch)
 assert int(p.multi_processor_count)==80,('GPU CU',p.multi_processor_count)
 assert int(p.total_memory)>=63*1024**3,('GPU memory',p.total_memory)
 return {'name':str(p.name),'gcnArchName':arch,'compute_units':int(p.multi_processor_count),'total_memory':int(p.total_memory)}
def check_hardware(torch,rank):
 p=validate_properties(torch.cuda.get_device_properties(rank))
 assert torch.cuda.current_device()==rank
 x=torch.arange(16,device='cuda',dtype=torch.float32)
 assert x.sum().item()==120.0
 torch.cuda.synchronize(rank)
 assert os.environ['NCCL_NET']=='IBext_v8'and os.environ['NCCL_SOCKET_IFNAME'].startswith('ib')
 p.update(tiny_F32_sum=120.0,before_Model_create_and_weights=True,NCCL_NET=os.environ['NCCL_NET'],NCCL_SOCKET_IFNAME=os.environ['NCCL_SOCKET_IFNAME'])
 return p
def network_log_gate(text):
 ib=bool(re.search(r'Using network IBext_v8\b',text))
 socket=bool(re.search(r'Using network Socket\b',text))
 return {'actual_IBext_v8':ib,'actual_Socket_data_fallback':socket,'passed':ib and not socket}
