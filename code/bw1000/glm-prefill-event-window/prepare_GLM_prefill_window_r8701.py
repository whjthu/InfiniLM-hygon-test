import hashlib,pathlib
r=pathlib.Path(__file__).resolve().parent
p=r/'glm_model_window_r5797.cpp';s=p.read_text();assert hashlib.sha256(p.read_bytes()).hexdigest()=='0ef9739d29f2797879f48ca12c81f79e02e74aa4f47407bbd4018177de0063bc'
s=s.replace('bool armed=false,active=false,pending=false,aborted=false;', 'bool armed=false,active=false,pending=false,aborted=false;std::array<size_t,8>rows{},pasts{},counts{};unsigned matched=0;')
s=s.replace('return s.armed&&i<8&&row==size_t(s.cohort*8)+i&&past==size_t(512+s.round-1)&&count==1;', 'bool ok=s.armed&&i<8&&row==size_t(s.cohort*8)+i&&past==(s.round==0?size_t(0):size_t(512+s.round-1))&&count==(s.round==0?size_t(512):size_t(1));if(ok){s.rows[i]=row;s.pasts[i]=past;s.counts[i]=count;s.matched|=1u<<i;}return ok;')
s=s.replace('if(round<63||round>65||cohort<0||cohort>1)', 'if((round!=0&&(round<63||round>65))||cohort<0||cohort>1)')
s=s.replace('s.used=0;s.aborted=false;s.armed=true;', 's.used=0;s.matched=0;s.aborted=false;s.armed=true;')
s=s.replace('if(!s.armed||layer!=38)return false;', 'if(!s.armed||layer!=size_t(s.round==0?s.cohort:38))return false;')
s=s.replace('if(stage!=1||rank<0||rank>=8||!valid)', 'if(stage!=(s.round==0?0:1)||rank<0||rank>=8||!valid||s.matched!=255)')
s=s.replace('{{"layer",38},{"stage",s.stage}', '{{"mode",s.round==0?"prefill-round0":"decode-round63to65"},{"observed_rows",s.rows},{"observed_pasts",s.pasts},{"observed_counts",s.counts},{"matched_requests",s.matched},{"layer",s.round==0?s.cohort:38},{"stage",s.stage}')
s=s.replace('{"past",512+s.round-1}', '{"past",s.round==0?0:512+s.round-1}')
(r/'glm_prefill_window_observer_r8701.cpp').write_bytes(s.encode())
p=r/'glm-window-r5797-src/glm_moe_dsa_balanced_text_model_r126.cpp';s=p.read_text();assert hashlib.sha256(p.read_bytes()).hexdigest()=='d94016e3adc4ffccac58adc24aef415c7069ad2503cb21e230f12753c0b2ad98'
s=s.replace('if(begin_+i==38&&glm_window_r5795::armed()){window_valid5795=requests.size()==8&&tokens==8;', 'if(glm_window_r5795::armed()){window_valid5795=requests.size()==8&&(tokens==8||tokens==4096);')
(r/'glm_prefill_window_model_r8701.cpp').write_bytes(s.encode())
print('TwoTU only: separate round0 past0/count512; cohort0/layer0 cohort1/layer1, stage0; ABI/header unchanged')
