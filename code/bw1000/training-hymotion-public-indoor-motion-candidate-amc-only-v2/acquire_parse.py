"""Exactly one official CMU trial, its calibration, and its original annotation page."""
import argparse
import hashlib
from html.parser import HTMLParser
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from parse_asf_amc import need,parse_asf,parse_amc

D=Path(__file__).resolve().parent
PRIOR=D.parents[2]/'local-only/training-hymotion-public-indoor-motion-candidate-v1-original'
PRIOR_PINS={'outcome.json': {'bytes': 1147, 'SHA256': '9fa1c14c3f981c952461029dd23b073a962a2c1362fa4328352925526655f755'}, 'result.json': {'bytes': 1492, 'SHA256': '13ab7ce53d1241e6e19baee1fe70765559cef2a7562b75845f34b27b6c1c7781'}, 'subject79-official.html': {'bytes': 42202, 'SHA256': '091f5700f8511d8f91d1be241a6f70b634ded937aa5fd8c446a87a4a869a51d5'}, '79.asf': {'bytes': 7257, 'SHA256': 'aa7fb254b0e7b9155df59e0f0cf76cbe34c476c6f4b3f015e63d5ae170d2fe2d'}}
def cached(name):
    raw=(PRIOR/name).read_bytes();need(pin(raw)==PRIOR_PINS[name],'original prerequisite bytes differ');return raw
PAGE='http://mocap.cs.cmu.edu/search.php?subjectnumber=79'
SUBJECT=79
TRIAL=81


def pin(raw):return dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())


def save(path,value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_bytes(raw);temp.replace(path)


class Rows(HTMLParser):
    def __init__(self):super().__init__(convert_charrefs=True);self.rows=[];self.links=[];self.row=None
    def handle_starttag(self,tag,attrs):
        if tag=='tr':self.row={'text':[],'links':[]}
        if tag=='a':
            href=dict(attrs).get('href')
            if href:
                self.links.append(href)
                if self.row is not None:self.row['links'].append(href)
    def handle_data(self,data):
        if self.row is not None:self.row['text'].append(data)
    def handle_endtag(self,tag):
        if tag=='tr' and self.row is not None:self.rows.append(self.row);self.row=None


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*_):return None


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--deadline',type=float,required=True);p.add_argument('--gate',required=True)
    a=p.parse_args();root=Path(a.root)
    while not Path(a.gate).exists():
        need(time.time()<a.deadline,'Job admission deadline');time.sleep(.01)
    gate=json.loads(Path(a.gate).read_bytes());need(gate['CPU_count']==2 and gate['memory_limit_bytes']==2147483648,'actual Job limits differ')
    prior_out=json.loads(cached('outcome.json'));prior_result=json.loads(cached('result.json'))
    need(prior_out['state']=='KNOWN_TERMINAL' and prior_out['Job']['returncode']==1 and prior_result['error_type']=='AttributeError','prior must remain known failure')
    need([x['name'] for x in prior_result['requests']]==['subject79-official.html','79.asf'],'AMC must be previously unrequested')
    result=dict(schema='CMU_OFFICIAL_INDOOR_AMC_ONLY_CANDIDATE_V2',passed=False,subject=SUBJECT,trial=TRIAL,
        planned_motion_files=1,completed_motion_files=0,planned_skeleton_files=1,completed_skeleton_files=0,
        requests=[],download_bytes=0,actual_model_or_GPU_or_training=0,training_GO=False,READY=False,
        official_server_revision='UNVERSIONED; freeze actual original bytes and metadata SHA',
        license='CMU custom research/product-use grant; no direct resale even converted; not CC0 or CC-BY',
        TLS_source_authentication=False,blind_or_existing_training_data_read=False)
    opener=urllib.request.build_opener(NoRedirect())
    def fetch(url,name,cap):
        need(name=='79_81.amc' and time.time()<a.deadline and len(result['requests'])<1,'original request/clock budget exhausted')
        u=urllib.parse.urlsplit(url);need(u.hostname=='mocap.cs.cmu.edu' and u.scheme in ('http','https') and not u.username and not u.password,'exact original publisher host required')
        need(url not in [r['URL'] for r in result['requests']],'no request replay')
        r={'URL':url,'name':name,'attempted':True,'status':None,'bytes':0,'EOF':False,'TLS':u.scheme=='https'}
        result['requests'].append(r);save(root/'progress.json',result)
        try:
            req=urllib.request.Request(url,headers={'Accept-Encoding':'identity'})
            with opener.open(req,timeout=min(20,a.deadline-time.time())) as response:
                r['status']=response.status;r['Content_Type']=response.headers.get('Content-Type');r['Content_Length']=response.headers.get('Content-Length')
                need(response.status==200 and response.headers.get('Content-Encoding','identity') in ('','identity'),'unencoded HTTP200 required')
                if r['Content_Length'] is not None:need(r['Content_Length'].isdigit() and 0<int(r['Content_Length'])<=cap,'declared source length exceeds cap')
                chunks=[]
                with (root/name).open('xb') as f:
                    while r['bytes']<=cap:
                        left=a.deadline-time.time();need(left>0,'original download epoch exhausted')
                        if response.fp is not None:response.fp.raw._sock.settimeout(min(20,left))
                        b=response.read1(min(65536,cap+1-r['bytes']))
                        if not b:r['EOF']=True;break
                        r['bytes']+=len(b);result['download_bytes']+=len(b)
                        need(result['download_bytes']<=5*1024*1024,'aggregate data cap')
                        f.write(b);chunks.append(b)
                    f.flush();os.fsync(f.fileno())
                body=b''.join(chunks);need(0<len(body)<=cap and r['EOF'],'complete bounded original required')
                if r['Content_Length'] is not None:need(len(body)==int(r['Content_Length']),'natural EOF and declared length disagree')
                r.update(pin(body));return body
        except urllib.error.HTTPError as e:
            r.update(status=e.code,Content_Type=e.headers.get('Content-Type'),failure_type=type(e).__name__,error_body_read=False)
            e.close();raise
        except Exception as e:
            r['failure_type']=type(e).__name__;raise
        finally:save(root/'progress.json',result)
    try:
        html=cached('subject79-official.html');parser=Rows();parser.feed(html.decode('utf-8',errors='strict'))
        selected=[r for r in parser.rows if any(re.search(r'(?:^|/)79_81\.amc$',urllib.parse.urlsplit(x).path) for x in r['links'])]
        need(len(selected)==1,'exact official trial row not found')
        text=' '.join(' '.join(selected[0]['text']).split());need('brushing teeth' in text.lower() and re.search(r'\b60\b',text),'official action/fps differs')
        def unique_link(links,pattern):
            found={urllib.parse.urljoin(PAGE,x) for x in links if re.search(pattern,urllib.parse.urlsplit(x).path)}
            need(len(found)==1,'exact original file link not unique');return found.pop()
        amc_url=unique_link(selected[0]['links'],r'(?:^|/)79_81\.amc$')
        asf_url=unique_link(parser.links,r'(?:^|/)79\.asf$')
        asf=cached('79.asf');need(asf_url==prior_result['requests'][1]['URL'],'original skeleton source differs');need(len(asf)==int(prior_result['requests'][1]['Content_Length']),'original skeleton CL incomplete');result['completed_skeleton_files']=1
        amc=fetch(amc_url,'79_81.amc',4*1024*1024);result['completed_motion_files']=1
        skeleton=parse_asf(asf);frames=parse_amc(amc,skeleton)
        import numpy as np
        native=np.asarray(frames,dtype=np.float64);need(native.ndim==2 and native.shape[1]==len(skeleton['channels']) and np.isfinite(native).all(),'actual native channel array invalid')
        translation=native[:,:3]*skeleton['root_length_to_meters']
        np.savez_compressed(root/'native-motion-not-HY201.npz',native_channels=native,root_translation_meters=translation,
            frame_numbers=np.arange(1,len(frames)+1,dtype=np.int32),source_fps=np.asarray(60,dtype=np.int32))
        result.update(passed=True,state='ONE_OFFICIAL_NATIVE_MOTION_PARSED_NOT_HY201',official_annotation='brushing teeth',annotation_page=pin(html),
            source_fps=60,actual_original_frames=len(frames),native_shape=list(native.shape),calibrated_bones=skeleton['bone_count'],
            original_native_channels=skeleton['channels'],root_translation_scale_to_meters=skeleton['root_length_to_meters'],
            editing_only_bones_excluded_from_future_measured_joint_claim=skeleton['editing_only_bones'],
            source_duration_seconds=(len(frames)-1)/60,has_at_least128_original_frames=len(frames)>=128,
            zero_padding_or_synthetic_frames_added=0,retargeted_motion135=False,body_model_keypoints66=False,HY201_cache_qualified=False,
            split={'train_candidate_recordings':1,'dev_recordings':0,'blind_recordings':0,'entire_subject79_reserved_for_train_candidate':True,
                   'same_recording_windows_and_derivatives_never_cross_split':True,'actor_disjointness_not_proven_across_subject_numbers':True},
            actual_t08_formal_recipe_modified=False,numpy=importlib.metadata.version('numpy'),
            required_bridge=['native ASF hierarchy/Euler rotations need validated retargeting to original body-model joint order',
                'freeze original temporal resampling/coordinate convention, normalization statistics and 135-feature definition',
                'use qualified original body_model to regenerate66 keypoints and original text conditions768/4096',
                'produce actual128-frame201 native cache with original schema and matching new A100 numeric/recovery reference; never pad/fabricate201'])
        save(root/'source-skeleton-metadata.json',skeleton)
    except Exception as e:result.update(state='FAILED_PARTIAL_OR_UNAVAILABLE',error_type=type(e).__name__,reason=str(e)[:512])
    result['prior_failure_prerequisites']=PRIOR_PINS
    result['prior_actual_GET_attempts']=2
    result['new_actual_GET_attempts']=len(result['requests'])
    result['no_prior_URL_replay']=True
    save(root/'result.json',result)
    print(json.dumps({k:result[k] for k in ('passed','state','download_bytes','completed_motion_files','completed_skeleton_files','READY')},sort_keys=True))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
