"""Only new synthetic metadata/header/no-replay boundaries, zero network/real dataset."""
import copy
from email.message import Message
import json
from pathlib import Path
from metadata import headers_gate,inventory,need,url_gate


def run():
    s=Path(__file__).parent;c=json.loads((s/'CONTRACT.json').read_bytes());ledger=json.loads((s/'WEB_ATTEMPTS.json').read_bytes())
    url_gate(c,ledger);siblings=[]
    for i,name in enumerate(c['expected_parquet_paths']):
        siblings.append(dict(rfilename=name,size=1000000+i,blobId=format(i,'040x'),lfs=dict(size=1000000+i,sha256=format(i,'064x'),pointerSize=133)))
    source=dict(id=c['dataset'],sha=c['revision'],private=False,gated=False,cardData=dict(license='cc-by-4.0'),siblings=siblings)
    result=inventory(source,c);need(result['parquet_file_count']==6 and result['Range_qualified'] is False and result['actual_parquet_body_bytes']==0,'Metadata falsely qualified data/range')
    rejected=[]
    def bad(name,fn):
        try:fn()
        except (ValueError,KeyError,TypeError):rejected.append(name);return
        raise AssertionError('Accepted '+name)
    x=copy.deepcopy(source);x['sha']='0'*40;bad('revision_drift',lambda:inventory(x,c))
    x2=copy.deepcopy(source);x2['siblings'][0].pop('lfs');bad('missing_lfs',lambda:inventory(x2,c))
    x3=copy.deepcopy(source);x3['siblings'][0]['lfs']['size']+=1;bad('size_inconsistent',lambda:inventory(x3,c))
    x4=copy.deepcopy(source);x4['siblings'][0]['rfilename']='../outside.parquet';bad('path_traversal',lambda:inventory(x4,c))
    x5=copy.deepcopy(source);x5['siblings'][1]['lfs']['sha256']=x5['siblings'][0]['lfs']['sha256'];bad('duplicate_LFS_body_ID',lambda:inventory(x5,c))
    x6=copy.deepcopy(source);x6['cardData']['license']='restricted';bad('license_drift',lambda:inventory(x6,c))
    x7=copy.deepcopy(source);x7['siblings'].pop();bad('missing_file_slot',lambda:inventory(x7,c))
    x8=copy.deepcopy(ledger);x8['web_tool_research_only'].append(dict(URL=c['URL']));bad('prior_exact_web_attempt',lambda:url_gate(c,x8))
    x9=copy.deepcopy(c);x9['URL']='https://huggingface.co/datasets/naver-clova-ix/cord-v2/raw/'+c['revision']+'/dataset_infos.json?nonce=new';bad('failed_raw_path_alias',lambda:url_gate(x9,ledger))
    x10=copy.deepcopy(c);x10['URL']=c['URL']+'&nonce=1';bad('query_alias_or_new_scope',lambda:url_gate(x10,ledger))
    headers=Message();headers['Content-Type']='application/json';headers['Content-Length']='200'
    need(headers_gate(200,headers,2097152)==200,'Good JSON header')
    bad('redirect_status',lambda:headers_gate(302,headers,2097152));bad('length_beforebody_cap',lambda:headers_gate(200,headers,199))
    dup=Message();dup['Content-Type']='application/json';dup['Content-Length']='1';dup['Content-Length']='1'
    bad('duplicate_length',lambda:headers_gate(200,dup,2097152))
    encoded=Message();encoded['Content-Type']='application/json';encoded['Content-Encoding']='gzip'
    bad('encoded_body',lambda:headers_gate(200,encoded,2097152))
    paged=Message();paged['Content-Type']='application/json';paged['Link']='<more>; rel="next"'
    bad('pagination_follow_forbidden',lambda:headers_gate(200,paged,2097152))
    return dict(passed=True,normal_contracts=3,rejected_count=len(rejected),rejected=rejected,synthetic_files=6,
        actual_HTTP_requests=0,real_repository_metadata_read=0,real_media_read=0,blind_gold_read=0,old_tests_repeated=False,Dataset_GO=False,training_GO=False,GPU_GO=False)


if __name__=='__main__':print(json.dumps(run(),sort_keys=True))
