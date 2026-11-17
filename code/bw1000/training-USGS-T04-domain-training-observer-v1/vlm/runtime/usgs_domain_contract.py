"""Independent explicit T04 domain contract; original396/CUDA64 contracts unchanged."""
import hashlib,json,math,os,time,socket
from pathlib import Path
from vlm.runtime.cuda_resume_contract import (MEMBERS,bounded_bytes,checked_path,require,
    verify_file)
SCHEMA='USGS_T04_DOMAIN_TRAINING_V1'
MODES={'fresh25':(0,25,100),'resume20to25':(20,25,20)}

def remaining(config):
    epoch,deadline=config['epoch'],config['deadline']
    require(type(epoch) in (int,float) and type(deadline) in (int,float) and math.isfinite(epoch) and math.isfinite(deadline)
        and 0<deadline-epoch<=config['maximum_seconds']<=7200 and epoch<=time.time()<deadline,'DOMAIN_ORIGINAL_EPOCH')
    return deadline-time.time()

def check_config(c,*,clock=True):
    require(c['schema']==SCHEMA and c['case']=='T04.sft' and c['arm'] in ('domain','generic') and c['mode'] in MODES,'DOMAIN_CASE_MODE')
    require(c['root_GO'] is True and c['GPU_GO'] is True and c['device']=='cuda' and c['formal17'] is False,'DOMAIN_OPTIN')
    require(c['world_size']==8 and c['physical_rows']==800 and c['unique_rows']==45 and c['max_steps']==25 and c['accumulation']==4,'DOMAIN_DENOMINATOR')
    require(c['initial_tensors']==504 and c['initial_elements']==43646976 and c['comparison_policy']=='EXACT_ENGINEERING_SAVED_STATE','DOMAIN_STATE_POLICY')
    require(isinstance(c['pair_id'],str) and len(c['pair_id'])==32 and not set(c['pair_id'])-set('0123456789abcdef'),'DOMAIN_PAIR_ID')
    for key in ('binding','native45','training800','occurrences'):
        pin=c[key];checked_path(pin['path']);require(type(pin['bytes']) is int and 0<pin['bytes']<=4*1024**2 and len(pin['SHA256'])==64 and not set(pin['SHA256'])-set('0123456789abcdef'),'DOMAIN_PIN')
    require(c['mode']!='fresh25' or c['arm']!='generic' or (c.get('initial_state') is not None and c.get('pair_initial_reference') is not None),'GENERIC_REQUIRES_ACTUAL_DOMAIN_INITIAL')
    if c['mode']=='resume20to25':
        require(c.get('continuous_reference') is not None and c.get('checkpoint_files') is not None and c.get('comparison_files') is not None,'OWN_CONTINUOUS_REFERENCE_REQUIRED')
        for rootkey,pinkey in [('checkpoint','checkpoint_files'),('comparison_checkpoint','comparison_files')]:
            checked_path(c[rootkey]);pins=c[pinkey]
            require(set(pins)==MEMBERS and sum(p['bytes'] for p in pins.values())<=512*1024**2,'FULL14_STATE_MEMBERS')
        require(Path(c['checkpoint']).name=='checkpoint-20' and Path(c['comparison_checkpoint']).name=='checkpoint-25','OWN20_25_PATHS')
    if clock:remaining(c)
    return c

def load_config():
    require(os.environ.get('USGS_T04_DOMAIN_OPTIN')=='1','DOMAIN_EXPLICIT_OPTIN')
    raw=bounded_bytes(checked_path(os.environ['USGS_T04_DOMAIN_CONFIG']),2*1024**2)
    require(hashlib.sha256(raw).hexdigest()==os.environ['USGS_T04_DOMAIN_CONFIG_SHA256'],'DOMAIN_CONFIG_PIN')
    return check_config(json.loads(raw))

def read_pin(pin,maximum=4*1024**2):
    raw=bounded_bytes(checked_path(pin['path']),maximum)
    require(len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'],'DOMAIN_ACTUAL_FILE_PIN')
    return raw

def load_inputs(c):
    binding=json.loads(read_pin(c['binding']));require(binding['spec']['case']=='T04.sft','DOMAIN_BINDING_CASE')
    rows=[r for r in json.loads(read_pin(c['native45'])) if r['split']=='train']
    require(len(rows)==45 and [r['ordinal'] for r in rows]==list(range(45)) and all(r['full_input_retained'] and r['supervised_EOS_count']>0 and not r['labels_None'] for r in rows),'ACTUAL_NATIVE45_ROWS')
    view=read_pin(c['training800']);require(len(view.splitlines())==800,'ACTUAL_PHYSICAL800')
    occurrence=json.loads(read_pin(c['occurrences']));indices=occurrence['original_index_by_physical_row']
    require(len(indices)==800 and all(type(i)is int and 0<=i<45 for i in indices) and occurrence['view_SHA256']==c['training800']['SHA256'] and occurrence['arm']==c['arm'],'ACTUAL_OCCURRENCE_BINDING')
    actual=[json.loads(line) for line in view.splitlines()]
    require(all(hashlib.sha256(row['id'].encode()).hexdigest()==rows[index]['source_id_SHA256'] for row,index in zip(actual,indices)),'PHYSICAL800_SOURCE_ID_MAPPING')
    return binding,rows,occurrence

def cursor_contract(mode,length,initial,initialized,saved_epoch=None):
    require(length==100 and len(initial)==len(initialized)==1,'ACTUAL_LOADER100')
    values=initial[0]['returned_values'];start=MODES[mode][0];updates=(length+3)//4;epochs=start//updates;skip=(start%updates)*4
    require(len(values)==7 and values[1]==updates and values[2]==800 and values[4]==32 and values[5]==length and values[6]==25 and initial[0]['actual_loader_length']==length,'ACTUAL_TRAINER_INITIAL_RETURNS')
    require(initialized[0]['returned_values']==[epochs,skip],'ACTUAL_TRAINER_SKIP_RETURN')
    if mode=='resume20to25':require(saved_epoch==0.8 and skip==80,'ACTUAL_CHECKPOINT20_CURSOR')
    return dict(loader_length=length,updates_per_epoch=updates,trained_epochs=epochs,skipped_microbatches=skip,expected_microbatches=MODES[mode][2],physical_duplicate_index_observed=False)


def verify_supervisor(config, *, direct_parent=False):
    """Exact existing A100 controller identity; no process scans or foreign-host PID inference."""
    row = config['supervisor']
    require(row['host'] == socket.gethostname() and row['uid'] == os.getuid()
            and row['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'CUDA resume supervisor host/UID/boot differs')
    require(type(row['PID']) is int and row['PID'] > 0
            and (not direct_parent or row['PID'] == os.getppid()), 'CUDA resume owned supervisor differs')
    proc = Path('/proc') / str(row['PID'])
    require(proc.stat().st_uid == os.getuid(), 'CUDA resume supervisor UID differs')
    fields = (proc / 'stat').read_text().split(') ', 1)[1].split()
    require(int(fields[19]) == row['startticks'] and fields[0] not in ('Z', 'X'),
            'CUDA resume supervisor identity not live')
    require(row.get('scope_request_id') == config['request_id'], 'CUDA supervisor belongs to another request')
    age = float(Path('/proc/uptime').read_text().split()[0]) - row['startticks'] / os.sysconf('SC_CLK_TCK')
    require(0 <= age <= time.time() - config['epoch'] + 10,
            'CUDA supervisor must be a new instance for this original epoch')
    return row


def verify_provider_files(config):
    require(len(config['providers']) == 15, 'CUDA resume provider source denominator differs')
    roots = [Path('/home/whj/nvidia-baseline/env/lib/python3.12/site-packages'),
             Path('/usr/local/lib/python3.12/dist-packages')]
    result = []
    for row in config['providers']:
        path = Path(row['canonical_path'])
        allowed = any(path.is_relative_to(root) for root in roots)
        allowed |= path == Path('/home/whj/nvidia-baseline/swift_callback_nvidia_baseline_v2.py')
        require(path.is_absolute() and allowed and path.resolve(strict=True) == path
                and 0 < row['bytes'] <= 1024**2, 'CUDA resume provider canonical path differs')
        remaining(config); raw = bounded_bytes(path, 1024**2, library=True)
        require(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['SHA256'],
                'CUDA resume provider source changed')
        result.append(dict(path=str(path), bytes=len(raw), SHA256=row['SHA256']))
    return result




def validate_supervised_counter(reference,actual_count,actual_eos):
    require(type(actual_count)is int and actual_count>0 and type(actual_eos)is int and actual_eos>0
        and actual_eos==reference['supervised_EOS_count'],'ACTUAL_SUPERVISED_TOKENS_EOS')
    available='effective_supervised_tokens' in reference
    if available:require(type(reference['effective_supervised_tokens'])is int and actual_count==reference['effective_supervised_tokens'],'REFERENCE_SUPERVISED_COUNTER')
    return available
