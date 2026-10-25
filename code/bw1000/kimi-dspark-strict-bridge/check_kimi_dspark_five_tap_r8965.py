"""CPU contract: real existing four-frontend TCP collector; no model/GPU."""
import pathlib,tempfile,json,hashlib,os,socket,threading,base64,ast
from kimi_dspark_five_tap_r8965 import *
from stage_control_r2386 import StageControl

def main():
    ranges=[(0,24),(24,47),(47,70),(70,93)];local=[];errors=[];answers=[]
    old=os.environ.get('INFINILM_KIMI_TRACE_DIR')
    with tempfile.TemporaryDirectory()as tmp:
        for node in range(4):
            directory=pathlib.Path(tmp)/str(node)
            with target_capture(directory,0):
                assert os.environ['INFINILM_KIMI_TRACE_LAYERS']=='8,24,52,68,84'
                assert os.environ['INFINILM_KIMI_TRACE_BOUNDARY_POINTS_R8506']=='attn_res'
                records=[]
                for L in CONSUMER_LAYERS:
                    if not ranges[node][0]<=L<ranges[node][1]:continue
                    # Synthetic distinct tap signatures ONLY for wire/schema.
                    raw=(int(L).to_bytes(2,'little'))*(2*HIDDEN_SIZE)
                    (directory/f'stage{node}-rank0-layer{L}-attn_res-full-bf16.bin').write_bytes(raw)
                    records.append(dict(stage=node,rank=0,layer=L,point='attn_res',hidden=HIDDEN_SIZE,tokens=2,packed_target_only=True,selected_shape=[1,2,HIDDEN_SIZE],requests=[dict(request=0,nonfinite=0)]))
                (directory/f'stage{node}-rank0.jsonl').write_text(''.join(json.dumps(v)+'\n'for v in records))
            assert os.environ.get('INFINILM_KIMI_TRACE_DIR')==old
            local.append(local_features(directory,node=node,stage=node,ranges=ranges,tp_nodes=1,tokens=2,jobseq=4))
        with socket.socket()as probe:probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
        def worker(node):
            control=None
            try:
                control=StageControl(node,'127.0.0.1',port,'cpu-5tap-only',4,5)
                answers.append(gather_features(control,local[node],jobseq=4,tokens=2))
            except BaseException as e:errors.append(repr(e))
            finally:
                if control:control.close()
        threads=[threading.Thread(target=worker,args=(i,))for i in range(4)]
        for t in threads:t.start()
        for t in threads:t.join(10)
        assert not errors and not any(t.is_alive()for t in threads),(errors,threads)
        assert len(answers)==4 and all(v==answers[0]for v in answers)
        req,body=committed_packet(answers[0],captured_tokens=2,retain_tokens=1,prefix_start=512,committed_target_past=513,anchor=17,case='x',reset=False)
        assert len(body)==5*HIDDEN_SIZE*2 and [int.from_bytes(body[k*HIDDEN_SIZE*2:k*HIDDEN_SIZE*2+2],'little')for k in range(5)]==list(CONSUMER_LAYERS)
        rejected=[]
        def bad(label,fn):
            try:fn()
            except(AssertionError,KeyError):rejected.append(label)
            else:raise AssertionError('acceptedbad:'+label)
        bad('uncommittedfuture',lambda:committed_packet(answers[0],captured_tokens=2,retain_tokens=2,prefix_start=512,committed_target_past=513,anchor=17,case='x',reset=False))
        bad('old6tap',lambda:check_feature_packet(dict(req,target_layers=[19,37,54,66,78,90]),bytes(6*HIDDEN_SIZE*2)))
        bad('wrongsemantic',lambda:check_feature_packet(dict(req,stream_semantics='rawresidual'),body))
        class Collected:
            def collect(self,label,payload):return [payload]
        bad('duplicatetap',lambda:gather_features(Collected(),answers[0]+answers[0][:1],jobseq=4,tokens=2))
        try:
            with target_capture(pathlib.Path(tmp)/'exception'):raise RuntimeError('injected')
        except RuntimeError:pass
        assert os.environ.get('INFINILM_KIMI_TRACE_DIR')==old
        # Thin worker prefix ledger uses actual finish() and verifies cachecrop.
        class Cache:
            n=520
            def get_seq_length(self):return self.n
            def crop(self,n):self.n=n
        state=DraftPrefix();state.past=512;state.case='x';cache=Cache();state.finish(req,cache);assert state.past==513 and cache.n==513
        return dict(CPU_only=True,GPU=False,passed=True,real_TCP_frontends=4,
           target_capture_source_executed=False,synthetic_wire_fixture=True,
           ordered_consumers=list(CONSUMER_LAYERS),accepted_prefix_crop=True,
           negative_cases=rejected,exception_env_restore=True,cache_crop=513,
           trained_backbone_forward='NOT_RUN_TORCH_OR_NEW_CHECKPOINT_UNAVAILABLE',
           scope='Actual local_features/gather/committedpacket/finish functions; actualexistingStageControl overTCP. SyntheticbytesnotTargetfeatures; noquantteacheracceptance/Modelquality/speed.')

if __name__=='__main__':
    q=main();p=pathlib.Path('Kimi-DSpark-five-tap-CPU-r8965.json');assert not p.exists();p.write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps(q))
