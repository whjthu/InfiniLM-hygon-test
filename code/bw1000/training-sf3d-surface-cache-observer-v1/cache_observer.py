"""Private metadata for the actual native cache; never invent/capture features."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time


class SurfaceCacheObserver:
    def __init__(self, output, *, labels_sha256, input_sha256, planned_objects):
        if type(planned_objects) is not int or planned_objects != 2:
            raise ValueError('Exactly the frozen two training inputs are required')
        if not all(isinstance(s,str) and len(s)==64 and all(c in '0123456789abcdef' for c in s)
                   for s in (labels_sha256,input_sha256)):
            raise ValueError('Actual loaded input/label identities required')
        self.root=Path(output)/'surface-cache-observation'
        self.root.parent.mkdir(parents=True,exist_ok=True);self.root.mkdir()
        self.started=time.monotonic()
        self.state={'schema':'SF3D_NATIVE_SURFACE_CACHE_OBSERVATION_V1','planned_objects':planned_objects,
                    'loaded_labels_SHA256':labels_sha256,'loaded_inputs_SHA256':input_sha256,'phases':[],
                    'cache_fields_and_labels_validated':False,'native_updates_observed_here':0,'tensor_values_exported':False,
                    'Model_GO':False,'formal17_GO':False,'training_READY':False}
        self.write()

    def write(self):
        self.state['elapsed_monotonic_seconds']=time.monotonic()-self.started
        raw=json.dumps(self.state,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'
        if len(raw)>64*1024:raise ValueError('Private cache metadata bound exceeded')
        temporary=self.root/'progress.tmp';temporary.write_bytes(raw);temporary.replace(self.root/'progress.json')

    @contextmanager
    def phase(self,name,object_index=None):
        if name not in {'device_gate','model_init','prepare_batch','scene_codes','query_triplane','image_features','material_fields','validate_cache'}:
            raise ValueError('Unregistered native phase')
        if object_index is not None and (type(object_index) is not int or not 0<=object_index<2):
            raise ValueError('Cache object index differs')
        row={'phase':name,'object_index':object_index,'entered':True,'completed':False}
        self.state['phases'].append(row);self.write();start=time.monotonic()
        try:
            yield
        except BaseException as error:
            # The actual native exception remains primary even if diagnostic
            # storage fails. No raw exception/tensor representation is exported.
            try:
                row.update(error_type=type(error).__name__,message_SHA256=hashlib.sha256(str(error)[:4096].encode()).hexdigest(),
                           message_hash_scope='first4096characters',wall_seconds=time.monotonic()-start)
                self.write()
            except BaseException:pass
            raise
        else:
            row.update(completed=True,wall_seconds=time.monotonic()-start);self.write()

    def validate_actual_cache(self, cpu_batches, targets, feature_spec, torch, *, expected_points):
        # Caller must FIRST use the unchanged native CPU cache validator.
        # This checks actual target values/order and query row count as well.
        if len(cpu_batches)!=2 or len(targets)!=2 or type(expected_points) is not int or expected_points<1:
            raise ValueError('Actual complete two-object cache required')
        summaries=[]
        for index,(batch,target) in enumerate(zip(cpu_batches,targets)):
            if batch['triplane_features'].shape[0]!=expected_points:
                raise ValueError('Actual triplane rows differ from configured query points')
            for name,shape in [('albedo',(1,3)),('roughness',(1,)),('metallic',(1,))]:
                expected=torch.tensor(target[name],device='cpu',dtype=torch.float32).reshape(shape)
                if not torch.equal(batch[name],expected):
                    raise ValueError('Actual cached material target differs from loaded surface labels')
            fields={}
            for name,value in batch.items():
                raw=value.detach().contiguous().view(torch.uint8).numpy().tobytes()
                fields[name]={'shape':list(value.shape),'dtype':str(value.dtype),'bytes':len(raw),
                              'SHA256':hashlib.sha256(raw).hexdigest()}
            summaries.append({'object_index':index,'fields':fields,'actual_loaded_labels_equal':True})
        self.state.update(cache_fields_and_labels_validated=True,completed_objects=2,actual_feature_spec=feature_spec,
                          actual_query_points_per_object=expected_points,objects=summaries)
        self.write()
