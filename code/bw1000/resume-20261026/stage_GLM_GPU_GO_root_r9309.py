import pathlib,json,hashlib
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');d={'GPU_authorized':True,'source_published':True,'publication_commit':'955a7efe7a7b29184835992260908f2b6bb9aaf1','publication_branch':'bw1000-experiments','controller_manifest_SHA256':'4a4796d29e1d2d6c4551283b40a6427ae76d497656aec8ab8f7cf71762ae769c','CPU_receipt_SHA256':'68cbba51c6f33d06457da1e2b434052de254378ef4563753921eb8b7014e8fd3','scope':'ExistinghumanSlurm/inferenceoptimizationauthorization; exactpublishedsingleM15component1GPU376n12+CPUkeeper407; noModelpromotion/no restart'}
p=r/'incoming/GLM-M15-controller-r123/root-GO-after-publication.json';b=(json.dumps(d,indent=2)+'\n').encode()
if p.exists():assert p.read_bytes()==b
else:
 with p.open('xb')as f:f.write(b)
print(json.dumps({'GO_staged_only':True,'path':str(p),'SHA256':hashlib.sha256(b).hexdigest(),'CPU_submitted':False,'GPU_submitted':False}))
