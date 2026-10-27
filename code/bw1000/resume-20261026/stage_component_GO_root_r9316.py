import pathlib,json,hashlib
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
common={'GPU_authorized':True,'source_published':True,'authorized_by':'parent-root','publication_branch':'bw1000-experiments','publication_commit':'e1968676a0a874f4d6d0d17ef79d010a578313a7','scope':'User-authorized bounded component validation; not whole Model promotion'}
entries=[('GLM-M15-offset8-r127',dict(common,controller_manifest_SHA256='67a303e9102a725c0ceb6be6bb58fbac8918b8103884647eb8e4dd2b9fa3595b',CPU_receipt_SHA256='68cbba51c6f33d06457da1e2b434052de254378ef4563753921eb8b7014e8fd3',parent_allocation='910376',nodes=['a02r4n12'])),('Kimi-TP32-I96-GPU-controller-r120e',dict(common,controller_manifest_SHA256='80a98cba1b6d54c094a6f16c2962908ceeeafc2a8f206c0b054a37d8a0937748',CPU_receipt_SHA256='f206430f2b313fb818375e223527038adb256170450aa88f09fe6cacbf3a1e3f',parent_allocation='910375',nodes=['a02r4n08','a02r4n09','a02r4n10','a02r4n11'],master_port=43120))]
out={'stage_only':True,'executed_code':False,'files':[]}
for folder,q in entries:
 p=r/'incoming'/folder/'root-GO-after-e196867.json';assert p.parent.is_dir()
 b=(json.dumps(q,indent=2)+'\n').encode()
 with p.open('xb')as f:f.write(b)
 out['files'].append({'path':str(p),'SHA256':hashlib.sha256(b).hexdigest(),'content':q})
print(json.dumps(out))
