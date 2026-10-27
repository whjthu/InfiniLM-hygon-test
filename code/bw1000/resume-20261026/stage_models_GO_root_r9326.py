import pathlib,json,hashlib
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');commit='b588627ef29920b6fb1ddc5c5d07aa4be385686f'
items=[('GLM-M15-Model-r129/root-GO-after-b588627.json',{'GPU_authorized':True,'source_published':True,'authorized_by':'parent-root','publication_branch':'bw1000-experiments','publication_commit':commit,'controller_manifest_SHA256':'b2045541a6a75a6820860d7708f387a86cee5734e4846b18f248888bd4cfaa05','CPU_receipt_SHA256':'83bd99026decb9f7b38acd4301506854288bfe2887c3265e32b3051f3813c009','offset127_result_SHA256':'5c1a28e7c00019ae1bdf802633ebf2a374fa248356596dba04745acf4f206293','offset127_watch_SHA256':'4695d0eb255d79ce47b3062cef62323e3d3f8aa3b8f622a0ad66daab0b33c866'}),('Kimi-ctx2k-Model-r127/launcher/GPU-GO-after-source-publication.json',{'authorized_by':'parent-root','manifest_SHA256':'ce2ab50fc5d37989f8131c03230eba38c00752fde33bc428480057c0ff104d45','Git_publication_SHA':commit,'publication_branch':'bw1000-experiments','GPU_authorized':True,'source_published':True})]
out={'stage_only':True,'executed_code':False,'files':[]}
for rel,q in items:
 p=r/'incoming'/rel;assert p.parent.is_dir();b=(json.dumps(q,indent=2)+'\n').encode()
 with p.open('xb')as f:f.write(b)
 out['files'].append({'path':str(p),'SHA256':hashlib.sha256(b).hexdigest(),'content':q})
print(json.dumps(out))
