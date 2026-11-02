from pathlib import Path
import ast,json,hashlib
D=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 for p in D.glob('*.py'):ast.parse(p.read_text());compile(p.read_bytes(),str(p),'exec')
 assert b'\r'not in(D/'CPU_inside.sh').read_bytes()
 files=sorted(p.name for p in D.iterdir()if p.is_file()and p.name!='CPU_SOURCE_MANIFEST.json')
 m=dict(schema=1,files_SHA256={n:sha(D/n)for n in files},CPU_claim=dict(parent=910407,node='a14r2n20',cores=1,mem_MiB=2048,total_s=120,inside_s=100,attempts=1,GPU=0,Native_dlopen=False,checkpoint_payload_IO=False,HOST_TUs_compiled=0,GPU_TUs_compiled=0,link_invocations=0),CPU800_remains_FAILED=True,reused_library_SHA256='a116adc27d070aea76c65b54527118f0653461620e55d1224c63d0d430f300e7',GPU_GO=False,Model_GO=False)
 p=D/'CPU_SOURCE_MANIFEST.json';p.write_text(json.dumps(m,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(SHA256=sha(p),members=len(files),GPU_GO=False)))
if __name__=='__main__':main()
