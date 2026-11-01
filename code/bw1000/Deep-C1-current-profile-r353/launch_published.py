"""One published exact source claim; verify final CPU685 before owned keeper."""
import pathlib,json,hashlib,subprocess,re,argparse
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
C=R/'incoming/Deep-C1-current-profile-r353/repair-r1/controller-v2'
OWNER='60bb0a4341ccb574fd8d14a5971cbf651cade93f8d9e5be78ffbe29fc7033c99'
CONTROLLER='b18c57d433e6227af5b9aed91c58e76eea0e32f6c52ed34d09f04dc4e511f485'
CPU685='ca01c1132a0dd62a7f3b5c33ddbf77517258e2fe3be01615fbcb8ca28eca71df'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--publication-commit',required=True);a=p.parse_args();assert re.fullmatch('[0-9a-f]{40}',a.publication_commit)
 cp=R/'candidates/Deep-C1-current-profile-r353r1-CPU/controller-contract.json';assert sha(cp)==CPU685;q=json.loads(cp.read_bytes());assert q['completed']and q['passed']and q['CPU_only']and not q['GPU']and not q['Native_dlopen']and not q['weights']and q['actual_step']=='910407.685'and q['owner_manifest_SHA256']==OWNER and q['controller_manifest_SHA256']==CONTROLLER
 assert sha(C/'SOURCE_MANIFEST.json')==CONTROLLER
 g=json.loads((C/'execution-GO.template.json').read_bytes());g.update(GPU_authorized=True,publication_commit=a.publication_commit,owner_manifest_SHA256=OWNER,controller_manifest_SHA256=CONTROLLER,final_CPU685_receipt_SHA256=CPU685)
 output=R/'results/Deep-C1-current-profile-Model-r353r1-execution-go.json'
 with output.open('x')as f:json.dump(g,f,indent=2)
 z=subprocess.run(['python3',str(C/'invoke_once_348.py'),'--execution-go',str(output)],check=True,text=True,capture_output=True,timeout=80)
 print(z.stdout)
if __name__=='__main__':main()
