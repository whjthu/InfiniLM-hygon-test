"""Execute the dispatcher's real pre-Slurm bootstrap in an isolated directory."""
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    source=Path(__file__).resolve().parent
    tree=ast.parse((source/'dispatch.py').read_bytes())
    stop,=(i for i,n in enumerate(tree.body) if isinstance(n,ast.FunctionDef) and n.name=='save')
    with tempfile.TemporaryDirectory(prefix='sdxl-bootstrap-') as target:
        target=Path(target)
        for n in ('CLAIM.json','scope.py','slurm_accounting.py'):
            shutil.copyfile(source/n,target/n)
        prefix=ast.unparse(ast.Module(body=tree.body[:stop],type_ignores=[]))
        runner=target/'dispatch_bootstrap.py'
        runner.write_text(prefix,encoding='utf-8')
        command=[sys.executable,'-B','-c',
                 'import runpy,sys,time; path,req=sys.argv[1:]; sys.argv=[path,str(time.time()),req]; runpy.run_path(path)',
                 str(runner),json.loads((source/'CLAIM.json').read_bytes())['request']]
        subprocess.run(command,cwd=target,check=True,timeout=10)
        intent=json.loads((target/'intent.json').read_bytes())
        assert intent['request']==command[-1] and intent['deadline']-intent['epoch']==1200
        again=subprocess.run(command,cwd=target,capture_output=True,timeout=10)
        assert again.returncode and b'FileExistsError' in again.stderr
    print('real HOST bootstrap writes exactly one intent and rejects reuse')


if __name__=='__main__':
    main()
