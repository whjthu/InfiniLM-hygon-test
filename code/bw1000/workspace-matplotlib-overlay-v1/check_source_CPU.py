"""Local lock/environment/source guards only; no GET, pip or Matplotlib import."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

D=Path(__file__).resolve().parent


def main():
    for path in D.glob('*.py'):ast.parse(path.read_bytes())
    sys.path.insert(0,str(D))
    from overlay_env import environment,OVERLAY,RUN
    original=dict(PYTHONPATH='unrelated-existing-path',MARKER='unchanged')
    value=environment(original)
    assert original==dict(PYTHONPATH='unrelated-existing-path',MARKER='unchanged')
    assert value['PYTHONPATH']==str(OVERLAY) and value['MPLCONFIGDIR']==str(RUN/'mplconfig') and value['MARKER']=='unchanged'
    import install_once
    try:install_once.main('a'*40,'b'*64)
    except ValueError as error:assert str(error)=='ROOT_GO_FALSE_PREPARATION_ONLY'
    else:raise AssertionError('source-only claim admitted install')
    lock=json.loads((D/'WHEEL_LOCK.json').read_bytes())
    assert len(lock['wheels'])==7 and sum(row['bytes'] for row in lock['wheels'].values())==10823520
    requirements=(D/'requirements.lock').read_text()
    for name,row in lock['wheels'].items():
        assert name+'=='+row['version']+' --hash=sha256:'+row['SHA256'] in requirements
        assert row['url'].startswith('https://files.pythonhosted.org/')
        from packaging.specifiers import SpecifierSet
        assert SpecifierSet(row['requires_python']).contains('3.12.14',prereleases=True)
        assert SpecifierSet(row['wheel_requires_python']).contains('3.12.14',prereleases=True)
    from job_guard import directory_bytes,DIRECTORY_CAP
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);(root/'a').write_bytes(b'a'*7);(root/'b').write_bytes(b'b'*9)
        assert directory_bytes(root)==16 and DIRECTORY_CAP<128*1024**2
    script=D.parents[2]/'code/bw1000/training-worldbank-macro-chart-v1/execute_once.py'
    tree=ast.parse(script.read_bytes())
    call=next(node for node in ast.walk(tree) if isinstance(node,ast.Call) and getattr(node.func,'id',None)=='guarded_run')
    assert not call.keywords
    assert "'-I'" not in script.read_text() and '"-I"' not in script.read_text()
    assert 'torch' not in sys.modules and 'matplotlib' not in sys.modules
    print(json.dumps(dict(passed=True,exact7wheel_and_hash_requirements=True,per_command_env_copy_unchanged=True,
        source_claim_blocks_before_install=True,own_directory_size_fixture_verified=True,WDI_sys_executable_child_inherits=True,
        actual_Windows_job_or_install_tested=False,pip_installs=0,GETs=0,WDI_GETs=0,Torch_imports=0,GPU=0)))


if __name__=='__main__':main()
