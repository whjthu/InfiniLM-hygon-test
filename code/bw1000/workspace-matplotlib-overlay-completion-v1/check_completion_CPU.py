"""Only new completion arithmetic/response/ordering boundaries; no GET/copy/install."""
import ast
import json
from pathlib import Path
import urllib.request

D=Path(__file__).resolve().parent


def main():
    tree=ast.parse((D/'complete_worker.py').read_bytes())
    nodes=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))
           and node.name in ('need','validate_response','NoRedirect')]
    ns={'urllib':__import__('urllib')};exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_completion_response','exec'),ns)
    binding=json.loads((D/'ORIGINAL_BINDING.json').read_bytes())
    assert 8122770-3473408+1==binding['range_body_bytes']==4649363
    assert 4649363+113890+11050==binding['new_network_bytes']==4774303
    assert binding['original_attempted_GETs']+binding['new_GET_denominator']==binding['combined_GET_denominator']==8
    headers={'Content-Range':'bytes 3473408-8122770/8122771','Content-Length':'4649363'}
    assert ns['validate_response'](206,headers,{'bytes':8122771},True)==4649363
    rejected=[]
    for name,status,values in [('no200fallback',200,headers),('wrongrange',206,dict(headers,**{'Content-Range':'bytes 3473409-8122770/8122771'})),
                               ('wrong_100k_length',206,dict(headers,**{'Content-Length':'4749363'})),
                               ('encoded_body',206,dict(headers,**{'Content-Encoding':'gzip'}))]:
        try:ns['validate_response'](status,values,{'bytes':8122771},True)
        except ValueError:rejected.append(name)
        else:raise AssertionError(name+' admitted')
    assert ns['NoRedirect']().redirect_request(None,None,302,'redirect',{},'https://other.example/') is None
    source=(D/'complete_worker.py').read_text()
    assert source.index("need(len(whole)==7")<source.index("pip_attempted=True")
    assert any(isinstance(node,ast.Try) and node.finalbody for node in ast.walk(tree))
    assert "RUN/'worker-result.json'" in source and 'combined_GET_denominator=8' in source
    for path in D.glob('*.py'):ast.parse(path.read_bytes())
    print(json.dumps(dict(passed=True,range_bytes=4649363,new_network_bytes=4774303,combined_GET_denominator=8,
                         exact206_accepted=True,rejected=rejected,redirect_rejected=True,
                         all7wholeSHA_gate_before_first_pip=True,worker_finally_failure_receipt_present=True,
                         actual_GETs=0,actual_local_copies=0,pip_installs=0,WDI_GETs=0,Torch=0,GPU=0)))


if __name__=='__main__':main()
