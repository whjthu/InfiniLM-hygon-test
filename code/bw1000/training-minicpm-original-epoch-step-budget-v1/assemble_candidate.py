"""Materialize only two function deltas against exact original published source."""
import ast
import hashlib
from pathlib import Path

D=Path(__file__).resolve().parent
G=D.parents[2]
ORIGINAL=G/'tools/control_minicpm_usgs_pair_target.py'
PIN={'bytes':15681,'SHA256':'750d54b23f44e670cc152a762a9e977cd72ea88399ecd17bc3f65b800d800bf0'}


def assemble():
    original=ORIGINAL.read_bytes()
    if len(original)!=PIN['bytes'] or hashlib.sha256(original).hexdigest()!=PIN['SHA256']:
        raise ValueError('Original published controller source differs; do not patch unknown bytes')
    helper=(D/'step_budget.py').read_bytes();source=original.decode();replacement=helper.decode()
    old_tree=ast.parse(source);new_tree=ast.parse(replacement)
    names={'step_command','execute_step'}
    old={n.name:n for n in old_tree.body if isinstance(n,ast.FunctionDef) and n.name in names}
    new={n.name:n for n in new_tree.body if isinstance(n,ast.FunctionDef)}
    if set(old)!=names or set(new)!=names:raise ValueError('Exact two-function delta required')
    lines=source.splitlines(keepends=True);new_lines=replacement.splitlines(keepends=True)
    for name,n in sorted(old.items(),key=lambda kv:kv[1].lineno,reverse=True):
        q=new[name];lines[n.lineno-1:n.end_lineno]=new_lines[q.lineno-1:q.end_lineno]
    candidate=''.join(lines).encode()
    parsed=ast.parse(candidate)
    unchanged=lambda t:[ast.dump(n,include_attributes=False) for n in t.body if not(isinstance(n,ast.FunctionDef) and n.name in names)]
    if unchanged(old_tree)!=unchanged(parsed):raise ValueError('Nonbudget controller AST changed')
    proof={'original':PIN,'original_publication':'b7b88058','replacement_functions':sorted(names),
           'helper_SHA256':hashlib.sha256(helper).hexdigest(),'candidate':{'bytes':len(candidate),'SHA256':hashlib.sha256(candidate).hexdigest()},
           'all_other_controller_AST_identical':True,'actually_installed_or_published':False}
    return candidate,proof
