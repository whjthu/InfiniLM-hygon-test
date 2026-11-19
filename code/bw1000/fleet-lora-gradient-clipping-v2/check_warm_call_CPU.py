"""Exercise the actual warm-loop caller with locally available stdlib mocks."""
import ast
from pathlib import Path
from types import SimpleNamespace


def main():
    source = ast.parse(Path(__file__).with_name('clip_operator.py').read_bytes())
    exercise, = (n for n in source.body if isinstance(n, ast.FunctionDef))
    start, = (i for i, n in enumerate(exercise.body)
              if isinstance(n, ast.Expr) and ast.unparse(n.value) == 'active()')
    stop, = (i for i,n in enumerate(exercise.body) if isinstance(n, ast.Assign)
             and ast.unparse(n.targets[0]) == 'records')
    calls = []
    params, optimizer = object(), object()
    now = iter((0, 0, 1000))
    scope = dict(active=lambda: calls.append('active'),
                 model=lambda packed: (params, optimizer, packed),
                 update=lambda p,o,k,step: calls.append((p,o,k,step)), device='fake-dcu',
                 torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda device: calls.append(device))),
                 end=500, time=SimpleNamespace(time=lambda: next(now)))
    exec(compile(ast.Module(body=exercise.body[start:stop],type_ignores=[]), '<warm-loop>', 'exec'),scope)
    assert calls == ['active',(params,optimizer,False,0),'fake-dcu',(params,optimizer,False,1),'fake-dcu']
    print('warm-loop regression passed')


if __name__ == '__main__':
    main()
