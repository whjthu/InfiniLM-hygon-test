"""New blind schema/clock/wiring contracts only; synthetic no media/model/provider execution."""
import ast,copy,hashlib,json,time
from pathlib import Path
from prepare_inputs import validate_rows
D=Path(__file__).resolve().parent


def main():
    rows=[dict(id='synthetic-'+str(i),chart_group=i//5,source_group=i%3,messages=[dict(role='user',content='synthetic')],images=['blind/media/'+str(i//5)+'.png']) for i in range(25)]
    media={'blind/media/'+str(i)+'.png':{} for i in range(5)}
    validate_rows(rows,media);accepted=1;rejected=[]
    for name,mutate in [
        ('missing25',lambda x:x.pop()),('duplicate_id',lambda x:x[1].update(id=x[0]['id'])),
        ('assistant',lambda x:x[0]['messages'].append(dict(role='assistant',content='forbidden'))),
        ('labels',lambda x:x[0].update(labels='forbidden')),('foreign_image',lambda x:x[0].update(images=['dev/media/0.png'])),
        ('wrong_groups',lambda x:[r.update(source_group=0) for r in x])]:
        bad=copy.deepcopy(rows);mutate(bad)
        try:validate_rows(bad,media)
        except ValueError:rejected.append(name)
        else:raise AssertionError(name+' not refused')
    source=(D/'run_CPU_once.py').read_bytes();module=ast.parse(source)
    only=ast.Module(body=[x for x in module.body if isinstance(x,ast.FunctionDef) and x.name=='schedule'],type_ignores=[])
    import math
    def need(ok,why):
        if not ok:raise ValueError(why)
    env=dict(math=math,need=need);exec(compile(only,'schedule-only','exec'),env)
    clock=env['schedule'](1000.,1100.)
    assert [clock[k]-1000 for k in ['deadline','work_deadline','supervisor_stop','held_cleanup_deadline','registry_cleanup_deadline','alarm_deadline','client_deadline']]==[300,260,270,282,285,290,295]
    for epoch,now in [(1000.,1260.),(1000.,999.),(float('nan'),1100.)]:
        try:env['schedule'](epoch,now)
        except ValueError:rejected.append('clock:'+repr((epoch,now)))
        else:raise AssertionError('bad clock accepted')
    submit=(D/'submit_once.py').read_text();assert "return 0 if remote and remote.get('passed') is True else 1" in submit
    tree=ast.parse(submit);fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
    assert isinstance(fn.body[0],ast.Assign) and ast.unparse(fn.body[0].value)=='time.time()'
    native=(D/'native_template_CPU.py').read_text()
    assert "for split in ['blind']" in native and "template.set_mode('transformers')" in native
    assert "encoded.get('labels') is None and labels is None" in native
    assert "str(a.data / 'train-swift.jsonl')" not in native
    parsed=0
    for path in D.glob('*.py'):ast.parse(path.read_bytes());parsed+=1
    return dict(passed=True,synthetic_valid=accepted,rejected_count=len(rejected),rejected=rejected,AST_files=parsed,
        source_AST_wiring_checks=5,real_blind_inputs_media_gold_read=False,real_processor_calls=0,network_calls=0,device_calls=0)


if __name__=='__main__':print(json.dumps(main(),allow_nan=False))
