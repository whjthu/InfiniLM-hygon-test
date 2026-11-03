"""Only the changed actual consumer, using the original GPU1348 Sum MAX output."""
from pathlib import Path
import ast,json

def main():
 d=Path(__file__).parent;q=json.loads((d/'METADATA_COUNTEREXAMPLE.json').read_bytes())
 root=ast.parse((d/'GPU_driver.py').read_text());main=next(x for x in root.body if isinstance(x,ast.FunctionDef)and x.name=='main')
 fn=next(x for x in main.body if isinstance(x,ast.FunctionDef)and x.name=='record_sdk_metadata');report={};env={'report':report}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'actual_changed_metadata_consumer','exec'),env)
 metadata=q['metadata'];assert metadata['SDK_default_metadata_only']['Sum']=={'active':[8],'default':18446744073709551615,'needs_resolution':False}
 env['record_sdk_metadata'](metadata)
 assert report['SDK_metadata']==metadata and report['SDK_metadata_used_as_math_gate']is False
 return dict(passed=True,actual_changed_consumer=True,original_GPU1348_provenance=q['provenance'],Sum_MAX_recorded_without_rejection=True,nine_metadata_items_recorded=len(metadata['SDK_default_metadata_only'])==9,old_controller_decoder_CXX14_not_rerun=True,no_new_job_or_GPU_or_Native=True,mathematical_qualification=False)

if __name__=='__main__':print(json.dumps(main()))
