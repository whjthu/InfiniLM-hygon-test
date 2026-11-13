"""Parse and compare passive loader names without confusing aliases with identities."""
from pathlib import Path

def parse_ldd(text):
 names={}
 for line in text.splitlines():
  row=line.split()
  if not row or row[0].startswith('linux-vdso'):continue
  if '=>' in row:
   if len(row)!=4 or row[1]!='=>' or not row[2].startswith('/'):raise ValueError('unresolved or invalid ldd dependency')
   name,path=row[0],row[2]
  else:
   if len(row)!=2 or not row[0].startswith('/'):raise ValueError('unexpected passive loader row')
   name,path=Path(row[0]).name,row[0]
  if name in names:raise ValueError('duplicate dependency name')
  names[name]=path
 if not names:raise ValueError('empty dependency map')
 return names

def classify(expected_names,expected_libraries,observed):
 rows=[]
 for name in sorted(set(expected_names)|set(observed)):
  actual=observed.get(name);expected=expected_names.get(name);pin=expected_libraries.get(expected)
  same=bool(pin and actual and actual['bytes']==pin['bytes'] and actual['SHA256']==pin['SHA256'])
  rows.append(dict(name=name,expected_realpath=expected,observed_realpath=actual['realpath'] if actual else None,byte_identity_matches=same,realpath_matches=bool(actual and actual['realpath']==expected)))
 return dict(name_sets_equal=set(expected_names)==set(observed),all_name_byte_identities_match=all(x['byte_identity_matches'] for x in rows),all_realpaths_match=all(x['realpath_matches'] for x in rows),rows=rows)
