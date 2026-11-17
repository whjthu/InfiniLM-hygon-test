"""Resolve only the seven saved exact official metadata JSONs; no network/install."""
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
from urllib.parse import urlsplit
from packaging.utils import parse_wheel_filename,canonicalize_name
from packaging.tags import Tag
from packaging.specifiers import SpecifierSet
from packaging.requirements import Requirement
from packaging.markers import default_environment

D=Path(__file__).resolve().parent
G=D.parents[2]


def main():
    folder=G/'local-only/workspace-matplotlib-overlay-v1-metadata-authorized-access'
    receipt=json.loads((folder/'receipt.json').read_bytes())
    assert receipt['passed'] and receipt['metadata_GETs_completed']==7 and receipt['total_metadata_bytes']<=1048576
    baseline={name:md.version(name) for name in ('numpy','Pillow','packaging','python-dateutil','tzdata')}
    available={canonicalize_name(name):version for name,version in baseline.items()}
    versions={name:row['version'] for name,row in receipt['rows'].items()};available.update(versions)
    env=default_environment();env['extra']=''
    assert env['platform_system']=='Windows' and env['python_version']=='3.12'
    priorities=[Tag('cp312','cp312','win_amd64'),Tag('cp312','abi3','win_amd64'),Tag('py3','none','any')]
    selected={}
    for name,version in sorted(versions.items()):
        raw=(folder/(name+'-'+version+'.json')).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==receipt['rows'][name]['SHA256']
        document=json.loads(raw)
        assert SpecifierSet(document['info']['requires_python']).contains('3.12.14',prereleases=True)
        candidates=[]
        for item in document['urls']:
            if item['packagetype']!='bdist_wheel' or item['yanked']:continue
            wn,wv,build,tags=parse_wheel_filename(item['filename'])
            ranks=[i for i,tag in enumerate(priorities) if tag in tags]
            if ranks and str(wv)==version and canonicalize_name(str(wn))==name:
                candidates.append((min(ranks),item,tags))
        assert candidates,name+' no allowed exact wheel'
        rank=min(item[0] for item in candidates);chosen=[item for item in candidates if item[0]==rank]
        assert len(chosen)==1
        _,item,tags=chosen[0];url=urlsplit(item['url'])
        assert url.scheme=='https' and url.hostname=='files.pythonhosted.org' and url.path.endswith('/'+item['filename'])
        assert item['size']>0 and len(item['digests']['sha256'])==64
        active=[]
        for text in document['info']['requires_dist'] or []:
            requirement=Requirement(text)
            if requirement.marker is not None and not requirement.marker.evaluate(env):continue
            dep=canonicalize_name(requirement.name)
            assert dep in available and requirement.specifier.contains(available[dep],prereleases=True),(name,text)
            active.append(dict(requirement=text,resolved_version=available[dep],
                               source='overlay' if dep in versions else 'existing_fixed_runtime'))
        selected[name]=dict(version=version,filename=item['filename'],url=item['url'],bytes=item['size'],
            SHA256=item['digests']['sha256'],requires_python=document['info']['requires_python'],
            wheel_requires_python=item['requires_python'],tags=sorted(str(tag) for tag in tags),
            requires_dist=document['info']['requires_dist'] or [],active_dependencies=active,metadata_source=receipt['rows'][name])
    baseline_dependencies=[]
    for name in baseline:
        for text in md.distribution(name).requires or []:
            requirement=Requirement(text)
            if requirement.marker is not None and not requirement.marker.evaluate(env):continue
            dep=canonicalize_name(requirement.name)
            assert dep in available and requirement.specifier.contains(available[dep],prereleases=True),(name,text)
            baseline_dependencies.append(dict(parent=name,requirement=text,resolved_version=available[dep]))
    lock=dict(schema='official_pypi_exact_windows312_wheel_lock_v1',python_version='3.12.14',platform='win_amd64',
        versions=versions,wheels=selected,existing_runtime=baseline,existing_runtime_active_dependencies=baseline_dependencies,
        official_metadata_source='https://pypi.org/pypi/<name>/<exact-version>/json',metadata_bytes=receipt['total_metadata_bytes'],
        wheel_download_bytes=sum(row['bytes'] for row in selected.values()),
        source_receipt=dict(path=(folder/'receipt.json').relative_to(G).as_posix(),bytes=(folder/'receipt.json').stat().st_size,
                            SHA256=hashlib.sha256((folder/'receipt.json').read_bytes()).hexdigest()),installations=0)
    (D/'WHEEL_LOCK.json').write_text(json.dumps(lock,indent=2)+'\n',encoding='utf8',newline='\n')
    lines=['--only-binary=:all:']+[name+'=='+row['version']+' --hash=sha256:'+row['SHA256'] for name,row in sorted(selected.items())]
    (D/'requirements.lock').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
    print(json.dumps(dict(wheels={name:{key:row[key] for key in ('filename','bytes','SHA256','requires_python')}
                                  for name,row in selected.items()},compressed_bytes=lock['wheel_download_bytes'],baseline=baseline)))


if __name__=='__main__':main()
