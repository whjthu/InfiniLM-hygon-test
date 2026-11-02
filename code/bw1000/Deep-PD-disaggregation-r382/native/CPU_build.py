import concurrent.futures, hashlib, json, os, pathlib, re, shutil, subprocess, time, traceback

D = pathlib.Path(__file__).resolve().parent
OUT = pathlib.Path('/refactor/candidates/Deep-PD-native-r382-cpu-v1')
C = json.loads((D/'BUILD_CONTRACT.json').read_bytes())
MIRROR = OUT/'project-mirror'
START = time.monotonic()
RESULT = {'completed': False, 'passed': False, 'CPU_only': True, 'GPU_initialized': False,
          'Native_dlopen': False, 'Model_loaded': False, 'GPU_TUs_compiled': 0, 'commands': []}

def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def run(argv, label, timeout=180):
    z = subprocess.run([str(x) for x in argv], capture_output=True, text=True, timeout=timeout)
    (OUT/(label+'.stdout')).write_text(z.stdout)
    (OUT/(label+'.stderr')).write_text(z.stderr)
    if z.returncode: raise RuntimeError(f'{label} rc={z.returncode}: {z.stderr[-3000:]}')
    return z.stdout
def canonical(s):
    prefix=C['private_parent_mirror']
    return s[len(prefix):] if s.startswith(prefix+'/') else s.removeprefix('/refactor')
def private(s): return str(MIRROR/canonical(s).lstrip('/'))
def dependency_paths(text):
    return text.replace('\\\n',' ').split(':',1)[1].split()
def rewrite_arg(a):
    if a.startswith('-I/refactor/'): return '-I'+private(a[2:])
    return a
def main():
    mf=D/'SOURCE_MANIFEST.json'
    assert sha(mf)==os.environ['DEEP_PD_NATIVE_MANIFEST_SHA256']
    manifest=json.loads(mf.read_bytes())
    assert all(sha(D/name)==expected for name,expected in manifest['files_SHA256'].items())
    assert not OUT.exists(), 'Immutable output already exists; never resume or overwrite'
    OUT.mkdir(parents=True)
    for path, expected in C['objects'].items(): assert sha(path)==expected, path
    assert sha(C['parent_library'])==C['parent_library_SHA256']
    shutil.copytree(C['private_parent_mirror'], MIRROR)
    modified={p.name:p for p in (D/'headers').glob('*.hpp')}
    replacements={x['object']:x for x in C['guard_compiled']}
    source_map={x['object']:x for x in C['source_mapping'] if x.get('source')}
    commands={x[-1]:x for x in C['classification_commands']}
    candidates=[]
    # Re-run actual project preprocessing for each HOST source; GPU objects are
    # preserved byte-for-byte. This finds every consumer of changed friend headers.
    for index,obj in enumerate(C['objects']):
        if obj in replacements:
            q=replacements[obj];source=q['source'];argv=q['compile_argv'];argv=argv[:argv.index('-MD')]+['-M',source]
        elif obj in source_map:
            source=source_map[obj]['source'];argv=commands[source]
        else: continue
        deps=dependency_paths(run(argv,f'classify-{index}'))
        affected=any(pathlib.PurePosixPath(x).name in modified for x in deps)
        special=obj.endswith('/Deep-native-context-guard-r180d/0.o') or obj.endswith('/Deep-native-context-guard-r180d/2.o')
        if not affected and not special: continue
        candidates.append((index,obj,source,argv[:-2],deps))
        # Private aliases preserve original quoted-include search semantics.
        for item in [source]+deps:
            if not item.startswith('/refactor/'): continue
            dst=pathlib.Path(private(item));src=pathlib.Path(item)
            if dst.exists():
                # Existing current guard aliases take precedence (notably the
                # qualified NativeSparseCacheConfig header).
                continue
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    # Install the same declaration-only friend addition at all aliases. There
    # are no new class data fields/vtables; all affected HOST units rebuild.
    patched=[]
    for p in MIRROR.rglob('*.hpp'):
        if p.name in modified:
            original=(D/'original'/p.name).read_bytes()
            assert p.read_bytes()==original, f'Current class header differs: {p}'
            p.write_bytes(modified[p.name].read_bytes());patched.append(str(p))
    jobs=[]
    for index,obj,source,flags,deps in candidates:
        target_source=pathlib.Path(private(source))
        if obj.endswith('/Deep-native-context-guard-r180d/0.o'):target_source.write_bytes((D/'native_sparse_model_pd_r382.cpp').read_bytes())
        if obj.endswith('/Deep-native-context-guard-r180d/2.o'):target_source.write_bytes((D/'modern_native_engine_probe_pd_r382.cpp').read_bytes())
        output=OUT/f'host-{index}.o'
        argv=[rewrite_arg(x) for x in flags]
        argv[1:1]=['-I'+str(D),'-I'+str(D/'headers')]
        argv+=['-MD','-MF',str(output.with_suffix('.d')),'-c',str(target_source),'-o',str(output)]
        jobs.append((obj,argv,output,target_source))
    base=C['guard_compiled'][0]['compile_argv'];base=base[:base.index('-MD')]
    for name in ['deep_pd_archive_r382.cpp','deep_pd_codec_r382.cpp']:
        output=OUT/(name+'.o');argv=[rewrite_arg(x) for x in base]
        argv[1:1]=['-I'+str(D),'-I'+str(D/'headers')]
        argv+=['-MD','-MF',str(output.with_suffix('.d')),'-c',str(D/name),'-o',str(output)]
        jobs.append((None,argv,output,D/name))
    def compile_one(item):
        old,argv,obj,source=item;run(argv,obj.stem);deps=dependency_paths(obj.with_suffix('.d').read_text())
        escaped=[x for x in deps if x.startswith('/refactor/') and not(x.startswith(str(MIRROR)+'/') or x.startswith(str(D)+'/'))]
        assert not escaped, escaped
        return {'replaces':old,'source':str(source),'source_SHA256':sha(source),'object':str(obj),'object_SHA256':sha(obj),'compile_argv':argv,
                'dependencies_SHA256':{x:sha(x) for x in deps}}
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        compiled=list(pool.map(compile_one,jobs))
    replace={x['replaces']:x['object'] for x in compiled if x['replaces']}
    link=[replace.get(x,x) for x in C['link_argv']]
    # Existing link argv includes -o oldDSO and all original49 objects.
    library=OUT/'libDeep_PD_native_r382.so';link[link.index('-o')+1]=str(library)
    link += [x['object'] for x in compiled if not x['replaces']]
    run(link,'link')
    old_exports={line.split()[-1] for line in run(['nm','-D','--defined-only',C['parent_library']],'old-symbols').splitlines()}
    new_exports={line.split()[-1] for line in run(['nm','-D','--defined-only',str(library)],'new-symbols').splitlines()}
    assert old_exports<=new_exports, sorted(old_exports-new_exports)
    required=[line.split('(')[0].split()[-1] for line in (D/'deep_pd_abi_r382.h').read_text().splitlines() if line.startswith(('int deep_','uint64_t deep_','const char *deep_'))]
    assert set(required)<=new_exports, sorted(set(required)-new_exports)
    ldd=run(['ldd',str(library)],'ldd');assert 'not found' not in ldd
    test=OUT/'CPU_archive_test'
    argv=[rewrite_arg(x) for x in base if not x.startswith(('-L','-l','-Wl,'))]
    argv[1:1]=['-I'+str(D)]
    argv += [str(D/'CPU_archive_test.cpp'),str(D/'deep_pd_archive_r382.cpp'),'-o',str(test)]
    run(argv,'compile-CPU-archive');test_result=json.loads(run([str(test)],'CPU-archive-test',60));assert test_result['passed']
    all_objects={replace.get(x,x):sha(replace.get(x,x)) for x in C['objects']}
    all_objects.update({x['object']:x['object_SHA256'] for x in compiled if not x['replaces']})
    RESULT.update(completed=True,passed=True,source_manifest_SHA256=sha(mf),actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],HOST_TUs_compiled=len(compiled),compiled=compiled,patched_headers=patched,
        library=str(library),library_SHA256=sha(library),all_objects_SHA256=all_objects,link_argv=link,
        all_original_dynamic_exports_retained=True,ABI_symbols=required,CPU_archive_test=test_result,
        scope='HOST compile/link and CPU archive protocol only; no Runtime, GPU state, handoff or Model qualification')

if OUT.exists():
    print(json.dumps({'completed':True,'passed':False,'error':'Immutable output exists; no files modified'}))
    raise SystemExit(1)
try: main()
except Exception as e: RESULT.update(completed=True,error=str(e),traceback=traceback.format_exc())
finally:
    RESULT['seconds']=time.monotonic()-START
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'CPU-build.json').write_text(json.dumps(RESULT,indent=2)+'\n')
    print(json.dumps({'completed':RESULT['completed'],'passed':RESULT['passed'],'seconds':RESULT['seconds'],'error':RESULT.get('error')}))
    if not RESULT['passed']: raise SystemExit(1)
