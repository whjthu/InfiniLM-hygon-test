
import datetime,hashlib,json,re,shlex,subprocess
from pathlib import Path
R=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
W=R/"candidates/Deep-C4-history-bank-alias-CPU-v1"
A=W/"artifacts"
def raw(path,maximum=1500000):
    b=Path(path).read_bytes()
    assert len(b)<maximum
    return b
def sha(path):
    return hashlib.sha256(raw(path,16000000)).hexdigest()
def check(path,pin):
    b=raw(path)
    assert hashlib.sha256(b).hexdigest()==pin
    return json.loads(b)
def host(path):
    if path.startswith("/build/"):return W/Path(path).relative_to("/build")
    if path.startswith("/refactor/"):return R/Path(path).relative_to("/refactor")
    raise ValueError("unexpected artifact root")
def deps(path):
    return set(shlex.split(raw(path).decode().replace("\\\n"," ").split(":",1)[1]))
receipt=check(A/"CPU-build.json","74b40092a766ffd72aa7a0b89f2668a37ed557afdfa7481e44d037421025c35d")
assert receipt["completed"] and receipt["publication_commit"]=="cb77f5d02f3c0fb5277f70e6a6edc071a55b9522"
assert receipt["claim_SHA256"]=="150c5fb37cae0cf74144b2057e8d9ad3491c4ccb8b71bfe12400c1da2171d56d"
assert not receipt["GPU_initialized"] and not receipt["Native_dlopen"]
assert receipt["actual_step"]=="1087656.70" and receipt["reused"]==49
rows=subprocess.check_output(["sacct","-n","-P","-j","1087656","--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw"],timeout=12,text=True)
match=[x.split("|") for x in rows.splitlines() if x.startswith("1087656.70|")]
assert len(match)==1 and match[0][1:4]==["i03-deep-c4-history-alias-cpu-v1","COMPLETED","0:0"]
base=check(R/"candidates/Deep-HC-fullpre-Native-r377-cpu-v1/CPU-build.json","837b0e824e79a33a460754eedc943a8c11786acfa856dde5ca2c438351ab5389")
parent=check(R/"results/deep-attention-inner-CPU-r9193a.json","2dcb7a947f64ebcbfd2d316dd1d5db6e06e61d3a412ea6411a0c627237ba01a3")
original=parent["new_compiled_units"][2]["compile_argv"]
source=original[original.index("-c")+1]
old_object=original[original.index("-o")+1]
old_md=original[original.index("-MF")+1]
assert sha(host(old_md))=="6488227565c27bc78cbd97a30d3144ae5c28fa9a897b2e0993f3074161d6b1e4"
assert sha(host(source))=="2611335a7d57ee12411f3d7d5f076bf138f6db17d8607fec184e8d209bff8c8c"
new_source=A/Path(source).name
new_object=A/Path(old_object).name
new_md=A/(Path(source).name+".d")
new_m=A/(Path(source).name+".M")
assert sha(new_source)=="ce08de9a8a3d61d3db2703a0c7f0d3e79b34cc59963cdda094c73dd9f6fab18e"
assert sha(new_object)=="bbc6f78463ce65b8a7e8ec56def9f7c672955863a2dd1b797e9821ccf432b4db"
assert sha(A/"history_bank_alias.hpp")=="af00e26217bb63347ce36725db118b790564400498aaad9b579069648a1c4169"
expected=deps(host(old_md))-{source}
expected.update(["/build/artifacts/"+Path(source).name,"/build/artifacts/history_bank_alias.hpp"])
assert deps(new_m)==deps(new_md)==expected
assert sha(new_m)==receipt["replaced"][0]["M_SHA256"] and sha(new_md)==receipt["replaced"][0]["MD_SHA256"]
objects=base["all50_objects_SHA256"]
assert len(objects)==50 and len(receipt["replaced"])==1
for name,pin in objects.items():
    assert sha(host(name))==pin
lib=A/"libDeep_c4_history_bank_alias_v1.so"
assert sha(lib)==receipt["library_SHA256"]=="5e6793ad220e2bb6501fadb3eae70f0b5270565759f2c6907997809ef1be3537"
expected_link=list(base["link_argv"])
expected_link[expected_link.index(old_object)]="/build/artifacts/"+Path(old_object).name
expected_link[expected_link.index(base["library"])]="/build/artifacts/"+lib.name
links=[x["argv"] for x in receipt["commands"] if x["label"]=="link"]
assert links==[expected_link]
commands={x["label"]:x["argv"] for x in receipt["commands"]}
expected_compile=[*original[:1],"-iquote",str(Path(source).parent),*original[1:]]
for old,new in ((source,"/build/artifacts/"+Path(source).name),(old_object,"/build/artifacts/"+Path(old_object).name),(old_md,"/build/artifacts/"+Path(source).name+".d")):
    assert expected_compile.count(old)==1
    expected_compile[expected_compile.index(old)]=new
assert commands["compile"]==expected_compile
def exports(p):
    out=subprocess.check_output(["nm","-D","--defined-only",str(p)],timeout=10,text=True)
    return {s[-1] for row in out.splitlines() if len(s:=row.split())>=3 and s[-2] in ("T","D","B","R")}
def needed(p):
    out=subprocess.check_output(["readelf","-d",str(p)],timeout=10,text=True)
    return re.findall(r"\(NEEDED\).*?\[(.*?)\]",out)
oldlib=host(base["library"])
assert sha(oldlib)==base["library_SHA256"]
a,b=exports(oldlib),exports(lib)
assert len(a)==337 and a<=b and b-a=={"deep_c4_history_bank_alias_set_v1","deep_c4_history_bank_alias_stats_v1"}
assert needed(oldlib)==needed(lib)
assert receipt["original_deadline"]-receipt["original_epoch"]==300
assert receipt["elapsed_s"]<300
print(json.dumps(dict(state="TERMINAL_ONE_HOST_ALIAS_BUILD_INDEPENDENTLY_VERIFIED",
    at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    accounting=match[0],CPU_receipt_SHA256="74b40092a766ffd72aa7a0b89f2668a37ed557afdfa7481e44d037421025c35d",
    library=receipt["library"],library_SHA256=receipt["library_SHA256"],new_object_SHA256=sha(new_object),
    source_SHA256=sha(new_source),header_SHA256=sha(A/"history_bank_alias.hpp"),
    original_objects_rechecked=50,reused_objects=49,rebuilt_HOST_units=1,
    M_MD_exact_dependency_set=True,dependency_count=len(expected),
    compiler_command_matches_original_plus_source_object_MD_private_quote=True,
    link_command_matches_parent_plus_one_object=True,
    parent_strong_exports=len(a),new_strong_exports=sorted(b-a),dynamic_NEEDED_unchanged=True,
    elapsed_s=receipt["elapsed_s"],original_epoch=receipt["original_epoch"],original_deadline=receipt["original_deadline"],
    Native_dlopen=False,GPU_initialized=False,weights_read=False,GPU_GO=False,performance_qualified=False)))
