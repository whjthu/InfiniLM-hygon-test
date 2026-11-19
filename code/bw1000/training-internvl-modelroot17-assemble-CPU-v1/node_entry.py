"""New fixed CPU observer only; frozen original3PASS remains mandatory."""
from common import D,load,need,sha,canonical

def main():
    from verify_package import verify
    v=verify(D);need(v['CPU_GO'] is v['assembly_GO'] is v['READ_GO'] is True,'ROOT_NEW_ASSEMBLY_ADOPTION')
    context=load(D.parent/'node-context.json');snapshot=load(D.parent/'upstream-snapshot.json',32768);need(sha(canonical(snapshot))==context['dependency_snapshot_SHA256'],'EXACT_SNAPSHOT')
    from assemble_worker import main as worker
    return worker(context,snapshot)

if __name__=='__main__':raise SystemExit(main())
