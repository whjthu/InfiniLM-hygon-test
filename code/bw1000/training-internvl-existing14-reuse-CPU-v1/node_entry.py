"""Only this numeric Slurm step enters the once14 payload audit."""
from common import D,P,RUN,load,need,sha

def main():
    from verify_package import verify
    verify(D)
    context=load(RUN/'node-context.json')
    need(sha((D/'SOURCE_FILES.json').read_bytes())==context['source_manifest_SHA256'],'INSTALLED_MANIFEST_CONTEXT')
    from audit_existing import main as audit
    return audit(context,P,load(D/'CLAIM.json'))
if __name__=='__main__':raise SystemExit(main())
