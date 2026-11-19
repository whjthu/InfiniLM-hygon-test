"""Only this fixed allocated new CPU step may run the one range worker."""
from common import D,P,RUN,load,need,sha

def main():
    from verify_package import verify
    verify(D)
    context=load(RUN/'node-context.json');need(sha((D/'SOURCE_FILES.json').read_bytes())==context['source_manifest_SHA256'],'INSTALLED_SOURCE_CONTEXT')
    from http_range import main as worker
    return worker(context)

if __name__=='__main__':raise SystemExit(main())
