"""Only a freshly held real compute step enters one bounded receiver."""
from common import D,load,need,sha,member

def main():
    from verify_package import verify
    verify(D);context=load(D.parent/'node-context.json');member(context)
    need(sha((D/'SOURCE_FILES.json').read_bytes())==context['source_manifest_SHA256'],'EXACT_INSTALLED_SOURCE')
    from endpoint import main as endpoint
    return endpoint('receiver',context)

if __name__=='__main__':raise SystemExit(main())
