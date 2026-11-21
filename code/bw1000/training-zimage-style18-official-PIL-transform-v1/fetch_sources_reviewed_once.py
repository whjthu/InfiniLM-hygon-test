"""New reviewed source-only operation after known zero-response sandbox failure."""
import hashlib
import json
from pathlib import Path
import fetch_sources_once as original


def main():
    prior = original.OUT / 'result.json'
    raw = prior.read_bytes()
    if len(raw)!=903 or hashlib.sha256(raw).hexdigest()!='54fc865bc96eeff08cdd8fc580fb3a145d4e6e2623d06ec52fd2c11a0b3dc14e':
        raise ValueError('Original zero-response failure evidence differs')
    failure=json.loads(raw)
    if failure['passed'] is not False or failure['download_bytes']!=0 or failure['actual_GETs']!=1:
        raise ValueError('Original failed source operation is not the known zero-response scope')
    original.OUT=original.G/'local-only/training-zimage-style18-official-PIL-transform-v1-source-reviewed-original'
    original.REQUEST='c7d3a9b946f74937bd6987749f4f785f'
    return original.main()


if __name__=='__main__':raise SystemExit(main())
