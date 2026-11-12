from pathlib import Path
import hashlib,os,sys
p=Path(PLAN['source_file']);assert p.is_file() and not p.is_symlink() and p.stat().st_size==PLAN['bytes'];raw=p.read_bytes();assert len(raw)==PLAN['bytes'] and hashlib.sha256(raw).hexdigest()==PLAN['SHA256'];sys.stdout.buffer.write(raw);sys.stdout.buffer.flush()
