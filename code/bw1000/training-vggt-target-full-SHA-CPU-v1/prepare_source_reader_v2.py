"""Use the source host selected by the actual frozen original relay caller."""
from pathlib import Path
import ast,hashlib,json
D=Path(__file__).resolve().parent;old=D.parent/'training-vggt-original-public-model-v1'
driver=(old/'package/transfer_once.py').read_bytes();assert hashlib.sha256(driver).hexdigest()=='48cec916996ad849a2e50707eb520e5220c9c5ad12f085718fc5fec4fb25b817' and b"'a100-26'" in driver
s=(D/'read_original_source_terminal_once.py').read_text(encoding='utf8').replace("'a100-27'","'a100-26'").replace('source-terminal','source-terminal-v2');ast.parse(s)
(D/'read_original_source_terminal_v2.py').write_bytes(s.encode())
record=dict(original_source_terminal_read_rc=1,original_source_terminal_read_host='a100-27',original_source_terminal_stderr_SHA256=hashlib.sha256((D/'source-terminal.stderr.private.txt').read_bytes()).hexdigest(),cause='First metadata reader selectedA100-27 from the current VLM context;the frozen original VGGT relay actually selectedA100-26.',actual_original_relay_caller_SHA256=hashlib.sha256(driver).hexdigest(),corrected_reader_host='a100-26',original_transfer_replayed=False,weight_file_bytes_read_by_metadata_reader=0)
(D/'source-reader-host-correction.json').write_bytes((json.dumps(record,indent=2)+'\n').encode())
print(json.dumps(dict(corrected_host_from_original_caller=True,original_transfer_replayed=False)))
