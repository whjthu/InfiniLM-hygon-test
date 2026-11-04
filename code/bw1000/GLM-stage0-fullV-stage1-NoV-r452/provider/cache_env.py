"""One startup-only scalar for actual inside+worker agreement; no devices."""
from pathlib import Path
import argparse,hashlib,json,os
from public_glm_native.cache_policy_r452 import for_config
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--config-sha',required=True);p.add_argument('--stage',type=int,required=True);a=p.parse_args(argv)
 raw=Path(a.config).read_bytes();assert hashlib.sha256(raw).hexdigest()==a.config_sha;policy=for_config(json.loads(raw),a.stage)
 if 'GLM_CACHE_POLICY_R452'in os.environ and os.environ['GLM_CACHE_POLICY_R452']!=policy.policy:raise ValueError('Shell/config cache identity mismatch')
 print(policy.noV_flag);return 0
if __name__=='__main__':raise SystemExit(main())
