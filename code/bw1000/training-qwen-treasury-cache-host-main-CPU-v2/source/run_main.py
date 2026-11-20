"""Enter the actual candidate argparse/__main__, HOST optin only; no Engine mocks."""
from pathlib import Path
import runpy
import sys

if __name__=="__main__":
    if "--host-main-preflight" not in sys.argv[1:]:
        raise ValueError("HOST main optin required; no GPU entry is authorized")
    main=Path(__file__).resolve().parent/"native_generation.py"
    sys.argv[0]=str(main)
    runpy.run_path(str(main),run_name="__main__")
