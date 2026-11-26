"""Exact 96 answer-free input records and 16 official-source PNGs."""
from pathlib import Path
from input_bundle import TARGET as INPUT_DIRECTORY, pin, verify

TARGET=str(INPUT_DIRECTORY)
CURRENT_INPUT_PIN={'bytes':2290,'SHA256':'1b2f60743e0eb6a389c151b96a68bb40e937e8826d290e9693395baeba661e57'}

def check_bundle(root=Path(TARGET)):
    rows,_,manifest=verify(root,CURRENT_INPUT_PIN)
    return rows,pin(manifest)
