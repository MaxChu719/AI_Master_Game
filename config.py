"""
Central config loader. Import CFG anywhere in the project:

    from config import CFG
    hp = CFG["fighter"]["hp"]
"""
import json
import pathlib

_path = pathlib.Path(__file__).parent / "config.json"
with open(_path) as _f:
    CFG = json.load(_f)
