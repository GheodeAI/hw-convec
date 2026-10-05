"""Small file helpers."""
import json


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str)+'\n')
