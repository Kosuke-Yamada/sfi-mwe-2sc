import json
from pathlib import Path


def _to_serializable(item):
    return {k: str(v) if isinstance(v, Path) else v for k, v in item.items()}


def write_jsonl(items, file):
    with open(file, "w") as fo:
        for item in items:
            print(json.dumps(_to_serializable(item), ensure_ascii=False), file=fo)


def write_json(items, file):
    with open(file, "w") as fo:
        json.dump(_to_serializable(items), fo, ensure_ascii=False, indent=2)


def read_jsonl(file):
    with open(file) as f:
        for line in f:
            yield json.loads(line)


def read_json(file):
    with open(file) as f:
        return json.load(f)
