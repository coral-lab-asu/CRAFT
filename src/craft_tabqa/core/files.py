"""Reading and writing the file formats the pipeline uses: JSONL and pickle."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Union

PathLike = Union[str, Path]


def read_jsonl(path: PathLike) -> Iterator[Dict[str, Any]]:
    """Yield one parsed object per non-empty line of a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: PathLike, records: Iterable[Dict[str, Any]]) -> None:
    """Write an iterable of dicts as JSONL, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True))
            f.write("\n")


def save_pickle(path: PathLike, obj: Any) -> None:
    """Pickle an object to disk, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: PathLike) -> Any:
    """Load a pickled object from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
