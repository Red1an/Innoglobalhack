"""Dataset cache identity shared by the terminal and notebook runners."""

import hashlib
import json
from pathlib import Path


GOLD_OVERLAP_POLICY = 'gold-normalized-overlap-v1'


def uses_gold_overlap(test_path):
    return Path(test_path).name == 'test_gold.parquet'


def dataset_id(train_path, test_path, assume_source):
    hashes = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
              for name, path in (('train', train_path), ('test', test_path))}
    identity = [hashes, assume_source]
    if uses_gold_overlap(test_path):
        identity.append(GOLD_OVERLAP_POLICY)
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
