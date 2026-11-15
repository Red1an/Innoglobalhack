"""Compare guide_v2 and guide_v3 train on the same hackathon test sets."""

import argparse
import json
from pathlib import Path

from build_guide_v3 import ROOT, TEST_OUT, TRAIN_OUT, TRAIN_PARENT
from compare_guide_v2 import compare as compare_versions

OUTPUT = ROOT / 'output/guide_v3_comparison.json'
VERSION_PATHS = {
    'guide_v2': TRAIN_PARENT / 'train.parquet',
    'guide_v3': TRAIN_OUT / 'train.parquet',
}


def compare():
    return compare_versions(TEST_OUT, VERSION_PATHS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = compare()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(args.out)
    print(args.out)
