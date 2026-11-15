"""Package the API notebook, its runtime and test datasets for Colab."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
FILES = [
    'research/experiments/api_colab.ipynb',
    'research/experiments/api_bench.py',
    'research/experiments/bench.py',
    'research/experiments/response_contract.py',
    'research/experiments/publish_clearml.py',
    'research/experiments/api_models.json',
    'research/experiments/prompts/classify_v2.txt',
    'research/experiments/prompts/classify_v2_seo_off.txt',
    'research/experiments/prompts/classify_v3.txt',
    'research/experiments/prompts/classify_v3_seo_on.txt',
    'research/experiments/requirements-api.txt',
    'research/experiments/API_BENCHMARK.md',
    'research/experiments/build_api_bundle.py',
    'research/experiments/test_api_bench.py',
    'research/experiments/test_bench.py',
    'research/experiments/test_clearml_export.py',
    'research/experiments/export_hard_test.py',
    'research/experiments/test_export_hard_test.py',
    'data/raw/public-test-dataset.parquet',
    'data/test/test_gold.parquet',
    'data/test/test_v7.parquet',
    'data/hard/hard_v0.xlsx',
]


def build(out):
    # Explicit allowlist: never package credentials, environments or run outputs.
    contents = {name: (ROOT / name).read_bytes() for name in FILES}
    manifest = {'files': {name: hashlib.sha256(body).hexdigest() for name, body in contents.items()}}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, body in contents.items():
            archive.writestr(name, body)
        archive.writestr('bundle_manifest.json', json.dumps(manifest, indent=2))
    print(out.resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path,
                        default=ROOT / 'research/api-runs/delivery/caila-baseline-bundle.zip')
    build(parser.parse_args().out)
