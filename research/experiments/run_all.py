"""Run the benchmark from a terminal with the same data presets, folders and ClearML export as colab.ipynb."""
import argparse
from pathlib import Path
import subprocess
import sys

from dataset_protocol import dataset_id

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[1]
# Keep in sync with the presets cell of colab.ipynb.
PRESETS = {
    'old': (ROOT / 'data/raw/public-train-annotated.jsonl', ROOT / 'data/raw/public-test-dataset.parquet'),
    'processed': (ROOT / 'data/processed/train_clean.parquet', ROOT / 'data/processed/test_clean.parquet'),
    'gold': (ROOT / 'data/processed/train_clean.parquet', ROOT / 'data/test/test_gold.parquet'),
    # Organizers' original train with the team's gold test. Compared later against a new train on the same test.
    'raw-gold': (ROOT / 'data/raw/public-train-annotated.jsonl', ROOT / 'data/test/test_gold.parquet'),
}
ASSUME_SOURCE = 101


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=list(PRESETS), default='gold')
    parser.add_argument('--participant', default='member2')
    parser.add_argument('--run-group', default='baseline')
    parser.add_argument('--models', nargs='+', default=['char-lr', 't-lite-q8', 'avibe-q8', 'rumodernbert'])
    parser.add_argument('--project', default='avito-microcategories')
    parser.add_argument('--no-upload', action='store_true', help='Do not send finished runs to ClearML')
    parser.add_argument('--no-dataset-upload', action='store_true', help='Send metrics and predictions without split texts')
    args = parser.parse_args()

    train, test = PRESETS[args.dataset]
    data_id = dataset_id(train, test, ASSUME_SOURCE)
    data = CODE / 'datasets' / f'{args.dataset}-{data_id}'
    if not data.exists():
        subprocess.run([sys.executable, str(CODE / 'bench.py'), 'prepare', '--train', str(train), '--test', str(test),
                        '--out', str(data), '--assume-source', str(ASSUME_SOURCE)], check=True)
    runs = CODE / 'runs' / args.participant / f'{args.dataset}-{data_id}-{args.run_group}'
    print(f'Data {data}\nRuns {runs}', flush=True)
    for number, model in enumerate(args.models, 1):
        print(f'\n=== {number}/{len(args.models)} {model} ===', flush=True)
        subprocess.run([sys.executable, '-u', str(CODE / 'bench.py'), 'run', '--model', model,
                        '--data', str(data), '--out', str(runs / model)], check=True, cwd=CODE)
        if not args.no_upload:
            command = [sys.executable, str(CODE / 'publish_clearml.py'), '--run', str(runs / model),
                       '--data', str(data), '--project', args.project]
            if not args.no_dataset_upload:
                command.append('--include-dataset')
            subprocess.run(command, check=True, cwd=CODE)
    subprocess.run([sys.executable, str(CODE / 'bench.py'), 'summarize', '--runs', str(runs),
                    '--out', str(runs / 'results.csv')], check=True)


if __name__ == '__main__':
    main()
