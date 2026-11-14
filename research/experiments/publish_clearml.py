"""Upload a completed benchmark to ClearML Cloud using configured credentials."""
import argparse
import json
from pathlib import Path

from bench import digest, read


def publish(folder, data, project, include_dataset=False):
    folder, data = Path(folder), Path(data)
    metadata = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
    scores = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
    manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    if metadata['dataset'] != manifest:
        raise ValueError('Dataset manifest does not match this run')
    gold, predictions = read(data / 'test.jsonl'), read(folder / 'predictions.jsonl')
    if len(gold) != len(predictions) or digest(gold) != manifest['hashes']['test']:
        raise ValueError('Incomplete run or modified test')
    for row, prediction in zip(gold, predictions):
        if digest(row) != prediction['row_hash']:
            raise ValueError('Predictions do not match the supplied test')
    artifacts = [folder / name for name in ('run.json', 'metrics.json', 'predictions.jsonl', 'protocol.py')]
    if (folder / 'errors.jsonl').is_file():
        artifacts.append(folder / 'errors.jsonl')
    artifacts.append(data / 'manifest.json')
    if include_dataset:
        for split in ('train', 'dev', 'test'):
            path = data / f'{split}.jsonl'
            if digest(read(path)) != manifest['hashes'][split]:
                raise ValueError('Modified dataset split')
            artifacts.append(path)
    for path in artifacts:
        if not path.is_file():
            raise ValueError(f'Missing artifact: {path}')

    from clearml import Task
    identity = digest({'metadata': metadata, 'predictions': predictions})[:16]
    task = Task.init(project_name=project, task_name=f'{folder.name}-{identity}',
                     task_type=Task.TaskTypes.testing, reuse_last_task_id=False,
                     output_uri=True, auto_connect_arg_parser=False,
                     auto_connect_frameworks=False, auto_connect_streams=False,
                     auto_resource_monitoring=False)
    try:
        task.connect(metadata['config'], name='experiment')
        task.connect(manifest['hashes'], name='dataset_hashes')
        task.connect({'signature': metadata['signature'], 'identity': identity,
                      'resolved_revision': metadata.get('resolved_revision', ''),
                      'protocol_hash': metadata.get('protocol_hash', '')}, name='provenance')
        logger = task.get_logger()
        for name, value in scores.items():
            if isinstance(value, (int, float)):
                logger.report_single_value(name=name, value=value)
        for category, values in scores['per_category'].items():
            for name, value in values.items():
                logger.report_scalar(title='category_' + category, series=name, value=value, iteration=0)
        if 'context_cue_slice' in scores:
            for name, value in scores['context_cue_slice'].items():
                logger.report_single_value(name='context_cue/' + name, value=value)
        table = [['row_hash', 'mcId', 'gold', 'prediction', 'valid', 'seconds']]
        if include_dataset:
            table[0].append('description')
        for row, pred in zip(gold, predictions):
            values = [pred['row_hash'], row['mcId'], json.dumps(row['labels']),
                      json.dumps(pred['labels']), pred['valid'], pred['seconds']]
            if include_dataset:
                values.append(row['description'])
            table.append(values)
        logger.report_table(title='Predictions', series='test', iteration=0, table_plot=table)
        for path in artifacts:
            success = task.upload_artifact(name=path.name, artifact_object=str(path.resolve()), wait_on_upload=True)
            if success is False:
                raise RuntimeError(f'Artifact upload failed: {path.name}')
        print(task.get_output_log_web_page())
    except BaseException:
        task.mark_failed(status_reason='Result export failed')
        raise
    finally:
        task.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--project', default='avito-microcategories')
    parser.add_argument('--include-dataset', action='store_true')
    args = parser.parse_args()
    publish(args.run, args.data, args.project, args.include_dataset)


if __name__ == '__main__':
    main()
