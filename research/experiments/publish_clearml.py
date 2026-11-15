"""Upload a completed benchmark to ClearML Cloud using configured credentials."""
import argparse
import json
from pathlib import Path

from bench import digest, read


def evaluated_gold(folder, gold, predictions, scores):
    """Validate explicit screening results; incomplete ordinary runs stay rejected."""
    path = Path(folder) / 'evaluation.json'
    if not path.is_file():
        if len(gold) != len(predictions):
            raise ValueError('Incomplete run or modified test')
        return gold
    evaluation = json.loads(path.read_text(encoding='utf-8'))
    count = len(predictions)
    prefix = gold[:count]
    status = evaluation.get('status')
    stopped = status == 'early_stopped'
    if (not 0 < count <= len(gold) or status not in ('completed', 'early_stopped')
            or stopped != (count < len(gold))
            or evaluation.get('planned_rows') != len(gold)
            or evaluation.get('evaluated_rows') != count
            or evaluation.get('planned_test_hash') != digest(gold)
            or evaluation.get('evaluated_test_hash') != digest(prefix)
            or scores.get('evaluation_status') != status
            or scores.get('evaluated_rows') != count or scores.get('rows') != count
            or scores.get('planned_rows') != len(gold)
            or scores.get('early_stopped') != stopped
            or (stopped and not evaluation.get('stop_reason'))):
        raise ValueError('Invalid screening result or modified test')
    return prefix


def publish(folder, data, project, include_dataset=False):
    folder, data = Path(folder), Path(data)
    metadata = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
    scores = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
    manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    if set(manifest['hashes']) - {'train', 'dev', 'test'}:
        raise ValueError('Unsupported dataset split in manifest')
    if metadata['dataset'] != manifest:
        raise ValueError('Dataset manifest does not match this run')
    gold, predictions = read(data / 'test.jsonl'), read(folder / 'predictions.jsonl')
    if digest(gold) != manifest['hashes']['test']:
        raise ValueError('Incomplete run or modified test')
    gold = evaluated_gold(folder, gold, predictions, scores)
    for row, prediction in zip(gold, predictions):
        if digest(row) != prediction['row_hash']:
            raise ValueError('Predictions do not match the supplied test')
    artifacts = [folder / name for name in ('run.json', 'metrics.json', 'predictions.jsonl', 'protocol.py')]
    if (folder / 'errors.jsonl').is_file():
        artifacts.append(folder / 'errors.jsonl')
    for name in ('bench.py', 'drafts.json', 'prompt.txt', 'model_catalog.json', 'notebook.ipynb',
                 'evaluation.json', 'response_contract.py', 'response_schema.json', 'reuse.json'):
        if (folder / name).is_file():
            artifacts.append(folder / name)
    artifacts.append(data / 'manifest.json')
    if include_dataset:
        for split in manifest['hashes']:
            path = data / f'{split}.jsonl'
            if digest(read(path)) != manifest['hashes'][split]:
                raise ValueError('Modified dataset split')
            artifacts.append(path)
    for path in artifacts:
        if not path.is_file():
            raise ValueError(f'Missing artifact: {path}')

    from clearml import Task
    identity = digest({'metadata': metadata, 'predictions': predictions})[:16]
    name = folder.name
    if manifest.get('kind') == 'api-zero-shot':
        name = f"api-{metadata['dataset_name']}-{metadata['config']['name']}"
    if scores.get('early_stopped'):
        name += '-early-stopped'
    task = Task.init(project_name=project, task_name=f'{name}-{identity}',
                     task_type=Task.TaskTypes.testing, reuse_last_task_id=False,
                     output_uri=True, auto_connect_arg_parser=False,
                     auto_connect_frameworks=False, auto_connect_streams=False,
                     auto_resource_monitoring=False)
    try:
        task.connect(metadata['config'], name='experiment')
        if scores.get('early_stopped'):
            task.add_tags(['early-stopped', 'screening-only'])
        if (folder / 'evaluation.json').is_file():
            task.connect(json.loads((folder / 'evaluation.json').read_text(encoding='utf-8')), name='evaluation')
        task.connect(manifest['hashes'], name='dataset_hashes')
        if 'prompt' in metadata:
            task.connect({'text': metadata['prompt'], 'hash': digest(metadata['prompt'])}, name='system_prompt')
        task.connect({'signature': metadata['signature'], 'identity': identity,
                      'resolved_revision': metadata.get('resolved_revision', ''),
                      'protocol_hash': metadata.get('protocol_hash', '')}, name='provenance')
        logger = task.get_logger()
        for name, value in scores.items():
            if isinstance(value, (int, float)):
                metric_name = 'screening/' + name if scores.get('early_stopped') else name
                logger.report_single_value(name=metric_name, value=value)
        for category, values in scores['per_category'].items():
            for name, value in values.items():
                title = ('screening/' if scores.get('early_stopped') else '') + 'category_' + category
                logger.report_scalar(title=title, series=name, value=value, iteration=0)
        if 'context_cue_slice' in scores:
            for name, value in scores['context_cue_slice'].items():
                prefix = 'screening/' if scores.get('early_stopped') else ''
                logger.report_single_value(name=prefix + 'context_cue/' + name, value=value)
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
        result = {'task_id': task.id, 'url': task.get_output_log_web_page()}
        print(result['url'])
        return result
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
