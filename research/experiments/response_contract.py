"""Versioned response validation; transformations never inspect gold labels."""
import json
import re
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError

VERSION = 'pydantic-normalized-v1'


class LabelResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    labels: list[Annotated[StrictInt, Field(ge=101, le=106)]]


def parse_labels(content: object, source: int) -> tuple[list[int], list[str], str | None]:
    """Unwrap a whole code fence, validate IDs, then remove source and repeats."""
    if not isinstance(content, str):
        return [], [], 'missing_content'
    issues = []
    match = re.fullmatch(r'\s*```(?:json)?\s*(.*?)\s*```\s*', content, re.S | re.I)
    if match:
        content = match.group(1)
        issues.append('markdown_fence_removed')
    try:
        value = json.loads(content)
    except (ValueError, TypeError):
        return [], issues, 'invalid_json_or_refusal'
    try:
        result = LabelResponse.model_validate(value)
    except ValidationError:
        error = 'missing_labels' if isinstance(value, dict) and 'labels' not in value else 'schema_validation'
        return [], issues, error
    if source in result.labels:
        issues.append('source_category_removed')
    if len(result.labels) != len(set(result.labels)):
        issues.append('duplicate_ids_removed')
    return sorted(set(result.labels) - {source}), issues, None


def response_schema(source: int = 101) -> dict:
    schema = LabelResponse.model_json_schema()
    schema['properties']['labels']['items']['enum'] = [x for x in range(101, 107) if x != source]
    schema['properties']['labels']['uniqueItems'] = True
    return schema
