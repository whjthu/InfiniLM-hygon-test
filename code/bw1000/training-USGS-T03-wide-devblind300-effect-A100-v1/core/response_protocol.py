"""Preregistered gold-free extraction only; never searches numeric candidates or calls a judge."""
from decimal import Decimal,InvalidOperation,localcontext
import re
NUMBER=r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)'
DEPTH_KINDS={'lookup_depth','depth_change'}
LABEL_KINDS={'maximum_depth','maximum_reported_number'}


def strict_format(text):
    # Exact original optional-prefix canonicalizer strictFINAL branch, unchanged grammar.
    return isinstance(text,str) and '\n' not in text and '\r' not in text and bool(re.fullmatch(r'FINAL: (-?\d+\.\d{2}|E\d{2}(?: E\d{2})*)',text.strip()))


def extract_content(text,schema):
    result=dict(valid=False,normalized=None,reason='invalid_text',strict_FINAL_format=strict_format(text))
    if not isinstance(text,str) or not 0<len(text)<=16384:return result
    lines=[s.strip() for s in text.splitlines() if s.strip()]
    if not lines:return result
    span=lines[-1];markers=text.count('FINAL:')
    if markers:
        if markers!=1 or not span.startswith('FINAL: '):result['reason']='ambiguous_or_misplaced_FINAL';return result
        span=span[len('FINAL: '):].strip()
    kind=schema['kind']
    if kind in LABEL_KINDS:
        if not re.fullmatch(r'[Ee]\d{2}(?:[ ,]+[Ee]\d{2})*',span):result['reason']='invalid_label_span';return result
        labels=re.split(r'[ ,]+',span.upper());known=schema['event_ids']
        if len(labels)!=len(set(labels)) or not set(labels)<=set(known):result['reason']='duplicate_or_unknown_labels';return result
        normalized=' '.join(sorted(labels,key=known.index))
    else:
        if kind not in DEPTH_KINDS|{'lookup_reported_magnitude'}:result['reason']='unknown_task_kind';return result
        # No scientific notation, locale/thousands separator, explanatory prefix or candidate search.
        match=re.fullmatch('('+NUMBER+r')(?:\s*(km|kilometers?|kilometres?|m|meters?|metres?))?',span,re.IGNORECASE)
        if not match or len(match.group(1))>64:result['reason']='invalid_numeric_span';return result
        unit=match.group(2).lower() if match.group(2) else None
        if kind=='lookup_reported_magnitude' and unit is not None:result['reason']='magnitude_has_physical_unit';return result
        try:
            with localcontext() as ctx:
                ctx.prec=128;value=Decimal(match.group(1))
                if unit in {'m','meter','meters','metre','metres'}:value*=Decimal('0.001')
                if not value.is_finite() or value!=value.quantize(Decimal('.01')):result['reason']='would_require_rounding';return result
                if value==0:value=Decimal('0')
                normalized=format(value,'.2f')
        except InvalidOperation:result['reason']='invalid_decimal';return result
    result.update(valid=True,normalized=normalized,reason='parsed_unique_final_span');return result


def content_prediction(prediction,schema):
    # Future sealed-output scorer may pass this normalized row to the unchanged frozen score_once.score.
    parsed=extract_content(prediction.get('text'),schema);value=dict(prediction)
    value.update(text=parsed['normalized'] if parsed['valid'] else 'INVALID_CONTENT_FORMAT',
        content_parse_valid=parsed['valid'],content_parse_reason=parsed['reason'],strict_FINAL_format=parsed['strict_FINAL_format'])
    return value
