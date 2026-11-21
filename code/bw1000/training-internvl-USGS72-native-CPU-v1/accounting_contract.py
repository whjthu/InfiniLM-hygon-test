"""Strict original allocation/step accounting; absence is never fabricated count1."""
import re


def select_original(lines,parent,label,node):
    rows=[]
    for line in lines:
        if not line.strip():continue
        fields=[part.strip() for part in line.split('|')]
        if len(fields)!=7:raise ValueError('MALFORMED_ACCOUNTING_ROW_NOT_DROPPED')
        rows.append(fields)
    parents=[r for r in rows if r[0]==parent]
    if len(parents)!=1:raise ValueError('UNIQUE_CURRENT_PARENT_ACCOUNTING_REQUIRED')
    exact=[r for r in rows if r[1]==label]
    if len(exact)>1:raise ValueError('DUPLICATE_ORIGINAL_LABEL')
    if exact and (re.fullmatch(re.escape(parent)+r'\.\d+',exact[0][0]) is None or exact[0][5]!=node):
        raise ValueError('ORIGINAL_LABEL_STEP_OR_NODE_BINDING_DIFFERS')
    return dict(parent=parents[0],step=exact[0] if exact else None)


def actual_step_count(row,parent,node):
    if row is None:return None
    if len(row)!=7 or re.fullmatch(re.escape(parent)+r'\.\d+',row[0]) is None or row[5]!=node:
        raise ValueError('ACTUAL_NUMBERED_STEP_NOT_BOUND')
    return 1
