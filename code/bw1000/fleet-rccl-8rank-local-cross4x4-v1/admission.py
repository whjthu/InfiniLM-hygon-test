"""Fresh allocation and active-step gates; no unknown step is ignored."""
import re


def need(test, reason):
    if not test:
        raise ValueError(reason)


def fields(line):
    return dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)',line))


def parent_gate(line, parent, uid, expected_nodes, expand):
    row=fields(line)
    owner=re.fullmatch(r'[^()]+\((\d+)\)',row.get('UserId',''))
    tres=dict(piece.split('=',1) for piece in row.get('AllocTRES','').split(',') if '=' in piece)
    need(row.get('JobId')==parent and row.get('JobState')=='RUNNING' and row.get('NumNodes')=='40'
         and owner is not None and int(owner[1])==uid and tres.get('node')=='40'
         and tres.get('gres/dcu')=='320' and set(expand(row.get('NodeList','')))==set(expected_nodes),
         'FRESH_OWNED_PARENT40_DCU320_MEMBERSHIP')
    return row


def step_gate(lines,parent,allowed_nodes,allowed_labels,expand):
    observed=[]
    for line in lines:
        row=fields(line);step=row.get('StepId','')
        need(step.startswith(parent+'.'),'STEP_PARENT')
        suffix=step[len(parent)+1:]
        if suffix not in ('batch','extern'):
            need(suffix.isdecimal() and allowed_nodes and row.get('Name') in allowed_labels
                 and row.get('NodeList') not in (None,'(null)')
                 and set(expand(row['NodeList'])) == set(allowed_nodes),
                 'CURRENT_NUMBERED_OR_UNKNOWN_STEP_CONFLICT')
        observed.append(row)
    need(bool(observed),'CURRENT_STEP_INVENTORY_MISSING')
    return observed
