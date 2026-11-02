"""Semantic final-answer object decoding; never parses tool args or reasoning.

Original bare-JSON format qualification remains separate. Only one entire json
code fence is accepted as an explicitly recorded presentation compatibility.
"""
import json
import math
import re


def _unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate JSON key')
        result[key]=value
    return result


def _finite_float(value):
    number=float(value)
    if not math.isfinite(number):raise ValueError('Non-finite JSON number')
    return number


def _reject_constant(value):
    raise ValueError('Non-finite JSON constant')


def parse_final_object(content):
    if not isinstance(content,str):raise ValueError('Final content must be text')
    text=content.strip();format_name='bare_JSON'
    if text.startswith('```'):
        match=re.fullmatch(r'```json[ \t]*\r?\n([\s\S]*?)\r?\n```',text)
        if not match:raise ValueError('Require one entire json fence without surrounding prose')
        text=match.group(1);format_name='whole_json_fence'
    result=json.loads(text,object_pairs_hook=_unique_object,parse_float=_finite_float,parse_constant=_reject_constant)
    if not isinstance(result,dict):raise ValueError('Final answer must be one complete JSON object')
    return result,{'presentation':format_name,'semantic_whole_object_parsed':True,
                   'strict_bare_JSON_format':format_name=='bare_JSON'}
