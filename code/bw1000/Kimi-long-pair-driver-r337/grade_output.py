"""Use the pinned existing XTML parser and exact independently derived arguments."""
import json
from frozen_agent.agent_protocol import prepare_agent_request,collect_message
from frozen_agent.tool_parsers import ToolCallParser


def grade_output(fixture,ids,finish,decode):
    if finish!='eos' or not ids or ids[-1] not in fixture['eos']:
        raise ValueError('Natural EOS required for this scoped tool task; cap remains a cap')
    data,schemas,choice=prepare_agent_request(fixture['body'],'kimi_k3')
    options=data['chat_template_kwargs']
    thinking=options.get('thinking',options.get('enable_thinking',True))
    parser=ToolCallParser(schemas,thinking=thinking,choice=choice,
                          parallel=data.get('parallel_tool_calls',True),parser='kimi_k3')
    text=decode(ids[:-1])
    message=collect_message(parser.feed(text,final=True))
    calls=message.get('tool_calls',[])
    if len(calls)!=1 or calls[0]['function']['name']!='calculate':
        raise ValueError('Exactly one real calculate call required')
    args=json.loads(calls[0]['function']['arguments'])
    expected=fixture['expected_calculate_arguments']
    if set(args)!=set(expected) or any(type(args[k]) is not int or args[k]!=v for k,v in expected.items()):
        raise ValueError('Actual tool arguments differ from independently derived integers')
    return dict(passed=True,scope='Single tool intent, not tool execution or a full benchmark',
                natural_EOS=True,tool='calculate',arguments=args,parser='pinned existing XTML')
