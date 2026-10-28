"""Generic API prompt constraints; does not filter tools or mutate history."""
import copy,json
def request_constraints(data, tools):
    choice = data.get('_agent_choice', data.get('tool_choice', 'auto'))
    if choice is None:
        choice = 'auto'
    if isinstance(choice, dict):
        choice = choice['function']['name']
    parallel = data.get('parallel_tool_calls', True)
    if type(parallel) is not bool:
        raise ValueError('parallel_tool_calls must be boolean')
    if choice == 'none':
        return '[Request tool-call constraints]\nTools are disabled for this assistant turn. Do not invoke any tool. Respond without a tool call.'
    if not tools:
        return ''
    rules = []
    if not parallel:
        rules.append('In this assistant turn, issue at most one real tool call. After issuing that call, end this assistant turn and wait for its tool result before deciding or calling any next tool. Do not combine later steps or additional invocations in this turn.')
    if choice == 'required':
        rules.append('This request requires at least one real tool call in this assistant turn. Do not substitute a text answer for the required tool call.')
    elif choice != 'auto':
        available = [t['function']['name'] for t in tools]
        if choice not in available:
            raise ValueError('Requested tool is unavailable after protocol filtering')
        rules.append('This request requires a real call to the function ' + json.dumps(choice, ensure_ascii=False) + ' in this assistant turn, following its provided schema. Do not substitute a text answer for the required call.')
    return '[Request tool-call constraints]\n' + '\n'.join(rules) if rules else ''

def copied_messages(messages,data,kwargs):
    result=copy.deepcopy(messages)
    rules=request_constraints(data,kwargs.get('tools'))
    if rules:
        if not result or result[0].get('role')!='system':
            result.insert(0,{'role':'system','content':''})
        content=result[0].get('content','')
        if not isinstance(content,str):
            raise ValueError('System content must be text after protocol preparation')
        result[0]['content']=content+'\n\n'+rules
    return result

def make_encode_chat(tokenizer):
    def encode_chat(messages,data,add_generation_prompt,kwargs):
        original=copy.deepcopy(messages)
        original_kwargs=copy.deepcopy(kwargs)
        copied=copied_messages(messages,data,kwargs)
        prompt=tokenizer.apply_chat_template(copied,tokenize=False,
            add_generation_prompt=add_generation_prompt,**copy.deepcopy(kwargs))
        ids=list(tokenizer.encode(prompt,add_special_tokens=False))
        assert messages==original and kwargs==original_kwargs
        return prompt,ids
    return encode_chat
