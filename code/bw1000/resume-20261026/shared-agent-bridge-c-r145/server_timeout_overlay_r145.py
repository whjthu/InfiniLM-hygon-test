"""Exact, isolated patch to 7ceb's existing timeout branches; no new HTTP stack."""
import ast
import hashlib
import pathlib

ORIGINAL_SERVER_SHA256 = '16c522a8e33b694eff469dc55250c7d03e63b6c02cbe0a8272ddc23d764f1c40'

def patch_timeout_routes(raw):
    assert hashlib.sha256(raw).hexdigest() == ORIGINAL_SERVER_SHA256
    text = raw.decode('utf-8')
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'InferenceServer')
    patches = []
    comparison = ast.dump(ast.parse('token_output.finish_reason == FinishReason.TIMEOUT', mode='eval').body)
    for method in ['_chat', '_stream_chat']:
        fn = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == method)
        nodes = [n for n in ast.walk(fn) if isinstance(n, ast.If) and ast.dump(n.test) == comparison]
        assert len(nodes) == 1 and isinstance(nodes[0].body[-1], ast.Break)
        node = nodes[0]
        indent = ' ' * node.col_offset
        body = [
            'if token_output.finish_reason == FinishReason.TIMEOUT:',
            '    _abort_reason = FinishReason.TIMEOUT',
            '    logger.warning(f"Request {request_id} exceeded its deadline")',
            '    if req and not req.is_finished():',
            '        self.engine.add_aborted_req(req, FinishReason.TIMEOUT)',
        ]
        if method == '_chat':
            body += [
                '    return JSONResponse(',
                '        content={"error": {"message": "Request deadline exceeded", "type": "timeout_error"}},',
                '        status_code=504,',
                '    )',
            ]
        else:
            body += [
                '    yield sse_json({"error": {"message": "Request deadline exceeded", "type": "timeout_error"}})',
                '    yield sse_json(chunk_json(request_id, finish_reason="timeout", model=self.model_id))',
                '    break',
            ]
        newline = '\r\n' if lines[node.lineno-1].endswith('\r\n') else '\n'
        replacement = ''.join(indent + line + newline for line in body)
        patches.append((node.lineno-1, node.end_lineno, replacement))
    for start,end,replacement in sorted(patches,reverse=True):
        lines[start:end] = [replacement]
    result = ''.join(lines).encode('utf-8')
    ast.parse(result.decode('utf-8'))
    return result

def write_isolated_overlay(original, destination):
    original = pathlib.Path(original)
    destination = pathlib.Path(destination)
    assert original.resolve() != destination.resolve() and not destination.exists()
    result = patch_timeout_routes(original.read_bytes())
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('xb') as out:
        out.write(result)
    return hashlib.sha256(result).hexdigest()
