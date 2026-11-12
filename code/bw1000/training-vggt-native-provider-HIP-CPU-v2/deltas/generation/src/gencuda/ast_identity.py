"""Stable source identity across Python versions that add empty AST fields."""
import ast
import hashlib
import json


def portable_ast_sha256(tree):
    """Ignore only absent-versus-empty type parameters, preserving nonempty ones."""
    def normalize(value):
        if isinstance(value, ast.AST):
            fields = list(ast.iter_fields(value))
            names = {name for name, _item in fields}
            # Python3.12 adds type_params=[] to function and class nodes. Test
            # synthetic nonempty fields on older interpreters as well.
            if hasattr(value, 'type_params') and 'type_params' not in names:
                fields.append(('type_params', value.type_params))
            return [type(value).__name__, [[name, normalize(item)] for name, item in fields
                    if not (name == 'type_params' and item == [])]]
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return [type(value).__name__, repr(value)]

    raw = json.dumps(normalize(tree), separators=(',', ':'), ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()
