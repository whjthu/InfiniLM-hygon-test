"""Portable AST identity must retain all nonempty semantic fields."""
import ast
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gencuda.ast_identity import portable_ast_sha256


class ASTIdentityTest(unittest.TestCase):
    def test_absent_and_empty_type_parameters_are_equal(self):
        absent = ast.parse('def compute(x):\n    return x + 1\n')
        for node in ast.walk(absent):
            if hasattr(node, 'type_params'):
                delattr(node, 'type_params')
        empty = copy.deepcopy(absent)
        empty.body[0].type_params = []
        self.assertEqual(portable_ast_sha256(absent), portable_ast_sha256(empty))

    def test_nonempty_type_parameters_and_math_changes_are_detected(self):
        original = ast.parse('def compute(x):\n    return x + 1\n')
        generic = copy.deepcopy(original)
        generic.body[0].type_params = [ast.Name(id='T', ctx=ast.Load())]
        self.assertNotEqual(portable_ast_sha256(original), portable_ast_sha256(generic))
        for changed in ['def compute(x):\n    return x - 1\n',
                        'def compute(x):\n    return x + 2\n']:
            self.assertNotEqual(portable_ast_sha256(original), portable_ast_sha256(ast.parse(changed)))


if __name__ == '__main__':
    unittest.main()
