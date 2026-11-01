"""CPU tests of explicit presentation compatibility; no model or HTTP imports."""
import unittest
from content_json import parse_final_object


class FinalObjectTests(unittest.TestCase):
    def test_bare_and_fence_same_semantics_distinct_format(self):
        a,am=parse_final_object(' {"total":3} \n')
        b,bm=parse_final_object('```json\n{"total":3}\n```')
        self.assertEqual(a,b);self.assertTrue(am['strict_bare_JSON_format'])
        self.assertFalse(bm['strict_bare_JSON_format'])

    def test_entire_fence_crlf_and_whitespace(self):
        v,m=parse_final_object(' \n```json \r\n{"a":{"sum":-4}}\r\n``` \n')
        self.assertEqual(v,{'a':{'sum':-4}})

    def test_prose_multiple_objects_and_extra_fences_rejected(self):
        for text in ['answer: {"x":1}','{"x":1} done','{"x":1}{"x":2}',
                     'before\n```json\n{}\n```','```json\n{}\n```\nafter',
                     '```json\n{}\n```\n```json\n{}\n```','```python\n{}\n```',
                     '```\n{}\n```','```json\n{}\n']:
            with self.subTest(text=text),self.assertRaises(ValueError):parse_final_object(text)

    def test_duplicate_keys_rejected_recursively(self):
        for text in ['{"total":1,"total":2}','{"a":{"sum":1,"sum":2}}',
                     '```json\n{"a":1,"a":1}\n```']:
            with self.subTest(text=text),self.assertRaises(ValueError):parse_final_object(text)

    def test_nonfinite_or_nonobject_rejected(self):
        for text in ['{"x":NaN}','{"x":Infinity}','{"x":-Infinity}','{"x":1e999}',
                     '[{"x":1}]','null','true','"{}"','']:
            with self.subTest(text=text),self.assertRaises(ValueError):parse_final_object(text)

    def test_parser_does_not_coerce_integer_booleans_or_strings(self):
        v,_=parse_final_object('{"bool":true,"text":"1","integer":1}')
        self.assertIs(type(v['bool']),bool);self.assertIs(type(v['text']),str)
        self.assertIs(type(v['integer']),int)


if __name__=='__main__':unittest.main(verbosity=2)
