import io
import time
import unittest

from direct import fetch


class Response(io.BytesIO):
    status = 206

    def __init__(self, body=b"data"):
        super().__init__(body)
        self.headers = {"Content-Range": "bytes 4-7/8", "Content-Length": "4"}
        self.body_read = False

    def read1(self, length=-1):
        self.body_read = True
        return super().read1(length)

    def geturl(self):
        return "https://cdn-lfs-cn-1.modelscope.cn/opaque"


class Opener:
    def __init__(self, response):
        self.response = response

    def open(self, request, *, timeout):
        if request.get_header("Range") != "bytes=4-7" or timeout <= 0:
            raise AssertionError("Unexpected publisher range")
        return self.response


class RangeTest(unittest.TestCase):
    def read(self, response):
        return fetch(Opener(response), {"bytes": 8, "relative_path": "text_encoder/source.safetensors"},
                     4, 4, time.time()+5)

    def test_exact_range(self):
        data, wall = self.read(Response())
        self.assertEqual(data, b"data")
        self.assertGreaterEqual(wall, 0)

    def test_full_response_rejected_before_any_body(self):
        response = Response()
        response.status = 200
        with self.assertRaises(ValueError):
            self.read(response)
        self.assertFalse(response.body_read)

    def test_wrong_absolute_range_is_rejected(self):
        response = Response()
        response.headers["Content-Range"] = "bytes 0-3/8"
        with self.assertRaises(ValueError):
            self.read(response)
        self.assertFalse(response.body_read)

    def test_extra_and_short_data_are_rejected(self):
        for raw in (b"data!", b"dat"):
            with self.subTest(size=len(raw)), self.assertRaises(ValueError):
                self.read(Response(raw))


if __name__ == "__main__":
    unittest.main()
