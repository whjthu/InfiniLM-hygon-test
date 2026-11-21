"""Test finite Graph64 authentication binding with ordinary subprocess pipes."""

import io
import subprocess
import sys
import unittest

from graph64_authenticated_exec import authenticated


def local(argv):
    return subprocess.Popen([sys.executable,*argv[1:]],
               stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)


class AuthenticatedTests(unittest.TestCase):
    def test_round_trip_uses_one_child_and_disallows_retry(self):
        children = []

        def start(argv):
            child = local(argv)
            children.append(child)
            return child

        with authenticated(start) as send:
            out,error = io.BytesIO(),io.BytesIO()
            result = send(b"print('one program')",out,error,10)
            self.assertEqual((result.returncode,out.getvalue(),error.getvalue()),(0,b'one program\r\n' if sys.platform=='win32' else b'one program\n',b''))
            with self.assertRaisesRegex(ValueError,'repeated'):
                send(b'print(2)',out,error,10)
        self.assertEqual(len(children),1)

    def test_closed_before_hello_never_enters_reservation_scope(self):
        def failed(_):
            return local(['ignored','-c','raise SystemExit(255)'])
        with self.assertRaisesRegex(RuntimeError,'before reservation'):
            with authenticated(failed):
                self.fail('unverified channel entered reservation')

    def test_close_without_submission_terminates_waiting_child(self):
        children = []

        def start(argv):
            child = local(argv)
            children.append(child)
            return child

        with authenticated(start):
            self.assertIsNone(children[0].poll())
        self.assertIsNotNone(children[0].poll())


if __name__ == '__main__':
    unittest.main()
