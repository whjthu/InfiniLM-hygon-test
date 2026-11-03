"""Produce an isolated public-server patch from the exact deployed source.

No installed tree is changed; the existing engine remains the default.
"""
import difflib
import hashlib
from pathlib import Path


def patched(source: bytes) -> bytes:
    text = source.decode("utf-8")
    old = '        tool_call_parser: str = "none",\n'
    assert text.count(old) == 1
    text = text.replace(old, old + "        engine_factory=None,\n")
    old = '        self.engine: AsyncLLMEngine = None\n'
    assert text.count(old) == 1
    text = text.replace(old, '''        # Trusted application configuration, never a request-body/import string.
        # A Native provider must own real construct/load/finish before returning
        # a ready engine. This callback itself grants no model qualification.
        if engine_factory is not None and not callable(engine_factory):
            raise TypeError("engine_factory must be callable or None")
        self.engine_factory = engine_factory
        self.engine: AsyncLLMEngine = None
''')
    old = "            self.engine = AsyncLLMEngine(\n"
    assert text.count(old) == 1
    text = text.replace(old, '''            factory = (AsyncLLMEngine if self.engine_factory is None
                       else self.engine_factory)
            self.engine = factory(
''')
    old = '''            self.engine.start()
            logger.info(f"Engine initialized with model at {self.model_path}")
            logger.info(f"  enable_graph: {self.enable_graph}")
            yield
            self.engine.stop()
'''
    assert text.count(old) == 1
    text = text.replace(old, '''            try:
                self.engine.start()
                logger.info(f"Engine initialized with model at {self.model_path}")
                logger.info(f"  enable_graph: {self.enable_graph}")
                yield
            except BaseException:
                # Cleanup must not replace the startup/request primary error.
                try:
                    self.engine.stop()
                except BaseException:
                    logger.exception("Engine cleanup failed after primary error")
                raise
            else:
                self.engine.stop()
''')
    compile(text, "infinilm/server/inference_server.py", "exec")
    return text.encode("utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("expected_SHA256")
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    original = args.source.read_bytes()
    assert hashlib.sha256(original).hexdigest() == args.expected_SHA256
    result = patched(original)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "inference_server.py").write_bytes(result)
    patch = "".join(difflib.unified_diff(
        original.decode().splitlines(True), result.decode().splitlines(True),
        fromfile="a/python/infinilm/server/inference_server.py",
        tofile="b/python/infinilm/server/inference_server.py"))
    (args.output_directory / "public-engine-provider.patch").write_text(patch, encoding="utf-8", newline="\n")
