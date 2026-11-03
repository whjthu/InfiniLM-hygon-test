"""Local stdlib-only consumer of actual public constructor/lifespan methods.

FastAPI/AsyncLLMEngine constructor leaves are recording leaves, not installed
imports. The supplied NativeEngineBridge is its real existing stdlib class.
This proves the public seam and cleanup, never model/HTTP qualification.
"""
import argparse
import ast
import asyncio
from contextlib import asynccontextmanager
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
from types import SimpleNamespace

from patch_server import patched


def check(source: Path, bridge_path: Path):
    original = source.read_bytes()
    code = patched(original)
    cls = next(n for n in ast.parse(code).body
               if isinstance(n, ast.ClassDef) and n.name == "InferenceServer")
    # Keep the deployed methods verbatim. Routes are outside this leaf boundary.
    cls.body = [n for n in cls.body if getattr(n, "name", None)
                in {"__init__", "_create_app"}]
    routes = []
    def register(self, app):
        routes.append(app)
    class App:
        def __init__(self, *, lifespan): self.lifespan = lifespan
    events = []
    class RecordedEngine:
        start_error = None
        stop_error = None
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            events.append(("construct_default", kwargs))
        def start(self):
            events.append("start")
            if self.start_error: raise self.start_error
        def stop(self):
            events.append("stop")
            if self.stop_error: raise self.stop_error
    log = logging.getLogger("provider_CPU")
    log.addHandler(logging.NullHandler())
    log.propagate = False
    ns = dict(asynccontextmanager=asynccontextmanager, FastAPI=App,
              AsyncLLMEngine=RecordedEngine, logger=log, os=__import__("os"),
              Optional=__import__("typing").Optional, KVTransferConfig=object,
              TOOL_CALL_FORMATS={})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 str(source), "exec"), ns)
    Server = ns["InferenceServer"]
    Server._register_routes = register
    spec = importlib.util.spec_from_file_location("existing_native_bridge", bridge_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results = []
    def passed(name): results.append(dict(case=name, passed=True))

    async def exercise():
        server = Server("/actual-model", max_cache_len=32768,
                        tensor_parallel_size=8, pipeline_parallel_size=4)
        app = server._create_app()
        async with app.lifespan(app):
            assert events[-1] == "start"
            assert server.engine.kwargs["pipeline_parallel_size"] == 4
            assert server.engine.kwargs["max_cache_len"] == 32768
        assert events[-1] == "stop" and app in routes
        baseline_kwargs = server.engine.kwargs
        passed("default_constructor_and_exact_settings_preserved")

        bridge = module.NativeEngineBridge(object(), 32768, SimpleNamespace())
        factory_calls = []
        def factory(**kwargs):
            # Native construction/load is external to this CPU leaf; do not
            # invent READY. This supplies the real resident queue object only.
            factory_calls.append(kwargs)
            return bridge
        server = Server("/actual-model", max_cache_len=32768,
                        tensor_parallel_size=8, pipeline_parallel_size=4,
                        engine_factory=factory)
        before = len(events)
        app = server._create_app()
        assert not factory_calls  # only lifespan owns engine construction
        async with app.lifespan(app):
            assert server.engine is bridge and bridge.is_healthy()
            assert factory_calls == [baseline_kwargs]
            assert len(events) == before  # never enters generic model factory
        assert bridge._stopped and bridge.take(timeout=.01) is None
        passed("real_resident_bridge_through_public_lifespan_no_generic_engine")

        for phase in ["start", "yield"]:
            engine = RecordedEngine()
            primary = RuntimeError("primary-" + phase)
            cleanup = RuntimeError("cleanup")
            engine.stop_error = cleanup
            if phase == "start": engine.start_error = primary
            server = Server("/actual-model", engine_factory=lambda **kw: engine)
            app = server._create_app()
            try:
                async with app.lifespan(app):
                    raise primary
            except RuntimeError as actual:
                assert actual is primary and events[-1] == "stop"
            else: raise AssertionError("primary lost")
            passed(phase + "_failure_cleanup_preserves_primary")

        engine = RecordedEngine()
        engine.stop_error = RuntimeError("close failed")
        app = Server("/actual-model", engine_factory=lambda **kw: engine)._create_app()
        try:
            async with app.lifespan(app): pass
        except RuntimeError as actual: assert actual is engine.stop_error
        else: raise AssertionError("cleanup falsely accepted")
        passed("normal_shutdown_failure_visible")
        try: Server("/actual-model", engine_factory="not-a-trusted-callable")
        except TypeError: pass
        else: raise AssertionError("invalid provider accepted")
        passed("noncallable_provider_rejected")

    asyncio.run(exercise())
    return dict(passed=True, cases=results, Model_READY=False, GPU=False,
                Native_construct_load_executed=False, HTTP_import_executed=False,
                actual_public_method_AST_consumed=True,
                recording_leaves=["FastAPI constructor", "default AsyncLLMEngine constructor",
                                  "route registration"],
                original_server_SHA256=hashlib.sha256(original).hexdigest(),
                patched_server_SHA256=hashlib.sha256(code).hexdigest(),
                real_bridge_source_SHA256=hashlib.sha256(bridge_path.read_bytes()).hexdigest())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("bridge", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    result = check(a.source, a.bridge)
    a.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
