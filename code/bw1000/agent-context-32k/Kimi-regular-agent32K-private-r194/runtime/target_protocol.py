"""Resident, real Native179 scoped32K greedy AR worker for the unchanged 7ceb API.

One admitted request at a time. Node0 coordinates HTTP, all four nodes execute
every original Native/PP command. No draft model, supplied answer, or fake token.
"""
import json
import os
import pathlib
import queue
import threading
import time


def startup(lib):
    from target_binding import startup as check
    return check(lib)


def preweight_startup(g):
    """Called on all nodes after Native bind, before any Native model/weights.

    The owner entry rechecks actual DSO maps and then performs a four-node
    barrier before create/load. A failed server import cannot waste a load.
    """
    if g["node"] != 0:return
    from long_codec_r179 import KimiLongCodec
    from native_engine_bridge_r143 import existing_server_app
    from long_input_r179 import KimiLongBridge, validate_prompt_ids
    from infinilm.llm import FinishReason
    import infinilm.server.inference_server as upstream_server
    import hashlib
    codec = KimiLongCodec(g["model"])
    def encode_chat(messages, data, add_generation_prompt, kwargs):
        prompt, ids = codec.encode_chat(messages, kwargs, add_generation_prompt)
        validate_prompt_ids(ids, g["vocab"])
        return prompt, ids
    engine = KimiLongBridge(codec, 32768, FinishReason, encode_chat=encode_chat,
                  decode_tokens=lambda ids: codec.decode(ids), eos_token_ids={163585,163586},
                  default_max_tokens=4096, max_pending=8)
    _, app = existing_server_app(engine, str(g["model"]), "kimi_k3", host="127.0.0.1", port=43844)
    g["_agent_prepared"] = {"codec": codec, "engine": engine, "app": app, "FinishReason": FinishReason}
    server_path=pathlib.Path(upstream_server.__file__)
    g["report"]["agent_preweight_import"] = {
        "passed": True, "server_module": str(server_path),
        "server_SHA256": hashlib.sha256(server_path.read_bytes()).hexdigest(),
        "tokenizer": codec.identity, "Native_model_constructed": False,
        "weight_payload_opened": False, "HTTP_started": False,
        "source": "7ceb routes with resident engine lifespan injection only",
    }


def run(g):
    from long_native_r179 import Native
    from long_ar_r179 import generate
    node, control, report, save = (g[k] for k in ["node", "control", "report", "save"])
    native = Native(g)
    model = g["model"]
    run_id = report["run"]
    root = pathlib.Path("/refactor")
    stop_file = root / "results" / ("Kimi-agent-service-stop-r194-" + run_id + ".json")
    identity_file = root / "results" / ("Kimi-agent-service-identity-r194-" + run_id + ".json")
    ready_file = root / "results" / ("Kimi-agent-service-ready-r194-" + run_id + ".json")
    port = int(os.environ.get("KIMI_AGENT_HTTP_PORT_R144D", "43844"))
    eos = {163585, 163586}
    engine = server = thread = None
    deadline = g["experiment_started_r9046"] + int(os.environ.get("KIMI_RUNTIME_INNER_S_R9225", "8500")) - 150
    requests = report.setdefault("agent_requests", [])
    try:
        if node == 0:
            import uvicorn
            prepared=g["_agent_prepared"];codec=prepared["codec"];engine=prepared["engine"]
            app=prepared["app"];FinishReason=prepared["FinishReason"]
            server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                                   log_level="info", access_log=False))
            thread = threading.Thread(target=server.run, name="Kimi-P0-HTTP", daemon=True)
            thread.start()
            started = time.monotonic()
            while not server.started:
                if not thread.is_alive() or time.monotonic() - started > 45:
                    raise RuntimeError("Actual HTTP server startup failed before requests")
                time.sleep(.1)
            identity = {
                "model": str(model), "actual_step": os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"],
                "nodes": g["manifest"]["nodes"], "cards": 32, "TP": 16, "PP": 2,
                "native_SHA256": g["manifest"]["native_DSO_SHA256"],
                "manifest_SHA256": report["manifest_SHA256"], "actual_CAPI_providers": report["actual_CAPI_providers"],
                "model_config_SHA256": g["manifest"]["model_view"]["config_SHA256"],
                "model_index_SHA256": g["manifest"]["model_view"]["index_SHA256"],
                "tokenizer": codec.identity, "protocol_commit": "41a013adf042e18e579689b46b6e9f11bf1fb77b",
                "backend_scheduling": "single serial Native queue; HTTP/SSE may overlap",
                "normal_decode": "Native179 explicit32K/cap1; actual prompt seed512 plus true-input width8 fullcommit, then original width1 greedy; no draft",
                "max_context_tokens": 32768, "max_prompt_tokens": 32767, "default_output_budget": 4096,
                "thinking_output_budget":8192,"thinking_default_effort":"high","honored_thinking_efforts":["low","high","max"],
                "CPU_receipt_SHA256":"37ac44052ffb3b0e43cf7e440178aad4096f09dcd97a8b12d6eca6fc65cd827d",
                "buffered_P0_SSE": True, "HTTP_incremental_latency_qualified": False,
                "validation_timeouts_s": dict(engine.validation_timeouts_s),
                "model_or_tools_qualified": False,
            }
            identity_file.write_bytes((json.dumps(identity, indent=2) + "\n").encode())
            ready_file.write_bytes((json.dumps({"ready": True, "service_node": g["manifest"]["nodes"][0],
                "URL_on_service_node": "http://127.0.0.1:%d/v1" % port,
                "identity_path": str(identity_file), "stop_file": str(stop_file),
                "actual_step": identity["actual_step"], "model_or_tools_qualified": False}, indent=2) + "\n").encode())
        control.collect("agent-http-ready", {"node": node})
        report.update(phase="agent_resident_ready", serving_protocol_complete=False,
                      agent_service_url="http://127.0.0.1:%d/v1" % port if node == 0 else None)
        save()
        ordinal = 0
        while True:
            req = None
            command = None
            if node == 0:
                if stop_file.exists() or time.monotonic() >= deadline:
                    command = {"action": "stop", "reason": "explicit stop file or reserved cleanup"}
                else:
                    try:
                        req = engine.take(timeout=.2)
                        if req is None:
                            command = {"action": "stop", "reason": "HTTP lifespan ended"}
                        else:
                            command = {"action": "request", "id": req.request_id,
                                "prompt_ids": list(req.prompt_token_ids), "max_new_tokens": req.max_new_tokens,
                                "cancelled": req.cancel_event.is_set(),
                                "accepted_monotonic": req.metadata["accepted_monotonic"],
                                "active_requests_at_accept": req.metadata["active_requests_at_accept"],
                                "client_session": req.metadata["client_session"]}
                    except queue.Empty:
                        command = {"action": "idle"}
            command = control.collect("agent-command:%d" % ordinal, {"command": command})[0]["command"]
            ordinal += 1
            if command["action"] == "stop":
                report["agent_stop"] = command
                break
            if command["action"] == "idle":
                continue
            assert command["action"] == "request"
            prompt = command["prompt_ids"]; cap = command["max_new_tokens"]
            assert 1 <= len(prompt) < 32768 and 1 <= cap <= 8192 and len(prompt) + cap <= 32768
            assert all(type(x) is int and 0 <= x < g["vocab"] for x in prompt)
            ident = command["id"]; label = "agent-" + ident
            generated = []; ended = None
            item = {"request_id": ident, "input_tokens": len(prompt), "output_budget": cap,
                    "started_monotonic": time.monotonic(), "generated_ids": generated,
                    "real_model": True, "normal_reset": False,
                    "accepted_monotonic": command["accepted_monotonic"],
                    "active_requests_at_accept": command["active_requests_at_accept"],
                    "client_session": command["client_session"]}
            requests.append(item)
            failure = None
            observation_start=len(getattr(native,'long_prefill_observations',[]))
            try:
                native.model_events.set_next(label, False)
                def input_progress(committed,total,transactions,wall):
                    item['long_prefill_progress']={'input_tokens_committed':committed,'prompt_tokens':total,'fullcommit_transactions':transactions,'elapsed_s':wall,'real_completed_Native_input_only':True}
                    save()
                    if node==0:req._events.put(('progress',None))
                native.input_commit_progress=input_progress
                native.reset(False)
                os.environ["INFINILM_KIMI_I192_GROUPED_R8669Y"] = "0"
                os.environ["INFINILM_KIMI_MOE_EXECUTION_R4665"] = "0"
                if not command["cancelled"]:
                    def emit(token):
                        generated.append(token)
                        if node == 0:req.emit_token(token)
                    def cancelled(step):
                        return control.collect("agent-cancel:%s:%d" % (ident, step),
                            {"cancelled": req.cancel_event.is_set() if node == 0 else None})[0]["cancelled"]
                    actual, ended = generate(native, prompt, cap, eos, emit, cancelled, deadline)
                    assert actual == generated
                else:ended = "canceled"
                if native.memory_guard:native.memory_guard("agent-end:" + ident)
            except BaseException as error:
                item["error"] = repr(error)
                failure = error
                save()
                raise
            finally:
                # Existing collective reset/drain completes before any HTTP finish.
                try:
                    native.reset(False)
                    control.collect("agent-request-reset:" + ident, {"node": node})
                    item["normal_reset"] = True
                except BaseException as reset_error:
                    item["reset_error"] = repr(reset_error)
                    if node == 0:req.fail(reset_error)
                    raise
                else:
                    if failure is not None and node == 0:req.fail(failure)
                native.input_commit_progress=None
                item['long_prefill_observations']=getattr(native,'long_prefill_observations',[])[observation_start:]
                item["finished_monotonic"] = time.monotonic();item["finish"] = ended
            if node == 0:
                reason = {"eos": FinishReason.EOS_TOKEN, "length": FinishReason.LENGTH,
                          "canceled": FinishReason.CANCELED, "timeout": FinishReason.TIMEOUT}[ended]
                req.finish(reason)
            report["phase"] = "agent_resident_ready"
            save()
        report["serving_protocol_complete"] = True
    finally:
        report["agent_native_forward_calls"] = len(native.calls)
        native.arm([], [], [], False)
        if server is not None:server.should_exit = True
        if thread is not None:
            thread.join(timeout=30)
            if thread.is_alive():raise RuntimeError("HTTP server did not stop normally")
        if engine is not None:engine.stop()
        report["agent_HTTP_normal_shutdown"] = bool(node != 0 or thread is not None and not thread.is_alive())
        save()
