"""CPU keeper supervision only; no Native, model, SSH or Git import.

This module is source-ready, not deployed. The already-owned service keeper
calls tick() on its existing loop and persists snapshot(). No retries/restarts.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import time

CLIENT_SHA = "41dbe048d14ec8f023bd3448426d6e87931f094e5919db6ee72957e0cdf1d77e"
NATIVE_SHA = "3057262312679f86bcdb181ce198b9b1f1d2fccc59fc204b4adbc197132b60dd"


class ResidentP0Supervisor:
    def __init__(self, *, root, source, run, version, env, account,
                 gpu_started, clock=time.monotonic, popen=subprocess.Popen):
        self.root, self.source = Path(root), Path(source)
        self.run, self.version, self.env = run, version, env
        self.account, self.clock, self.popen = account, clock, popen
        self.stop_by = gpu_started + 6600  # earlier than Native 6850s cleanup
        self.name = "Kimi-agent-P0-client-" + version
        self.output = self.root / "results" / (self.name + "-" + run)
        prefix = "Kimi-agent-service-"
        self.ready = self.root / "results" / (prefix + "ready-" + version + "-" + run + ".json")
        self.identity = self.root / "results" / (prefix + "identity-" + version + "-" + run + ".json")
        self.stop = self.root / "results" / (prefix + "stop-" + version + "-" + run + ".json")
        self.marker = self.root / "results" / (self.name + "-launch.json")
        self.process = None
        self.state = {"client_submitted": False, "client_actual_handle": None,
                      "client_terminal": None, "client_P0_passed": None,
                      "stop_written": False, "retries": 0}

    @staticmethod
    def _hash(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _readiness(self, gpu_handle):
        ready = json.loads(self.ready.read_bytes())
        identity = json.loads(self.identity.read_bytes())
        assert ready["ready"] is True and ready["service_node"] == "a14r2n08"
        assert ready["actual_step"] == identity["actual_step"] == gpu_handle
        assert ready["URL_on_service_node"] == "http://127.0.0.1:43844/v1"
        assert identity["native_SHA256"] == NATIVE_SHA
        assert (identity["cards"], identity["TP"], identity["PP"]) == (32, 16, 2)
        assert identity["max_context_tokens"] == 1024
        self.state["identity_SHA256"] = self._hash(self.identity)

    def _stop(self, reason):
        # Exact source/run and same actual GPU handle were validated first.
        if not self.state["stop_written"]:
            payload = {"reason": reason, "owner": "same CPU keeper automatic P0 supervision",
                       "client_actual_handle": self.state["client_actual_handle"],
                       "client_P0_passed": self.state["client_P0_passed"]}
            with self.stop.open("x") as stream:
                json.dump(payload, stream)
            self.state["stop_written"] = True

    def tick(self, gpu_handle):
        if not gpu_handle or self.state["stop_written"]:
            return
        if not self.state["client_submitted"]:
            if not (self.ready.exists() and self.identity.exists()):
                return
            self._readiness(gpu_handle)
            remaining = int(self.stop_by - self.clock())
            if remaining < 900:
                self.state["client_skipped"] = "Ready too late for bounded P0; no false acceptance"
                self._stop("insufficient reserved client time")
                return
            client = self.source / "client/client_p0_r143.py"
            assert self._hash(client) == CLIENT_SHA
            assert not self.output.exists() and not self.marker.exists()
            limit = min(2700, remaining - 120)
            # Launch intent is exclusive and durable before Popen. Never retry.
            with self.marker.open("x") as stream:
                json.dump({"owned_job_name": self.name, "expected_GPU_handle": gpu_handle,
                           "identity_SHA256": self.state["identity_SHA256"],
                           "client_SHA256": CLIENT_SHA, "output": str(self.output),
                           "no_restarts": True, "CPU_only": True}, stream)
            args = ["srun", "--jobid=910380", "--overlap", "--exact", "--immediate=30",
                    "--cpu-bind=none", "-N1", "-n1", "-c1", "--gres=none", "--mem=2048M",
                    "-w", "a14r2n08", "-t", str(limit), "--job-name=" + self.name,
                    "--output=" + str(self.root / "logs" / (self.name + "-%J.log")),
                    "python3", "-B", "-u", str(client),
                    "--base-url", "http://127.0.0.1:43844/v1", "--model", "Kimi-K3",
                    "--output-dir", str(self.output), "--identity", str(self.identity),
                    "--max-tokens", "96", "--timeout", "1000", "--thinking-case"]
            # Slurm integer time limits are minutes, not seconds.
            args[args.index("-t") + 1] = "%02d:%02d:%02d" % (limit // 3600, limit // 60 % 60, limit % 60)
            log = self.root / "logs" / (self.name + "-client.log")
            with log.open("x") as stream:
                self.process = self.popen(args, env=self.env, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            self.state.update(client_submitted=True, client_PID=self.process.pid,
                              client_deadline=self.clock() + limit + 30)
            return
        row = self.account(self.name)
        if row:
            handle = row[0]
            assert self.state["client_actual_handle"] in (None, handle)
            self.state["client_actual_handle"] = handle
            status = row[2].split()[0].rstrip("+")
            if status not in {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "SUSPENDED"}:
                self.state["client_terminal"] = row
                result = self.output / "result.json"
                if result.exists():
                    report = json.loads(result.read_bytes())
                    assert report["identity_SHA256"] == self.state["identity_SHA256"]
                    self.state["client_result_SHA256"] = self._hash(result)
                    self.state["client_P0_passed"] = bool(
                        status == "COMPLETED" and row[3] == "0:0"
                        and report.get("completed") and report.get("passed"))
                else:
                    self.state["client_P0_passed"] = False
                self._stop("unique CPU client terminal; drain service normally")
                return
        if self.clock() >= self.state["client_deadline"]:
            # No guessed handle/scancel. Root follows the recorded exact handle;
            # service stop is requested and Slurm's bounded client limit owns exit.
            self.state["client_P0_passed"] = False
            self.state["client_observation_bound_exceeded"] = True
            self._stop("bounded CPU client deadline; no acceptance")

    def snapshot(self):
        return dict(self.state)

    def close_owned_client(self, terminal, run=subprocess.run):
        """After service terminal, stop only this keeper's unique CPU client."""
        if self.process is None:
            return
        row = self.account(self.name)
        if row:
            handle = row[0]
            assert self.state["client_actual_handle"] in (None, handle)
            self.state["client_actual_handle"] = handle
            if row[2].split()[0].rstrip("+") in {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "SUSPENDED"}:
                run(["scancel", "--signal=TERM", handle], env=self.env, check=True, timeout=25)
                row = terminal(self.name, handle)
            self.state["client_terminal"] = row
        elif self.process.poll() is None:
            # srun itself is our direct child; no inferred Slurm handle.
            self.process.terminate()
            self.process.wait(timeout=30)
            self.state["unassigned_owned_srun_terminated"] = True
