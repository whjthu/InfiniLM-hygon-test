# VGGT .1318 Dev24 zero-GRES CPU readback

The earlier HOST request `e31048eef0024ce1b13725d222ddd62a` stopped at `HOST_NUMPY_UNAVAILABLE` before target staging. The original scoring request `6bd97bc61f0249c0a255ad26fe55a162` records a separate successful `1103606.1509` zero-GRES audit with two CPUs, 2 GiB and a 4661-byte independent receipt. The exact local .1509 launcher and NumPy ABI evidence could not be read. The new package uses the separately published `1087656.173` CPU runtime precedent, with Apptainer 1.3.4, the DTK sandbox and the original T06 shared runtime mount. That step reported NumPy 1.25.0. The new worker records the actual runtime directly.

The new geometry is one independently admitted node, one task, two logical CPUs, 2048 MiB and zero GRES. The worker budget is 100 seconds from the admitted epoch. The independent terminal gate allows the five-second teardown, requires the exact numeric step and admits only `cpu=2,mem=2G,node=1` or its 2048M spelling. Historical node IDs establish provenance. The current resource owner chooses placement.

## Local verification and staging

```powershell
python -B -X utf8 code/bw1000/training-vggt-original1318-dev24-cpu-readback-v1/prepare_local.py check --output local-only/1318-cpu-independent-absent.json
python -B -X utf8 code/bw1000/training-vggt-original1318-dev24-cpu-readback-v1/prepare_local.py packet --output local-only/1318-cpu-source-offer.json
python -B -X utf8 code/bw1000/training-vggt-original1318-dev24-cpu-readback-v1/prepare_local.py install-command --request REQUEST --output local-only/1318-cpu-install-command.txt
```

When the current owner authorizes a new request, send the generated install command as one held ordinary SSH remote command and the exact source offer as stdin. The isolated HOST bootstrap validates the 21 members and atomically stages them under the new request's `incoming` directory. The source packet contains the frozen audit, its source and public receipt dependencies, and three CPU implementation files. Historical results remain read-only. Stage installation uses zero scheduler calls.

From the request's staged `checkout` directory, use the original absolute HOST Python with `-B -u host.py readback-install --request REQUEST --install-sha SHA` for independent source readback. After fresh CPU-only placement, call `host.py run --request REQUEST --install-sha SHA --admission ABSOLUTE_REQUEST_STAGE_ADMISSION_JSON --admission-sha SHA`. The admission JSON binds the request, parent, node, epoch, deadline, observed node epoch, positive CPU-only admission and exact resource dictionary. The runner checks real parent ownership, membership and current numbered steps before persisting its one-time launch intent. A timed-out or ambiguous launch preserves the request directory for reconciliation.

## Provider and independent accounting

The same compute worker actually imports NumPy, checks the reported distribution version, performs a small CPU norm and records the Python package and loaded core extension paths and byte digests. It saves `PROVIDER.json` before invoking the frozen `.1318` `target-read-only` entry in that same process. The container sees the original project results tree and runtime through read-only binds. It can record an independent provider witness even if an audit target file is missing. The original audit may stat model artifacts to check identity. No model weights are read or loaded, and no predictions or updates occur.

Independently identify the real numeric step for the request label. With `TZ=UTC` and `SLURM_TIME_FORMAT=%Y-%m-%dT%H:%M:%S`, capture `sacct -nP -j STEP --format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End`. Use the staged `verify_terminal.py` with `--request`, `--step`, `--intent-sha`, `--accounting`, `--accounting-sha` and a new `--output` under the same request evidence directory. A missing worker summary can retain an independently bound provider witness with zero qualified input rows. The fixed original denominator is 24 windows, with 22 locally verified inputs and two original missing windows. The target count rises only after valid target audit and exact successful independent accounting.

This delivery contains local verification and captured launch commands only. The target runtime, source install and CPU step need new independent evidence.
