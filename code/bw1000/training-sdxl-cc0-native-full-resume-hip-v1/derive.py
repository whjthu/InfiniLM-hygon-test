"""Derive a separate full-checkpoint HIP provider from the exact frozen originals."""

import ast
from hashlib import sha256
import re


TRAINER = "b3e85543eaeeb42edd67dca0ffbc54391e4d8fa33b44aa1522e24b31ffc6c7bd"
BACKEND = "1965014be6e0040ffb6ba033f491569c5530a8a8d665012b46247a5ff43ed5e7"
HELPER = "3392f0ea6755e8d734ee02cc9d518436f9f579a555b599d6d46720d092048f38"
SCHEMA = "sdxl-native-lora-full-step-hip-v1"


def need(ok, code):
    if not ok:
        raise ValueError(code)


def replace(source, old, new, count=1):
    need(source.count(old) == count, "ORIGINAL_SOURCE_ANCHOR")
    return source.replace(old, new, count)


def parse(raw, digest):
    need(sha256(raw).hexdigest() == digest, "EXACT_ORIGINAL_PROVIDER")
    return raw.decode()


def trainer_source(raw):
    source = parse(raw, TRAINER)
    source = replace(source, "from genmlu import sdxl_training_resume_checkpoint as genmlu_checkpoint",
                     "import sdxl_hip_checkpoint as genmlu_checkpoint")
    # Both arms finish normally and export the native adapter. Final inference
    # is outside this diagnostic and would load the full pipeline a second time.
    line = '        save_lora_weights(Path(args.output_dir) / "pytorch_lora_weights.safetensors")\n'
    source = replace(source, line, line + "        if args.genmlu_full_checkpoint:\n"
                     "            accelerator.end_training()\n            return\n")
    ast.parse(source)
    return source.encode()


def backend_source(raw):
    source = parse(raw, BACKEND)
    source = replace(source, "    environment = dict(os.environ)\n",
                     """    if config.get("sdxl_hip_full_checkpoint") is not True:
        raise ValueError("EXPLICIT_SD_XL_HIP_FULL_CHECKPOINT_OPT_IN_REQUIRED")
    command.extend(["--genmlu_full_checkpoint",
                    "--genmlu_source_root", "/training-runtime/site",
                    "--genmlu_runtime_image_id", config["runtime_image_id"],
                    "--genmlu_measurement_warmup_steps", str(warmup_steps)])
    environment = dict(os.environ)
""")
    ast.parse(source)
    return source.encode()


def helper_source(raw):
    source = parse(raw, HELPER)
    source = replace(source, "import math\n", "import math\nimport os\n")
    source = replace(source, 'SCHEMA = "sdxl-native-lora-full-step-v1"', f'SCHEMA = "{SCHEMA}"')
    source = replace(source, 'DIFFUSERS_REVISION = "a3608b512ed7248499a44c61d954965ed9bdae4d"',
                     'DIFFUSERS_REVISION = "pinned-installed-diffusers-0.39.0"')
    source = replace(source, "torch.mlu.", "torch.cuda.", 5)
    source = replace(source, "torch_mlu_manual_seed", "torch_cuda_manual_seed", 2)
    source = replace(source, ', "torch_mlu",', ',')
    source = replace(source, ' + (("torch_mlu",) if mlu else ())', '')
    source = re.sub(r"\bmlu\b", "hip", source).replace("MLU", "HIP")
    source = replace(source, 'self.hip = accelerator.device.type == "hip"', 'self.hip = accelerator.device.type == "cuda"')
    source = replace(source, "        device = native_training_device(accelerator, torch)\n",
                     "        device = native_training_device(accelerator, torch)\n"
                     "        validate_rng(capture_rng(self.hip), self.hip)\n")
    source = replace(source, 'device.type == "hip" and torch.cuda.device_count() == 1,',
                     'device.type == "cuda" and torch.version.hip and torch.version.cuda is None\n'
                     '            and torch.cuda.device_count() == 1,')
    source = replace(source, 'return torch.device("hip", ordinal)', 'return torch.device("cuda", ordinal)')
    source = replace(source, 'and tensor.ndim == 1 and tensor.numel() > 0, "invalid HIP RNG byte state")',
                     'and tuple(tensor.shape) == (16,), "invalid HIP RNG byte state")')
    source = replace(source, "            probe = live.clone_state()\n",
                     "            old_live = live.get_state().cpu().clone()\n            probe = live.clone_state()\n")
    source = replace(source, 'require(torch.equal(probe.get_state().cpu(), tensor), "HIP RNG private clone roundtrip mismatch")',
                     'require(torch.equal(probe.get_state().cpu(), tensor)\n'
                     '                    and torch.equal(live.get_state().cpu(), old_live), "HIP RNG private clone roundtrip mismatch")')
    source = replace(source, '"backend_sha256", "imports", "parameters", "buffers", "optimizer", "scheduler", "lora_policy", "hip",',
                     '"backend_sha256", "trainer_sha256", "imports", "parameters", "buffers", "optimizer", "scheduler", "lora_policy", "hip",')
    source = replace(source, '"source_sha256", "helper_sha256", "backend_sha256")',
                     '"source_sha256", "helper_sha256", "backend_sha256", "trainer_sha256")')
    source = replace(source, 'file_sha(Path(__file__).parent / "backends/sdxl_training.py"),',
                     'file_sha(Path(__file__).with_name("sdxl_hip_common.py")),\n'
                     '             "trainer_sha256": file_sha(Path(__file__).with_name("train_text_to_image_lora_sdxl.py")),')
    source = replace(source, '"schema": SCHEMA, "completed_step": self.step, "next_step": self.step + 1,',
                     '"schema": SCHEMA, "completed_step": self.step, "next_step": self.step + 1,\n'
                     '            "process_pid": os.getpid(),\n'
                     '            "process_startticks": int(Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()[19]),')
    need(not re.search(r"\b(?:mlu|torch_mlu)\b", source), "MLU_BRANCH_LEAKED_INTO_HIP")
    ast.parse(source)
    return source.encode()
