"""Directed relative-translation loss for the two-frame VGGT TUM workload."""

from hashlib import sha256
from pathlib import Path


WEIGHT = 0.1
MINIMUM_BASELINE = 0.005
PREDICTED_NORM_FLOOR = 0.001
PROVIDER_SHA256 = {
    "pose_enc.py": "845fc3a46cb851bebf0720b8fc2b9f34334ae8e11f1982376ba59f90ff18f22b",
    "rotation.py": "ad3a904e243c6654f0aa5f1d0206ace5f2d61011bc1bf8f492b40e0b76cb2070",
}
PROBE_ENCODINGS = ((0.01, 0.01, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0),
                   (0.01, 0.03, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0))


def relative_translation(extrinsics):
    """Compute the translation from camera 0 to camera 1 for world-to-camera poses."""
    if extrinsics.ndim != 4 or extrinsics.shape[1:] != (2, 3, 4):
        raise ValueError("Expected a batch of two OpenCV 3x4 camera poses")
    first, second = extrinsics[:, 0], extrinsics[:, 1]
    rotation = second[:, :, :3] @ first[:, :, :3].swapaxes(-2, -1)
    return second[:, :, 3] - (rotation @ first[:, :, 3:4])[:, :, 0]


def directional_cost(predicted, target):
    """Return per-pair costs and squared target distances in the input dtype."""
    pred_translation = relative_translation(predicted)
    target_translation = relative_translation(target)
    if pred_translation.shape != target_translation.shape:
        raise ValueError("Predicted and reference camera batches must match")
    pred_sq = (pred_translation * pred_translation).sum(-1)
    target_sq = (target_translation * target_translation).sum(-1)
    scale = (pred_sq.clip(PREDICTED_NORM_FLOOR**2, None) *
             target_sq.clip(MINIMUM_BASELINE**2, None)) ** 0.5
    cosine = (pred_translation * target_translation).sum(-1) / scale
    return 1 - cosine.clip(-1, 1), target_sq


def training_penalty(torch, decode, prediction, batch):
    """Compute the additional term in FP32 with the original training pose decoder."""
    image_hw = batch["images"].shape[-2:]
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        predicted, _ = decode(prediction.float(), image_hw, pose_encoding_type="absT_quaR_FoV",
                              build_intrinsics=False)
        costs, distance_sq = directional_cost(predicted, batch["extrinsics"].float())
        frames = batch["point_masks"].sum(dim=(-1, -2)) > 100
        if frames.shape != (len(costs), 2):
            raise ValueError("Expected original valid-depth masks for both frames")
        usable = frames.all(dim=1) & (distance_sq >= MINIMUM_BASELINE**2)
        return torch.where(usable, costs, 0.0).sum() / usable.sum().clamp_min(1)


def checked_decoder(torch, source_root):
    """Bind the mounted decoder and prove that its prediction path has a gradient."""
    expected_root = Path(source_root).resolve() / "vggt" / "utils"
    for name, wanted in PROVIDER_SHA256.items():
        path = expected_root / name
        if sha256(path.read_bytes()).hexdigest() != wanted:
            raise ValueError("VGGT direction-loss decoder source does not match: " + name)

    from vggt.utils import pose_enc, rotation

    for module in (pose_enc, rotation):
        expected = expected_root / (module.__name__.rsplit(".", 1)[-1] + ".py")
        if Path(module.__file__).resolve() != expected:
            raise ValueError("VGGT direction-loss decoder resolved to another source tree")

    decoder = pose_enc.pose_encoding_to_extri_intri
    with torch.enable_grad():
        prediction = torch.tensor([PROBE_ENCODINGS],
                                  dtype=torch.float32, device="cpu", requires_grad=True)
        target = torch.eye(3, 4, dtype=torch.float32, device="cpu").repeat(1, 2, 1, 1)
        target[0, 1, 0, 3] = 0.02
        images = torch.empty((1, 2, 3, 16, 16), dtype=torch.float32, device="cpu")
        masks = torch.ones((1, 2, 16, 16), dtype=torch.bool, device="cpu")
        cost = training_penalty(torch, decoder, prediction,
                                dict(extrinsics=target, images=images, point_masks=masks))
        gradient, = torch.autograd.grad(cost, prediction)
    if (not torch.allclose(cost, torch.ones_like(cost), atol=1e-5, rtol=0)
            or not bool(torch.isfinite(gradient).all())
            or not bool(gradient[..., :3].abs().max() > 0)
            or not bool(gradient[..., 3:7].abs().max() > 0)):
        raise ValueError("VGGT direction-loss decoder failed the CPU gradient preflight")
    return decoder
