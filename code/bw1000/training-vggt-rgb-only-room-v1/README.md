# VGGT Freiburg room RGB-only camera candidate

This overlay adds the explicit `vggt_input_mode="tum_room_rgb_only"` mode to the pinned VGGT backend. The reader loads the seven existing RGB224 pairs in windows 02–08 from the twelve-window Freiburg room plan. The seven candidates share one scene. It checks the public and private pins, each ordered crop and original frame identity, the original pose time match, and the stored camera transform against the two matched authoritative poses. A failure rejects the read before building any native batch. The original BAG frame timestamps serve as ordered frame IDs.

The RGB-only path builds camera targets from the existing first-camera extrinsics and shared crop intrinsics. It keeps the native camera pose encoding, stage weights and loss calculation. The candidate supplies boolean `pose_valid` from the two checked pose associations. It creates no depth or point masks. Its batch hash binds the RGB-only mode, order, images, camera parameters and pose validity. The first-update recorder consumes the same native target and validity interface.

## Stage the candidate in an isolated framework

The renderer accepts only the canonical `vggt_training.py` with SHA-256 `49bfe39b7aa6a14df9fdf1dda73e6db00836a81d2806012b312c16b2d3fbed6c`. Run the renderer with the frozen source and an absent output path. Install the rendered file as `gencuda/backends/vggt_training.py` only in an isolated candidate framework. Copy `vggt_rgb_only_room.py` beside it. Keep the exact original backend in the existing default RGB-D deployment.

Configure the isolated candidate with `vggt_input_mode="tum_room_rgb_only"` and `rgb_only_room_root` pointing to the checkout that has the three published result manifests and their original private crops and manifests. The reader expects `image_size=224`, `frames_per_step=2`, and all seven `cached_batches`; omitted values use those settings for this mode. The rest of the canonical framework, model weights and runtime dependencies are still required for a native run.

The rendered overlay retains the original default-mode processing and camera computation by structural comparison. The native checkpoint identity records the actual backend source digest, so the candidate has its own identity. No checkpoint was migrated or tested across those identities.

## Local verification

From the experiment checkout, run `verify_local.py` with the checkout path and an absent private receipt path. It checks syntax, the whole default RGB-D AST after removing the opt-in branches, all seven original RGB and camera pairs, and targeted failures for corrupt bytes, changed order, invalid image storage and incorrect pose evidence. The content-hash test exercises the rendered backend projection with a NumPy-only test serializer. Native Torch execution, target capture, optimizer updates, and held-out evaluation remain unverified.
