# ComfyUI Wan Chunked Sampler

This repo contains a Comfy-native custom sampler node and a ready-to-load workflow:

```text
ComfyUI-WanChunkedSampler/
main workflow.json
```

The workflow replaces the normal `KSampler` with `Chunked Wan KSampler`. The node samples video latents in smaller time chunks so longer videos do not go through one huge sampler pass.

## Install

Copy the custom node into ComfyUI:

```bash
cp -r ComfyUI-WanChunkedSampler /root/comfy/ComfyUI/custom_nodes/
cp "main workflow.json" /root/comfy/ComfyUI/
```

Restart ComfyUI.

## Test

Load this workflow in ComfyUI:

```text
main workflow.json
```

The sampler node is already connected.

Default chunk settings:

```text
latent_chunk_frames: 16
overlap_latent_frames: 0
```

The sampler widget order is intentionally plain and stable:

```text
noise_seed
steps
cfg
sampler_name
scheduler
denoise
latent_chunk_frames
overlap_latent_frames
```

If the output has visible temporal cuts, try:

```text
latent_chunk_frames: 12
overlap_latent_frames: 2
```

## Important

This stays inside ComfyUI. There is no external API runner or background queue script.

The workflow is still limited by upstream nodes. If `VHS_LoadVideo`, pose detection, or `WanAnimateToVideo` load/process the entire input video before sampling, those nodes can still consume time and VRAM. This node specifically reduces the KSampler long-video bottleneck.
