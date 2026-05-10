# ComfyUI Wan Chunked Sampler

This repo contains a Comfy-native custom sampler node and a ready-to-load workflow:

```text
ComfyUI-WanChunkedSampler/
main workflow.json
```

The workflow replaces the normal `KSampler` with `Wan Latent Chunk Sampler`. The node samples video latents in smaller time chunks so longer videos do not go through one huge sampler pass.

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
chunk_len: 16
overlap_len: 0
```

The sampler widget order is intentionally plain and stable:

```text
base_noise_seed
sample_steps
guidance_cfg
sampler
schedule
noise_denoise
chunk_len
overlap_len
```

If the output has visible temporal cuts, try:

```text
chunk_len: 12
overlap_len: 2
```

## User Inputs

The workflow is not hardcoded to one image or video. It has normal Comfy input nodes:

```text
LoadImage       -> reference image upload/select
VHS_LoadVideo   -> input video upload/select
```

The filenames in the workflow are only defaults. For every run, upload/select the new reference image in `LoadImage` and the new video in `VHS_LoadVideo`.

## Important

This stays inside ComfyUI. There is no external API runner or background queue script.

The workflow is still limited by upstream nodes. If `VHS_LoadVideo`, pose detection, or `WanAnimateToVideo` load/process the entire input video before sampling, those nodes can still consume time and VRAM. This node specifically reduces the KSampler long-video bottleneck.
