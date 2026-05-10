# ComfyUI Wan Chunked Sampler

This repo contains a Comfy-native custom sampler node and a ready-to-load workflow:

```text
ComfyUI-WanChunkedSampler/
main workflow.json
```

The workflow replaces the normal `KSampler` with `Wan Latent Chunk Sampler`. The node samples video latents in smaller time chunks so longer videos do not go through one huge sampler pass.

The workflow also replaces full-batch decode/upscale nodes with chunked versions:

```text
Wan Chunked VAE Decode
Wan Chunked Image Upscale
Wan Chunked Image Scale
```

These reduce peak VRAM/RAM during VAE decode and upscaling.

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
decode_chunk_frames: 16
upscale_chunk_frames: 8
scale_chunk_frames: 8
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
UPLOAD REFERENCE IMAGE -> reference image upload/select
UPLOAD MOTION VIDEO    -> input video upload/select
```

The filenames in the workflow are only defaults. For every run, upload/select the new reference image in `LoadImage` and the new video in `VHS_LoadVideo`.

The reference image node was moved away from the large video loader node so it is visible on the canvas.

## Remaining Limit

This workflow still creates the full Wan latent before decode. The sampling, decode, model upscale, and resize steps are chunked, but `VHS_LoadVideo`, pose detection, and `WanAnimateToVideo` still operate on the selected video length.

## Important

This stays inside ComfyUI. There is no external API runner or background queue script.

The workflow is still limited by upstream nodes. If `VHS_LoadVideo`, pose detection, or `WanAnimateToVideo` load/process the entire input video before sampling, those nodes can still consume time and VRAM. This node specifically reduces the KSampler long-video bottleneck.
