# ComfyUI Wan Video Chunking

Custom ComfyUI nodes for rendering long Wan Animate videos in smaller chunks and stitching them into one final video.

## What It Does

- Splits a source video into fixed-size chunks, for example 2 seconds at 30 FPS.
- Queues each chunk through a Wan/ComfyUI workflow.
- Preserves full-video output while reducing the per-run GPU load.
- Stitches rendered chunks with ffmpeg.
- Optionally re-attaches source audio to the final video.

## Files

- `ComfyUI-VideoChunking/` - custom ComfyUI node package.
- `wan_chunked_master_workflow.json` - master workflow containing the chunk runner node.
- `gemini final 1.json` - current Wan workflow baseline.

## Install

Copy the custom node folder into your ComfyUI installation:

```bash
cp -r ComfyUI-VideoChunking /path/to/ComfyUI/custom_nodes/
```

Restart ComfyUI.

If OpenCV is missing:

```bash
pip install opencv-python
```

`ffmpeg` must also be available in `PATH`.

## Required Workflow Export

The master runner needs your normal Wan workflow exported in ComfyUI API format:

1. Open the Wan workflow in ComfyUI.
2. Enable developer mode.
3. Export with `Save (API Format)`.
4. Save it on the server, for example:

```text
ComfyUI/workflows/gemini_final_1_api.json
```

Then load:

```text
wan_chunked_master_workflow.json
```

Set the node fields for your server paths, video path, prompts, chunk size, and output path.

## Note

The master node is an API orchestrator. For reliable production usage, run it against a worker ComfyUI API endpoint so the controller can queue chunk jobs and wait for them without blocking the same execution queue.
