# ComfyUI Video Chunking Nodes

Custom nodes for rendering video in fixed-size chunks, for example 2 seconds at 30 FPS = 60 frames.

## Install On The Droplet

Copy this folder into your ComfyUI install:

```bash
cp -r ComfyUI-VideoChunking /path/to/ComfyUI/custom_nodes/
```

Restart ComfyUI.

If OpenCV is missing in the ComfyUI Python environment:

```bash
pip install opencv-python
```

`FFmpeg Concat Chunks` also needs `ffmpeg` available in `PATH`.

## Nodes

### Wan Chunked Workflow Runner

This is the master controller node.

It:

- reads the source video frame count
- calculates 2-second chunks
- modifies an API-format Wan workflow for each chunk
- queues each chunk through the ComfyUI HTTP API
- waits for each chunk to finish
- finds the rendered chunk video files
- stitches them with ffmpeg
- optionally re-attaches audio from the source video

Required setup:

1. Open your normal Wan workflow in ComfyUI.
2. Enable developer mode in ComfyUI settings.
3. Export the workflow using **Save (API Format)**.
4. Save it on the droplet, for example:

```text
ComfyUI/workflows/gemini_final_1_api.json
```

5. Load `wan_chunked_master_workflow.json`.
6. Set `workflow_api_json_path` to the API JSON path.
7. Set `comfyui_api_url` to the ComfyUI API URL.

Node IDs expected in the API workflow:

- `1`: `VHS_LoadVideo`
- `7`: positive `CLIPTextEncode`
- `8`: negative `CLIPTextEncode`
- `10`: `PoseAndFaceDetection`
- `11`: `DrawViTPose`
- `12`: `WanAnimateToVideo`
- `13`: `KSampler`
- `18`: `SaveVideo`

These match the current `gemini final 1.json` workflow.

Important deployment note:

Do not run this node against the same single ComfyUI queue if it blocks waiting for child prompts. The recommended setup is:

- controller ComfyUI instance: runs the master node
- worker ComfyUI instance: same GPU/models/custom nodes, receives the chunk jobs through `comfyui_api_url`

For production, the same orchestration code can run in your backend instead of inside ComfyUI.

### Video Info Probe

Reads a video file and outputs:

- total frames
- FPS
- duration
- width
- height

### Video Chunk Planner

Inputs:

- `total_frames`
- `chunk_index`
- `fps`
- `chunk_seconds`

Outputs:

- `skip_first_frames`
- `frame_load_cap`
- `chunk_count`
- `is_last_chunk`
- `chunk_label`

For 30 FPS and 2 seconds, each chunk is 60 frames. The last chunk is automatically capped to the remaining frame count.

### Video Chunk Loader

Loads only one chunk from the video and outputs:

- `images`
- `frame_count`
- `fps`

Connect:

- `images` to pose detection and the video/face input path
- `frame_count` to `WanAnimateToVideo.length`

Use this instead of loading the whole video into Wan.

### Chunk Output Prefix

Creates a unique filename prefix per chunk. Connect it to a save-video node if that node accepts a filename prefix input. If the save node only exposes a widget, set the prefix manually per chunk.

### FFmpeg Concat Chunks

Stitches rendered chunk videos after all chunks are generated. It can also attach audio from the original video.

## Important ComfyUI Limitation

A normal ComfyUI node cannot make downstream nodes run repeatedly for every chunk in one queue execution. This package gives you the chunk-aware nodes needed to render one chunk at a time inside the graph.

`Wan Chunked Workflow Runner` solves this by acting as an API orchestrator: it queues a full chunk workflow repeatedly and stitches the outputs after all chunk prompts finish.
