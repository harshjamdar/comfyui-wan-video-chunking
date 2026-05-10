# ComfyUI Wan Video Chunking

Custom ComfyUI nodes and a master workflow for rendering long Wan Animate videos in 2-second chunks, then stitching the chunks into one final video.

Repository:

```text
https://github.com/harshjamdar/comfyui-wan-video-chunking.git
```

## What This Is For

Wan video generation becomes very slow when the full video is sent to the GPU at once. This project makes the workflow chunk-based:

```text
full video -> 2s chunks -> Wan/KSampler per chunk -> rendered chunk videos -> ffmpeg stitch -> final video
```

Default chunk size:

```text
30 FPS x 2 seconds = 60 frames per chunk
```

## Files

```text
ComfyUI-VideoChunking/          Custom ComfyUI node package
wan_chunked_master_workflow.json Master workflow with the chunk runner node
wan_chunked_master_workflow_api.json API prompt for deployment
wan_chunked_master_payload.json API /prompt payload for direct curl runs
workflows/gemini_final_1_api.json Full Wan worker pipeline in API format
gemini final 1.json              Current visual Wan baseline workflow
```

## Server Install

Run these commands on your GPU server.

Set your ComfyUI path first:

```bash
export COMFYUI_DIR="$HOME/ComfyUI"
```

If your ComfyUI is somewhere else, change it:

```bash
export COMFYUI_DIR="/path/to/ComfyUI"
```

Clone this repo:

```bash
cd "$HOME"
git clone https://github.com/harshjamdar/comfyui-wan-video-chunking.git
cd comfyui-wan-video-chunking
```

Install the custom nodes:

```bash
cp -r ComfyUI-VideoChunking "$COMFYUI_DIR/custom_nodes/"
cp wan_chunked_master_workflow.json "$COMFYUI_DIR/"
cp wan_chunked_master_workflow_api.json "$COMFYUI_DIR/"
cp wan_chunked_master_payload.json "$COMFYUI_DIR/"
mkdir -p "$COMFYUI_DIR/workflows"
cp workflows/gemini_final_1_api.json "$COMFYUI_DIR/workflows/"
cp "gemini final 1.json" "$COMFYUI_DIR/"
```

Install dependencies in the same Python environment used by ComfyUI:

```bash
cd "$COMFYUI_DIR"
python -m pip install opencv-python
```

Install ffmpeg if it is missing:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

Verify:

```bash
python - <<'PY'
import cv2
print("opencv ok", cv2.__version__)
PY

ffmpeg -version | head -n 1
```

Restart ComfyUI after installing the node.

## Update Existing Server Install

Use this when you already cloned the repo before:

```bash
cd "$HOME/comfyui-wan-video-chunking"
git pull
rm -rf "$COMFYUI_DIR/custom_nodes/ComfyUI-VideoChunking"
cp -r ComfyUI-VideoChunking "$COMFYUI_DIR/custom_nodes/"
cp wan_chunked_master_workflow.json "$COMFYUI_DIR/"
cp wan_chunked_master_workflow_api.json "$COMFYUI_DIR/"
cp wan_chunked_master_payload.json "$COMFYUI_DIR/"
mkdir -p "$COMFYUI_DIR/workflows"
cp workflows/gemini_final_1_api.json "$COMFYUI_DIR/workflows/"
```

Restart ComfyUI.

## Deployment-Ready JSON Files

There are two workflow JSONs for deployment:

```text
workflows/gemini_final_1_api.json
```

This is the full Wan Animate worker pipeline converted to ComfyUI API format. It contains the model loaders, video loader, pose detection, WanAnimateToVideo, sampler, VAE decode, upscale, CreateVideo, and SaveVideo nodes.

```text
wan_chunked_master_workflow_api.json
```

This is the API-format master prompt. It contains the `WanChunkedWorkflowRunner`, which loads `workflows/gemini_final_1_api.json`, runs it once per chunk, then stitches all chunk videos into the final MP4.

```text
wan_chunked_master_payload.json
```

This wraps the master prompt as `{"prompt": ...}` so it can be posted directly to ComfyUI.

Recommended production layout:

```text
Controller ComfyUI, port 8188 -> receives your deployment API call
Worker ComfyUI, port 8190     -> receives each Wan chunk job
```

Start them:

```bash
cd "$COMFYUI_DIR"
python main.py --listen 0.0.0.0 --port 8188
```

In a second terminal:

```bash
cd "$COMFYUI_DIR"
python main.py --listen 127.0.0.1 --port 8190
```

Queue the complete chunked pipeline with one command:

```bash
cd "$COMFYUI_DIR"
curl -sS -X POST \
  -H "Content-Type: application/json" \
  --data-binary @wan_chunked_master_payload.json \
  http://127.0.0.1:8188/prompt
```

Or queue it with Python:

```bash
cd "$COMFYUI_DIR"
python - <<'PY'
import json
import urllib.request

prompt = json.load(open("wan_chunked_master_workflow_api.json", "r", encoding="utf-8"))
payload = json.dumps({"prompt": prompt}).encode("utf-8")
req = urllib.request.Request(
    "http://127.0.0.1:8188/prompt",
    data=payload,
    headers={"Content-Type": "application/json"},
)
print(urllib.request.urlopen(req).read().decode("utf-8"))
PY
```

Before running, put your source files here or update the paths in `wan_chunked_master_workflow_api.json`:

```text
$COMFYUI_DIR/input/test.mp4
$COMFYUI_DIR/input/ChatGPT Image May 10, 2026, 05_20_13 PM.png
```

Final output:

```text
$COMFYUI_DIR/output/wan22_chunked_final.mp4
```

## Optional API Workflow Export

The repo already includes `workflows/gemini_final_1_api.json`. Re-export only if you change the visual Wan pipeline in ComfyUI.

In your browser ComfyUI:

1. Load `gemini final 1.json`.
2. Open ComfyUI settings.
3. Enable developer mode.
4. Use `Save (API Format)`.
5. Save the exported file on the server as:

```text
$COMFYUI_DIR/workflows/gemini_final_1_api.json
```

Create the workflows folder if needed:

```bash
mkdir -p "$COMFYUI_DIR/workflows"
```

Expected node IDs in the API workflow:

```text
1   VHS_LoadVideo
7   Positive CLIPTextEncode
8   Negative CLIPTextEncode
10  PoseAndFaceDetection
11  DrawViTPose
12  WanAnimateToVideo
13  KSampler
18  SaveVideo
```

These match the included `gemini final 1.json`.

## Load The Master Workflow

In ComfyUI, load:

```text
wan_chunked_master_workflow.json
```

You should see:

```text
Wan Chunked Workflow Runner
```

Set these fields:

```text
workflow_api_json_path: /home/YOUR_USER/ComfyUI/workflows/gemini_final_1_api.json
comfyui_api_url: http://127.0.0.1:8190
video_path: /home/YOUR_USER/ComfyUI/input/test.mp4
output_dir: /home/YOUR_USER/ComfyUI/output
final_output_path: /home/YOUR_USER/ComfyUI/output/wan22_chunked_final.mp4
fps: 30
chunk_seconds: 2.0
width: 480
height: 848
steps: 10
cfg: 1.0
```

For the video path, use the real path on the server.

## Important Run Pattern

The master node is an API orchestrator. It queues chunk jobs into ComfyUI and waits for them.

Best production-safe setup:

```text
Controller ComfyUI -> runs Wan Chunked Workflow Runner
Worker ComfyUI     -> receives chunk jobs through comfyui_api_url
```

For quick testing, you can try one ComfyUI instance, but if the node waits while the same queue is blocked, it can deadlock. If that happens, run a second ComfyUI worker on another port.

Example two-instance setup:

Terminal 1, controller:

```bash
cd "$COMFYUI_DIR"
python main.py --listen 0.0.0.0 --port 8188
```

Terminal 2, worker:

```bash
cd "$COMFYUI_DIR"
python main.py --listen 127.0.0.1 --port 8190
```

Then set the master node:

```text
comfyui_api_url: http://127.0.0.1:8190
```

## Upload Input Video

Put the input video in ComfyUI input:

```bash
cp /path/to/your/video.mp4 "$COMFYUI_DIR/input/test.mp4"
```

Or update `video_path` in the master node to the actual file path.

## Final Output

After the master node completes, the final stitched video is written to:

```text
$COMFYUI_DIR/output/wan22_chunked_final.mp4
```

Rendered chunk files are also left in the output folder with the configured chunk prefix.

## Troubleshooting

If the node does not appear:

```bash
ls "$COMFYUI_DIR/custom_nodes/ComfyUI-VideoChunking"
```

Then restart ComfyUI and check the terminal logs.

If OpenCV is missing:

```bash
cd "$COMFYUI_DIR"
python -m pip install opencv-python
```

If ffmpeg is missing:

```bash
sudo apt install -y ffmpeg
```

If you get an API JSON error, re-export from ComfyUI using `Save (API Format)`. The normal visual workflow JSON will not work with the ComfyUI `/prompt` API.
