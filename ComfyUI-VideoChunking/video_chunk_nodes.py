import math
import os
import copy
import json
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import numpy as np
import torch


def _require_cv2():
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(
            "Video Chunk Loader needs opencv-python installed in the ComfyUI environment. "
            "Install it with: pip install opencv-python"
        ) from exc
    return cv2


def _video_capture(path):
    cv2 = _require_cv2()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    return cv2, capture


def _http_json(url, payload=None, timeout=30):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else {}


def _is_api_prompt(workflow):
    return isinstance(workflow, dict) and all(
        isinstance(value, dict) and "class_type" in value and "inputs" in value
        for value in workflow.values()
    )


def _set_api_input(prompt, node_id, key, value):
    node_key = str(node_id)
    if node_key not in prompt:
        raise KeyError(f"API prompt does not contain node id {node_id}")
    prompt[node_key].setdefault("inputs", {})[key] = value


def _queue_prompt(server_url, prompt, client_id):
    response = _http_json(
        f"{server_url.rstrip('/')}/prompt",
        {"prompt": prompt, "client_id": client_id},
        timeout=60,
    )
    if "prompt_id" not in response:
        raise RuntimeError(f"ComfyUI did not return prompt_id: {response}")
    return response["prompt_id"]


def _wait_for_prompt(server_url, prompt_id, poll_seconds, timeout_seconds):
    started = time.time()
    while True:
        history = _http_json(f"{server_url.rstrip('/')}/history/{prompt_id}", timeout=60)
        if prompt_id in history:
            return history[prompt_id]
        if time.time() - started > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for prompt {prompt_id}")
        time.sleep(poll_seconds)


def _find_rendered_video(output_dir, prefix, started_at):
    output_path = Path(output_dir)
    candidates = []
    for ext in ("*.mp4", "*.webm", "*.mov", "*.mkv"):
        candidates.extend(output_path.rglob(f"{prefix}*{ext[1:]}"))
    candidates = [item for item in candidates if item.stat().st_mtime >= started_at - 2]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _concat_video_files(files, output_path, reencode, attach_audio_from):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    list_path = output.with_name(output.stem + "_concat_list.txt")
    list_path.write_text("".join(f"file '{Path(f).as_posix()}'\n" for f in files), encoding="utf-8")

    temp_output = output
    if attach_audio_from:
        temp_output = output.with_name(output.stem + "_video_only" + output.suffix)

    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path)]
    if reencode:
        cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast"]
    else:
        cmd += ["-c", "copy"]
    cmd += [str(temp_output)]
    subprocess.run(cmd, check=True)

    if attach_audio_from:
        audio_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(temp_output),
            "-i",
            attach_audio_from,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output),
        ]
        subprocess.run(audio_cmd, check=True)
        try:
            os.remove(temp_output)
        except OSError:
            pass
    return output


class VideoChunkPlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total_frames": ("INT", {"default": 540, "min": 1, "max": 1000000}),
                "chunk_index": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 240}),
                "chunk_seconds": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 60.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("skip_first_frames", "frame_load_cap", "chunk_count", "is_last_chunk", "chunk_label")
    FUNCTION = "plan"
    CATEGORY = "Video Chunking"

    def plan(self, total_frames, chunk_index, fps, chunk_seconds):
        chunk_frames = max(1, int(round(fps * chunk_seconds)))
        chunk_count = int(math.ceil(total_frames / chunk_frames))
        safe_index = min(max(chunk_index, 0), max(chunk_count - 1, 0))
        skip = safe_index * chunk_frames
        cap = max(0, min(chunk_frames, total_frames - skip))
        is_last = safe_index == chunk_count - 1
        end = skip + cap - 1 if cap else skip
        label = f"chunk_{safe_index:04d}_f{skip:06d}_{end:06d}"
        return skip, cap, chunk_count, is_last, label


class VideoInfoProbe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "input/test.mp4"}),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("total_frames", "fps", "duration_seconds", "width", "height")
    FUNCTION = "probe"
    CATEGORY = "Video Chunking"

    def probe(self, video_path):
        cv2, capture = _video_capture(video_path)
        try:
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = float(total_frames / fps) if fps else 0.0
            return total_frames, fps, duration, width, height
        finally:
            capture.release()


class VideoChunkLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "input/test.mp4"}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                "frame_load_cap": ("INT", {"default": 60, "min": 1, "max": 100000}),
                "force_rate": ("INT", {"default": 30, "min": 1, "max": 240}),
                "width": ("INT", {"default": 480, "min": 64, "max": 8192}),
                "height": ("INT", {"default": 848, "min": 64, "max": 8192}),
                "crop": (["center", "disabled"], {"default": "center"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "FLOAT")
    RETURN_NAMES = ("images", "frame_count", "fps")
    FUNCTION = "load"
    CATEGORY = "Video Chunking"

    def _resize(self, cv2, frame, width, height, crop):
        if crop == "disabled":
            return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)

        src_h, src_w = frame.shape[:2]
        target_ratio = width / height
        src_ratio = src_w / src_h
        if src_ratio > target_ratio:
            new_w = int(src_h * target_ratio)
            x0 = max(0, (src_w - new_w) // 2)
            frame = frame[:, x0 : x0 + new_w]
        else:
            new_h = int(src_w / target_ratio)
            y0 = max(0, (src_h - new_h) // 2)
            frame = frame[y0 : y0 + new_h, :]
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)

    def load(self, video_path, skip_first_frames, frame_load_cap, force_rate, width, height, crop):
        cv2, capture = _video_capture(video_path)
        frames = []
        try:
            source_fps = float(capture.get(cv2.CAP_PROP_FPS) or force_rate)
            step = max(1, int(round(source_fps / force_rate))) if force_rate else 1
            capture.set(cv2.CAP_PROP_POS_FRAMES, skip_first_frames)
            source_frame = skip_first_frames

            while len(frames) < frame_load_cap:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                if (source_frame - skip_first_frames) % step == 0:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frame_rgb = self._resize(cv2, frame_rgb, width, height, crop)
                    frames.append(frame_rgb.astype(np.float32) / 255.0)
                source_frame += 1

            if not frames:
                raise RuntimeError(
                    f"No frames loaded from {video_path}. "
                    f"skip_first_frames={skip_first_frames}, frame_load_cap={frame_load_cap}"
                )

            tensor = torch.from_numpy(np.stack(frames, axis=0))
            return tensor, len(frames), float(force_rate)
        finally:
            capture.release()


class ChunkOutputPrefix:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_prefix": ("STRING", {"default": "wan22_chunk"}),
                "chunk_label": ("STRING", {"default": "chunk_0000_f000000_000059"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename_prefix",)
    FUNCTION = "prefix"
    CATEGORY = "Video Chunking"

    def prefix(self, base_prefix, chunk_label):
        clean_base = base_prefix.strip().replace("\\", "/").rstrip("/")
        clean_label = chunk_label.strip().replace("\\", "/").replace("/", "_")
        return f"{clean_base}_{clean_label}",


class FFmpegConcatChunks:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chunks_dir": ("STRING", {"default": "output"}),
                "chunk_prefix": ("STRING", {"default": "wan22_chunk"}),
                "output_path": ("STRING", {"default": "output/final_stitched.mp4"}),
                "reencode": ("BOOLEAN", {"default": False}),
                "attach_audio_from": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "concat"
    OUTPUT_NODE = True
    CATEGORY = "Video Chunking"

    def concat(self, chunks_dir, chunk_prefix, output_path, reencode, attach_audio_from):
        chunks_path = Path(chunks_dir)
        files = sorted(chunks_path.glob(f"{chunk_prefix}*.mp4"))
        if not files:
            raise FileNotFoundError(f"No chunk mp4 files found in {chunks_path} with prefix {chunk_prefix}")

        list_path = chunks_path / "concat_list.txt"
        list_path.write_text("".join(f"file '{f.as_posix()}'\n" for f in files), encoding="utf-8")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temp_output = output
        if attach_audio_from:
            temp_output = output.with_name(output.stem + "_video_only" + output.suffix)

        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path)]
        if reencode:
            cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast"]
        else:
            cmd += ["-c", "copy"]
        cmd += [str(temp_output)]
        subprocess.run(cmd, check=True)

        if attach_audio_from:
            audio_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(temp_output),
                "-i",
                attach_audio_from,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(output),
            ]
            subprocess.run(audio_cmd, check=True)
            try:
                os.remove(temp_output)
            except OSError:
                pass

        return f"stitched {len(files)} chunks -> {output}",


class WanChunkedWorkflowRunner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_api_json_path": ("STRING", {"default": "workflows/gemini_final_1_api.json"}),
                "comfyui_api_url": ("STRING", {"default": "http://127.0.0.1:8190"}),
                "video_path": ("STRING", {"default": "input/test.mp4"}),
                "output_dir": ("STRING", {"default": "output"}),
                "final_output_path": ("STRING", {"default": "output/wan22_chunked_final.mp4"}),
                "positive_prompt": ("STRING", {"default": "the person is dancing, cinematic lighting, high quality", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "blurry, low quality, static, deformed, extra limbs", "multiline": True}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 240}),
                "chunk_seconds": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 60.0, "step": 0.1}),
                "width": ("INT", {"default": 480, "min": 64, "max": 8192}),
                "height": ("INT", {"default": 848, "min": 64, "max": 8192}),
                "steps": ("INT", {"default": 10, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "seed": ("INT", {"default": 963067643049281, "min": 0, "max": 0xffffffffffffffff}),
                "randomize_seed_per_chunk": ("BOOLEAN", {"default": False}),
                "output_prefix": ("STRING", {"default": "wan22_chunk"}),
                "reencode_stitch": ("BOOLEAN", {"default": False}),
                "attach_audio_from_source": ("BOOLEAN", {"default": True}),
                "poll_seconds": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 30.0, "step": 0.5}),
                "timeout_minutes_per_chunk": ("INT", {"default": 120, "min": 1, "max": 1440}),
            },
            "optional": {
                "total_frames_override": ("INT", {"default": 0, "min": 0, "max": 1000000}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("final_video_path",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "Video Chunking"

    def _prepare_chunk_prompt(
        self,
        base_prompt,
        video_path,
        positive_prompt,
        negative_prompt,
        fps,
        width,
        height,
        chunk_index,
        skip_first_frames,
        frame_load_cap,
        seed,
        randomize_seed_per_chunk,
        steps,
        cfg,
        output_prefix,
    ):
        prompt = copy.deepcopy(base_prompt)
        end_frame = skip_first_frames + frame_load_cap - 1
        chunk_prefix = f"{output_prefix}_{chunk_index:04d}_f{skip_first_frames:06d}_{end_frame:06d}"

        _set_api_input(prompt, 1, "video", video_path)
        _set_api_input(prompt, 1, "force_rate", fps)
        _set_api_input(prompt, 1, "custom_width", width)
        _set_api_input(prompt, 1, "custom_height", height)
        _set_api_input(prompt, 1, "frame_load_cap", frame_load_cap)
        _set_api_input(prompt, 1, "skip_first_frames", skip_first_frames)
        _set_api_input(prompt, 1, "select_every_nth", 1)

        _set_api_input(prompt, 7, "text", positive_prompt)
        _set_api_input(prompt, 8, "text", negative_prompt)

        _set_api_input(prompt, 10, "width", width)
        _set_api_input(prompt, 10, "height", height)
        _set_api_input(prompt, 11, "width", width)
        _set_api_input(prompt, 11, "height", height)

        _set_api_input(prompt, 12, "width", width)
        _set_api_input(prompt, 12, "height", height)
        _set_api_input(prompt, 12, "length", frame_load_cap)

        chunk_seed = seed + chunk_index if randomize_seed_per_chunk else seed
        _set_api_input(prompt, 13, "seed", chunk_seed)
        _set_api_input(prompt, 13, "steps", steps)
        _set_api_input(prompt, 13, "cfg", cfg)

        _set_api_input(prompt, 18, "filename_prefix", chunk_prefix)
        return prompt, chunk_prefix

    def run(
        self,
        workflow_api_json_path,
        comfyui_api_url,
        video_path,
        output_dir,
        final_output_path,
        positive_prompt,
        negative_prompt,
        fps,
        chunk_seconds,
        width,
        height,
        steps,
        cfg,
        seed,
        randomize_seed_per_chunk,
        output_prefix,
        reencode_stitch,
        attach_audio_from_source,
        poll_seconds,
        timeout_minutes_per_chunk,
        total_frames_override=0,
    ):
        workflow_path = Path(workflow_api_json_path)
        if not workflow_path.exists():
            raise FileNotFoundError(f"Workflow API JSON not found: {workflow_path}")

        base_prompt = json.loads(workflow_path.read_text(encoding="utf-8"))
        if not _is_api_prompt(base_prompt):
            raise ValueError(
                "workflow_api_json_path must point to ComfyUI API-format JSON. "
                "In ComfyUI, enable dev mode, then export with 'Save (API Format)'. "
                "The visual workflow JSON is not accepted by /prompt."
            )

        if total_frames_override and total_frames_override > 0:
            total_frames = int(total_frames_override)
        else:
            cv2, capture = _video_capture(video_path)
            try:
                total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            finally:
                capture.release()

        chunk_frames = max(1, int(round(fps * chunk_seconds)))
        chunk_count = int(math.ceil(total_frames / chunk_frames))
        client_id = str(uuid.uuid4())
        rendered_files = []

        for chunk_index in range(chunk_count):
            skip = chunk_index * chunk_frames
            cap = min(chunk_frames, total_frames - skip)
            prompt, chunk_prefix = self._prepare_chunk_prompt(
                base_prompt=base_prompt,
                video_path=video_path,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                fps=fps,
                width=width,
                height=height,
                chunk_index=chunk_index,
                skip_first_frames=skip,
                frame_load_cap=cap,
                seed=seed,
                randomize_seed_per_chunk=randomize_seed_per_chunk,
                steps=steps,
                cfg=cfg,
                output_prefix=output_prefix,
            )

            started_at = time.time()
            prompt_id = _queue_prompt(comfyui_api_url, prompt, client_id)
            _wait_for_prompt(
                comfyui_api_url,
                prompt_id,
                poll_seconds=poll_seconds,
                timeout_seconds=timeout_minutes_per_chunk * 60,
            )

            rendered = _find_rendered_video(output_dir, chunk_prefix, started_at)
            if rendered is None:
                raise FileNotFoundError(
                    f"Chunk {chunk_index} completed but no video was found in {output_dir} "
                    f"with prefix {chunk_prefix}"
                )
            rendered_files.append(rendered)

        audio_source = video_path if attach_audio_from_source else ""
        final_path = _concat_video_files(rendered_files, final_output_path, reencode_stitch, audio_source)
        return str(final_path),


NODE_CLASS_MAPPINGS = {
    "VideoChunkPlanner": VideoChunkPlanner,
    "VideoInfoProbe": VideoInfoProbe,
    "VideoChunkLoader": VideoChunkLoader,
    "ChunkOutputPrefix": ChunkOutputPrefix,
    "FFmpegConcatChunks": FFmpegConcatChunks,
    "WanChunkedWorkflowRunner": WanChunkedWorkflowRunner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoChunkPlanner": "Video Chunk Planner",
    "VideoInfoProbe": "Video Info Probe",
    "VideoChunkLoader": "Video Chunk Loader",
    "ChunkOutputPrefix": "Chunk Output Prefix",
    "FFmpegConcatChunks": "FFmpeg Concat Chunks",
    "WanChunkedWorkflowRunner": "Wan Chunked Workflow Runner",
}
