import copy

import torch


def _get_builtin_node(class_type):
    import nodes

    try:
        return nodes.NODE_CLASS_MAPPINGS[class_type]
    except KeyError as exc:
        raise RuntimeError(f"Built-in ComfyUI node not found: {class_type}") from exc


def _slice_latent_frames(latent, start, end):
    chunk = copy.copy(latent)
    samples = latent["samples"]
    chunk["samples"] = samples[:, :, start:end].contiguous()

    for key in ("noise_mask", "batch_index"):
        value = latent.get(key)
        if isinstance(value, torch.Tensor) and value.ndim >= 3 and value.shape[2] == samples.shape[2]:
            chunk[key] = value[:, :, start:end].contiguous()

    return chunk


def _chunk_ranges(total, chunk_size):
    chunk_size = max(1, int(chunk_size))
    for start in range(0, total, chunk_size):
        yield start, min(total, start + chunk_size)


class WanLatentChunkSampler:
    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers

        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "base_noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "sample_steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "guidance_cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS,),
                "schedule": (comfy.samplers.KSampler.SCHEDULERS,),
                "noise_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chunk_len": ("INT", {"default": 16, "min": 1, "max": 512}),
                "overlap_len": ("INT", {"default": 0, "min": 0, "max": 128}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/video"

    def _run_ksampler(
        self,
        model,
        noise_seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise,
    ):
        from nodes import KSampler

        return KSampler().sample(
            model,
            noise_seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise,
        )[0]

    def _slice_latent(self, latent, start, end):
        return _slice_latent_frames(latent, start, end)

    def _slice_value(self, value, start, end, total_frames):
        if isinstance(value, torch.Tensor):
            if value.ndim >= 3 and value.shape[2] == total_frames:
                return value[:, :, start:end].contiguous()
            return value

        if isinstance(value, dict):
            return {key: self._slice_value(item, start, end, total_frames) for key, item in value.items()}

        if isinstance(value, list):
            return [self._slice_value(item, start, end, total_frames) for item in value]

        if isinstance(value, tuple):
            return tuple(self._slice_value(item, start, end, total_frames) for item in value)

        return value

    def _slice_conditioning(self, conditioning, start, end, total_frames):
        return self._slice_value(conditioning, start, end, total_frames)

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        base_noise_seed,
        sample_steps,
        guidance_cfg,
        sampler,
        schedule,
        noise_denoise,
        chunk_len,
        overlap_len,
    ):
        samples = latent_image["samples"]
        if samples.ndim != 5:
            return (
                self._run_ksampler(
                    model,
                    base_noise_seed,
                    sample_steps,
                    guidance_cfg,
                    sampler,
                    schedule,
                    positive,
                    negative,
                    latent_image,
                    noise_denoise,
                ),
            )

        total_frames = samples.shape[2]
        chunk_size = max(1, int(chunk_len))
        overlap = max(0, min(int(overlap_len), chunk_size - 1))

        if total_frames <= chunk_size:
            return (
                self._run_ksampler(
                    model,
                    base_noise_seed,
                    sample_steps,
                    guidance_cfg,
                    sampler,
                    schedule,
                    positive,
                    negative,
                    latent_image,
                    noise_denoise,
                ),
            )

        stride = chunk_size - overlap
        pieces = []
        start = 0
        chunk_index = 0

        while start < total_frames:
            end = min(total_frames, start + chunk_size)
            latent_chunk = self._slice_latent(latent_image, start, end)
            positive_chunk = self._slice_conditioning(positive, start, end, total_frames)
            negative_chunk = self._slice_conditioning(negative, start, end, total_frames)

            result = self._run_ksampler(
                model,
                base_noise_seed,
                sample_steps,
                guidance_cfg,
                sampler,
                schedule,
                positive_chunk,
                negative_chunk,
                latent_chunk,
                noise_denoise,
            )

            result_samples = result["samples"]
            if pieces and overlap > 0:
                result_samples = result_samples[:, :, overlap:]
            pieces.append(result_samples)

            if end >= total_frames:
                break
            start += stride
            chunk_index += 1

        output = copy.copy(latent_image)
        output["samples"] = torch.cat(pieces, dim=2)
        return (output,)


class WanChunkedVAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "decode_chunk_frames": ("INT", {"default": 16, "min": 1, "max": 512}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "latent/video"

    def decode(self, samples, vae, decode_chunk_frames):
        VAEDecode = _get_builtin_node("VAEDecode")

        latent_samples = samples["samples"]
        if latent_samples.ndim != 5:
            return VAEDecode().decode(vae, samples)

        total_frames = latent_samples.shape[2]
        decoded_chunks = []
        decoder = VAEDecode()

        for start, end in _chunk_ranges(total_frames, decode_chunk_frames):
            latent_chunk = _slice_latent_frames(samples, start, end)
            decoded = decoder.decode(vae, latent_chunk)[0]
            decoded_chunks.append(decoded)

        return (torch.cat(decoded_chunks, dim=0),)


class WanChunkedImageUpscaleWithModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "image": ("IMAGE",),
                "upscale_chunk_frames": ("INT", {"default": 8, "min": 1, "max": 512}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, upscale_model, image, upscale_chunk_frames):
        ImageUpscaleWithModel = _get_builtin_node("ImageUpscaleWithModel")

        upscaled_chunks = []
        upscaler = ImageUpscaleWithModel()

        for start, end in _chunk_ranges(image.shape[0], upscale_chunk_frames):
            chunk = image[start:end].contiguous()
            upscaled = upscaler.upscale(upscale_model, chunk)[0]
            upscaled_chunks.append(upscaled)

        return (torch.cat(upscaled_chunks, dim=0),)


class WanChunkedImageScale:
    @classmethod
    def INPUT_TYPES(cls):
        ImageScale = _get_builtin_node("ImageScale")

        return {
            "required": {
                "image": ("IMAGE",),
                "scale_chunk_frames": ("INT", {"default": 8, "min": 1, "max": 512}),
                "upscale_method": (ImageScale.upscale_methods,),
                "width": ("INT", {"default": 1080, "min": 0, "max": 16384}),
                "height": ("INT", {"default": 1920, "min": 0, "max": 16384}),
                "crop": (["disabled", "center"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, image, scale_chunk_frames, upscale_method, width, height, crop):
        ImageScale = _get_builtin_node("ImageScale")

        scaled_chunks = []
        scaler = ImageScale()

        for start, end in _chunk_ranges(image.shape[0], scale_chunk_frames):
            chunk = image[start:end].contiguous()
            scaled = scaler.upscale(chunk, upscale_method, width, height, crop)[0]
            scaled_chunks.append(scaled)

        return (torch.cat(scaled_chunks, dim=0),)


NODE_CLASS_MAPPINGS = {
    "WanLatentChunkSampler": WanLatentChunkSampler,
    "WanChunkedVAEDecode": WanChunkedVAEDecode,
    "WanChunkedImageUpscaleWithModel": WanChunkedImageUpscaleWithModel,
    "WanChunkedImageScale": WanChunkedImageScale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanLatentChunkSampler": "Wan Latent Chunk Sampler",
    "WanChunkedVAEDecode": "Wan Chunked VAE Decode",
    "WanChunkedImageUpscaleWithModel": "Wan Chunked Image Upscale",
    "WanChunkedImageScale": "Wan Chunked Image Scale",
}
