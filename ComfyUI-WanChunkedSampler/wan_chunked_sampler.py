import copy

import torch


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
        chunk = copy.copy(latent)
        samples = latent["samples"]
        chunk["samples"] = samples[:, :, start:end].contiguous()

        for key in ("noise_mask", "batch_index"):
            value = latent.get(key)
            if isinstance(value, torch.Tensor) and value.ndim >= 3 and value.shape[2] == samples.shape[2]:
                chunk[key] = value[:, :, start:end].contiguous()

        return chunk

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

            result = self._run_ksampler(
                model,
                base_noise_seed + chunk_index,
                sample_steps,
                guidance_cfg,
                sampler,
                schedule,
                positive,
                negative,
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


NODE_CLASS_MAPPINGS = {
    "WanLatentChunkSampler": WanLatentChunkSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanLatentChunkSampler": "Wan Latent Chunk Sampler",
}
