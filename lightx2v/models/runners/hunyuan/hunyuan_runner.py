import os
import numpy as np
import torch
import torchvision
from PIL import Image
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.hunyuan.scheduler import HunyuanScheduler
from lightx2v.models.schedulers.hunyuan.feature_caching.scheduler import HunyuanSchedulerTaylorCaching, HunyuanSchedulerTeaCaching, HunyuanSchedulerAdaCaching, HunyuanSchedulerCustomCaching
from lightx2v.models.input_encoders.hf.llama.model import TextEncoderHFLlamaModel
from lightx2v.models.input_encoders.hf.clip.model import TextEncoderHFClipModel
from lightx2v.models.input_encoders.hf.llava.model import TextEncoderHFLlavaModel
from lightx2v.models.networks.hunyuan.model import HunyuanModel
from lightx2v.models.video_encoders.hf.autoencoder_kl_causal_3d.model import VideoEncoderKLCausal3DModel
import torch.distributed as dist
from lightx2v.utils.profiler import ProfilingContext


@RUNNER_REGISTER("hunyuan")
class HunyuanRunner(DefaultRunner):
    def __init__(self, config):
        super().__init__(config)

    @ProfilingContext("Load models")
    def load_model(self):
        if self.config["parallel_attn_type"]:
            cur_rank = dist.get_rank()
            torch.cuda.set_device(cur_rank)
        image_encoder = None
        if self.config.cpu_offload:
            init_device = torch.device("cpu")
        else:
            init_device = torch.device("cuda")

        if self.config.task == "t2v":
            text_encoder_1 = TextEncoderHFLlamaModel(os.path.join(self.config.model_path, "text_encoder"), init_device)
        else:
            text_encoder_1 = TextEncoderHFLlavaModel(os.path.join(self.config.model_path, "text_encoder_i2v"), init_device)
        text_encoder_2 = TextEncoderHFClipModel(os.path.join(self.config.model_path, "text_encoder_2"), init_device)
        text_encoders = [text_encoder_1, text_encoder_2]
        model = HunyuanModel(self.config.model_path, self.config, init_device, self.config)
        vae_model = VideoEncoderKLCausal3DModel(self.config.model_path, dtype=torch.float16, device=init_device, config=self.config)
        return model, text_encoders, vae_model, image_encoder

    def init_scheduler(self):
        if self.config.feature_caching == "NoCaching":
            scheduler = HunyuanScheduler(self.config)
        elif self.config.feature_caching == "Tea":
            scheduler = HunyuanSchedulerTeaCaching(self.config)
        elif self.config.feature_caching == "Taylor":
            scheduler = HunyuanSchedulerTaylorCaching(self.config)
        elif self.config.feature_caching == "Ada":
            scheduler = HunyuanSchedulerAdaCaching(self.config)
        elif self.config.feature_caching == "Custom":
            scheduler = HunyuanSchedulerCustomCaching(self.config)
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {self.config.feature_caching}")
        self.model.set_scheduler(scheduler)

    def run_text_encoder(self, text, text_encoders, config, image_encoder_output):
        text_encoder_output = {}
        for i, encoder in enumerate(text_encoders):
            if config.task == "i2v" and i == 0:
                text_state, attention_mask = encoder.infer(text, image_encoder_output["img"], config)
            else:
                text_state, attention_mask = encoder.infer(text, config)
            text_encoder_output[f"text_encoder_{i + 1}_text_states"] = text_state.to(dtype=torch.bfloat16)
            text_encoder_output[f"text_encoder_{i + 1}_attention_mask"] = attention_mask
        return text_encoder_output

    def get_closest_ratio(self, height: float, width: float, ratios: list, buckets: list):
        aspect_ratio = float(height) / float(width)
        diff_ratios = ratios - aspect_ratio

        if aspect_ratio >= 1:
            indices = [(index, x) for index, x in enumerate(diff_ratios) if x <= 0]
        else:
            indices = [(index, x) for index, x in enumerate(diff_ratios) if x > 0]

        closest_ratio_id = min(indices, key=lambda pair: abs(pair[1]))[0]
        closest_size = buckets[closest_ratio_id]
        closest_ratio = ratios[closest_ratio_id]

        return closest_size, closest_ratio

    def generate_crop_size_list(self, base_size=256, patch_size=32, max_ratio=4.0):
        num_patches = round((base_size / patch_size) ** 2)
        assert max_ratio >= 1.0
        crop_size_list = []
        wp, hp = num_patches, 1
        while wp > 0:
            if max(wp, hp) / min(wp, hp) <= max_ratio:
                crop_size_list.append((wp * patch_size, hp * patch_size))
            if (hp + 1) * wp <= num_patches:
                hp += 1
            else:
                wp -= 1
        return crop_size_list

    def run_image_encoder(self, config, image_encoder, vae_model):
        img = Image.open(config.image_path).convert("RGB")

        if config.i2v_resolution == "720p":
            bucket_hw_base_size = 960
        elif config.i2v_resolution == "540p":
            bucket_hw_base_size = 720
        elif config.i2v_resolution == "360p":
            bucket_hw_base_size = 480
        else:
            raise ValueError(f"config.i2v_resolution: {config.i2v_resolution} must be in [360p, 540p, 720p]")

        origin_size = img.size

        crop_size_list = self.generate_crop_size_list(bucket_hw_base_size, 32)
        aspect_ratios = np.array([round(float(h) / float(w), 5) for h, w in crop_size_list])
        closest_size, closest_ratio = self.get_closest_ratio(origin_size[1], origin_size[0], aspect_ratios, crop_size_list)

        config.target_height, config.target_width = closest_size

        resize_param = min(closest_size)
        center_crop_param = closest_size

        ref_image_transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(resize_param), torchvision.transforms.CenterCrop(center_crop_param), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.5], [0.5])]
        )

        semantic_image_pixel_values = [ref_image_transform(img)]
        semantic_image_pixel_values = torch.cat(semantic_image_pixel_values).unsqueeze(0).unsqueeze(2).to(torch.float16).to(torch.device("cuda"))

        img_latents = vae_model.encode(semantic_image_pixel_values, config).mode()

        scaling_factor = 0.476986
        img_latents.mul_(scaling_factor)

        return {"img": img, "img_latents": img_latents}

    def set_target_shape(self):
        vae_scale_factor = 2 ** (4 - 1)
        self.config.target_shape = (
            1,
            16,
            (self.config.target_video_length - 1) // 4 + 1,
            int(self.config.target_height) // vae_scale_factor,
            int(self.config.target_width) // vae_scale_factor,
        )
