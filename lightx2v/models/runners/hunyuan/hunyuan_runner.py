import os
import numpy as np
import torch
import torchvision
from PIL import Image
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.hunyuan.scheduler import HunyuanScheduler
from lightx2v.models.schedulers.hunyuan.feature_caching.scheduler import HunyuanSchedulerTaylorCaching, HunyuanSchedulerTeaCaching
from lightx2v.models.input_encoders.hf.llama.model import TextEncoderHFLlamaModel
from lightx2v.models.input_encoders.hf.clip.model import TextEncoderHFClipModel
from lightx2v.models.input_encoders.hf.llava.model import TextEncoderHFLlavaModel
from lightx2v.models.networks.hunyuan.model import HunyuanModel
from lightx2v.models.video_encoders.hf.autoencoder_kl_causal_3d.model import VideoEncoderKLCausal3DModel
import torch.distributed as dist
from lightx2v.utils.profiler import ProfilingContext


##################################################################################################################################
# 1. 在注册器RUNNER_REGISTER中注册'hunyuan' -> class HunyuanRunner()
@RUNNER_REGISTER("hunyuan")
class HunyuanRunner(DefaultRunner):
    ######################################## 1. __main__.py中声明并初始化一个HunyuanRunner实例 ########################################
    #################### 1.1 最终HunyuanRunner初始化好了属性config, image_encoder, text_encoders, model, vae_model ###################
    # 1.1 初始化
    # 1.1.1 首先调用父类DefaultRunner设置self.config = config
    # 1.1.2 然后父类会调用子类的.load_model()设置self.image_encoder, text_encoders, model, vae_model
    def __init__(self, config):
        super().__init__(config)

    # 1.2 初始化.load_model()
    @ProfilingContext("Load models")
    def load_model(self):
        # 1.2.1 如果多卡并行
        if self.config["parallel_attn_type"]:
            cur_rank = dist.get_rank()
            torch.cuda.set_device(cur_rank)

        # 1.2.2 如果加载到cpu
        if self.config.cpu_offload:
            init_device = torch.device("cpu")
        else:
            init_device = torch.device("cuda")

        # 1.2.3 先指定image_encoder为空
        image_encoder = None

        # 1.2.4 如果是t2v任务，第一个text_encoder就使用.../t2v/text_encoder下的Llama
        # 1.2.4 如果是i2v任务，第一个text_encoder就使用.../i2v/text_encoder_i2v下的Llava
        if self.config.task == "t2v":
            text_encoder_1 = TextEncoderHFLlamaModel(os.path.join(self.config.model_path, "text_encoder"), init_device)
        else:
            text_encoder_1 = TextEncoderHFLlavaModel(os.path.join(self.config.model_path, "text_encoder_i2v"), init_device)
        
        # 1.2.4.1 对于i2v和t2v任务，text_encoder2都使用.../xtv/text_encoder_2下的Clip
        # 1.2.4.2 然后text_encoders列表由text_encoder_1和text_encoder2组成
        text_encoder_2 = TextEncoderHFClipModel(os.path.join(self.config.model_path, "text_encoder_2"), init_device)
        text_encoders = [text_encoder_1, text_encoder_2]

        # 1.2.5 加载hunyuanModel
        model = HunyuanModel(self.config, init_device)

        # 1.2.6 加载VideoEncoderKLCausal3DModel
        vae_model = VideoEncoderKLCausal3DModel(self.config.model_path, dtype=torch.float16, device=init_device, config=self.config)

        return image_encoder, text_encoders, model, vae_model

    #############################################################################################################################
    ########## 1.2.1 初始化调度器 ##########
    def init_scheduler(self):
        if self.config.feature_caching == "NoCaching":
            scheduler = HunyuanScheduler(self.config)
        elif self.config.feature_caching == "Tea":
            scheduler = HunyuanSchedulerTeaCaching(self.config)
        elif self.config.feature_caching == "TaylorSeer":
            scheduler = HunyuanSchedulerTaylorCaching(self.config)
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {self.config.feature_caching}")
        self.model.set_scheduler(scheduler)

    ########## 1.2.2 图像编码 ##########
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

    ########## 1.2.2 文本编码 ##########
    def run_text_encoder(self, config, text_encoders, image_encoder_output):
        # 1.1 文本编码输出
        text_encoder_output = {}

        # 1.2 遍历text_encoders
        # 1.2.1 i是下标，encoder是具体的编码器
        for i, encoder in enumerate(text_encoders):
            # 1.2.2 对于图像任务 + 第一个编码器
            # 1.2.2.1 用文本 + 图像 + 配置，得到输出 + 注意力掩码
            # 1.2.2.2 都是多维张量
            if config.task == "i2v" and i == 0:
                text_state, attention_mask = encoder.infer(config.prompt, image_encoder_output["img"], config)
            
            # 1.2.3 对于文本任务 + 第一个编码器，文本任务 + 第二个编码器
            # 1.2.3.1 用文本 + 配置，得到输出 + 注意力掩码
            # 1.2.2.3 都是多维张量
            else:
                text_state, attention_mask = encoder.infer(config.prompt, config)
            
            # 1.2.4 存储到结果中
            text_encoder_output[f"text_encoder_{i + 1}_text_states"] = text_state.to(dtype=torch.bfloat16)
            text_encoder_output[f"text_encoder_{i + 1}_attention_mask"] = attention_mask
        
        # 1.2.5 返回文本编码结果
        # 1.2.5 一共有text_encoder_output["text_encoder_1_text_states"] = 多维张量
        # 1.2.5 一共有text_encoder_output["text_encoder_1_attention_mask"] = 多维张量
        # 1.2.5 一共有text_encoder_output["text_encoder_2_text_states"] = 多维张量
        # 1.2.5 一共有text_encoder_output["text_encoder_2_attention_mask"] = 多维张量
        return text_encoder_output

    ########## 1.2.2 总编码 ##########
    # 1. 设置config.target_shape
    # 1.1 (1, 16, 对帧数按4分组，对高度按8分组，对宽度按8分组)
    def set_target_shape(self):
        vae_scale_factor = 2 ** (4 - 1)
        self.config.target_shape = (
            1,
            16,
            (self.config.target_video_length - 1) // 4 + 1,
            int(self.config.target_height) // vae_scale_factor,
            int(self.config.target_width) // vae_scale_factor,
        )

    #############################################################################################################################
