import gc
import torch
import torch.distributed as dist
from lightx2v.utils.profiler import ProfilingContext4Debug, ProfilingContext
from lightx2v.utils.utils import save_videos_grid, cache_video
from lightx2v.utils.envs import *


######################################################################################################################################################
# 1. 注册器RUNNER_REGISTER中'hunyuan' -> class HunyuanRunner(), 'wan2.1' -> class WanRunner()的父类
class DefaultRunner:
    ################################################## 1. HunyuanRunner初始化 ############################################################
    # 1.1 初始化
    # 1.1.1 根据__main__.py中传入的config，设置config
    # 1.1.2 虽然此父类中没有声明.load_model()的接口，但子类中实现了这个方法，父类会调用子类中的方法，设置image_encoder, text_encoders, model, vae_model
    def __init__(self, config):
        self.config = config
        self.image_encoder, self.text_encoders, self.model, self.vae_model = self.load_model()

    ################################################# 2. HunyuanRunner.run_pipeline()推理 #################################################
    # 1.2 推理
    def run_pipeline(self):
        #################### 1.2.1 初始化调度器，调用子类init_scheduler() #################
        self.init_scheduler()

        ############### 1.2.2 图像/文本编码，父类run_input_encoder() ##############
        # 1.1 设置了HunyuanRunenr的inputs
        self.run_input_encoder()

        ########### 1.2.3 network层，实例属性model = HunyuanModel ###########
        # 1.1 单独拿出inputs中的图像编码层输出，让调度器准备
        # 1.1 设置了调度器的image_encoder_output, latents, guidance, freqs_cos, freqs_sin
        self.model.scheduler.prepare(self.inputs["image_encoder_output"])

        # 1.2 开始infer_steps次推理
        latents, generator = self.run()

        # 1.3 针对CPU内存清理
        self.end_run()

        ################### 1.2.4 视频编码，父类run_vae() ###################
        images = self.run_vae(latents, generator)

        ######################### 1.2.5 保存视频，父类save_video() ###########################
        self.save_video(images)

    ########## 1.2.2 图像/文本编码 ##########
    def run_input_encoder(self):
        # 1.1 设置图像编码层输出
        image_encoder_output = None

        # 1.2 图像任务
        # 1.2.1 子类的run_image_encoder()
        # 1.2.2 此时HunyuanRunner.config有，image_encoder是None直接传入，vae_model有
        # 1.2.3 图像编码层输出image_encoder_output["img"]是图片
        # 1.2.3 图像编码层输出image_encoder_output["img_latents"]是一个巨大的多维张量
        if self.config["task"] == "i2v":
            with ProfilingContext("Run Img Encoder"):
                image_encoder_output = self.run_image_encoder(self.config, self.image_encoder, self.vae_model)
        
        # 1.3 文本任务
        # 1.3.1 子类的run_text_encoder()
        # 1.3.2 此时HunyuanRunner.config有，HunyuanRunner.text_encoders有，图像编码层输出需要有/没有就是None
        # 1.3.3 文本编码层输出text_encoder_output["text_encoder_1_text_states"] = [1,256,4096]，一句话，256个单词，每个词向量4096维
        # 1.3.3 文本编码层输出text_encoder_output["text_encoder_1_attention_mask"] = [1,256]，掩码，256个1或0
        # 1.3.3 文本编码层输出text_encoder_output["text_encoder_2_text_states"] = 1,256,4096]，一句话，256个单词，每个词向量4096维
        # 1.3.3 文本编码层输出text_encoder_output["text_encoder_2_attention_mask"] = [1,256]，掩码，256个1或0
        with ProfilingContext("Run Text Encoder"):
            text_encoder_output = self.run_text_encoder(self.config, self.text_encoders, image_encoder_output)

        # 1.4 设置HunyuanRuner的config.target_shape
        # 1.4.1 设置为config.target_shape = (1, 16, 对帧数按4分组，对高度按8分组，对宽度按8分组)，为一个五元组
        self.set_target_shape()

        # 1.5 设置HunyuanRunner的属性inputs
        # 1.5.1 为self.inputs["image_encoder_output"]["img"], ["img_latents"]
        # 1.5.1 为self.inputs["text_encoder_output"]["text_encoder_1_text_states"], ["text_encoder_1_attention_mask"], ["text_encoder_2_text_states"], ["text_encoder_2_attention_mask"]
        self.inputs = {"image_encoder_output": image_encoder_output, "text_encoder_output": text_encoder_output}

        # 1.6 清理CPU、GPU的内存
        gc.collect()
        torch.cuda.empty_cache()

    ########## 1.2.3 network层 ##########
    # 1.1 开始network层推理
    def run(self):
        # 1.1 推理次数为infer_steps
        for step_index in range(self.model.scheduler.infer_steps):
            # 1.2 打印推理infer_steps
            print(f"==> step_index: {step_index + 1} / {self.model.scheduler.infer_steps}")

            # 1.3 pre计算
            # 1.3.1 设置model.scheduler的step_index = 当前推理步数索引，model.scheduler的latents转bf16
            with ProfilingContext4Debug("step_pre"):
                self.model.scheduler.step_pre(step_index=step_index)

            # 1.4 infer/transformer计算
            with ProfilingContext4Debug("infer"):
                self.model.infer(self.inputs)

            # 1.5 post计算
            with ProfilingContext4Debug("step_post"):
                self.model.scheduler.step_post()

        # 1.6 返回一个巨大的多维张量调度器的latents，和调度器的随机张量生成器
        return self.model.scheduler.latents, self.model.scheduler.generator

    def end_run(self):
        if self.config.cpu_offload:
            self.model.scheduler.clear()
            del self.inputs, self.model.scheduler, self.model, self.text_encoders
            torch.cuda.empty_cache()

    # 1.3 图模式用
    def run_step(self, step_index=0):
        self.init_scheduler()
        self.run_input_encoder()
        self.model.scheduler.prepare(self.inputs["image_encoder_output"])
        self.model.scheduler.step_pre(step_index=step_index)
        self.model.infer(self.inputs)
        self.model.scheduler.step_post()

    ########## 1.2.4 视频编码 ##########
    @ProfilingContext("Run VAE")
    def run_vae(self, latents, generator):
        images = self.vae_model.decode(latents, generator=generator, config=self.config)
        return images

    ########## 1.2.5 保存视频 ##########
    @ProfilingContext("Save video")
    def save_video(self, images):
        if not self.config.parallel_attn_type or (self.config.parallel_attn_type and dist.get_rank() == 0):
            if self.config.model_cls == "wan2.1":
                cache_video(tensor=images, save_file=self.config.save_video_path, fps=16, nrow=1, normalize=True, value_range=(-1, 1))
            else:
                save_videos_grid(images, self.config.save_video_path, fps=24)
