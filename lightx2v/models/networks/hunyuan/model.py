import os
import torch
from lightx2v.models.networks.hunyuan.weights.pre_weights import HunyuanPreWeights
from lightx2v.models.networks.hunyuan.weights.post_weights import HunyuanPostWeights
from lightx2v.models.networks.hunyuan.weights.transformer_weights import HunyuanTransformerWeights
from lightx2v.models.networks.hunyuan.infer.pre_infer import HunyuanPreInfer
from lightx2v.models.networks.hunyuan.infer.post_infer import HunyuanPostInfer
from lightx2v.models.networks.hunyuan.infer.transformer_infer import HunyuanTransformerInfer
from lightx2v.models.networks.hunyuan.infer.feature_caching.transformer_infer import HunyuanTransformerInferTaylorCaching, HunyuanTransformerInferTeaCaching

import lightx2v.attentions.distributed.ulysses.wrap as ulysses_dist_wrap
import lightx2v.attentions.distributed.ring.wrap as ring_dist_wrap


######################################################################################################################################################
# 1. HunyuanModel
class HunyuanModel:
    ############################## 1. HunyuanModel初始化: model_path, config, device, 三个权重实例，三个推理实例 ##############################
    # 1.1 指定权重类
    pre_weight_class = HunyuanPreWeights
    transformer_weight_class = HunyuanTransformerWeights
    post_weight_class = HunyuanPostWeights

    # 1. 初始化
    # 1.1 指定权重类，推理类
    # 1.2 实例化三个权重实例，三个推理实例
    def __init__(self, config, device):
        # 1.1 属性: model_path, config, device
        self.model_path = config.model_path
        self.config = config
        self.device = device

        # 1.2 初始化: 权重类，推理类，权重实例，推理实例
        self._init_infer_class()
        self._init_weights()
        self._init_infer()

        # 1.3 是否多卡并行
        if config["parallel_attn_type"]:
            if config["parallel_attn_type"] == "ulysses":
                ulysses_dist_wrap.parallelize_hunyuan(self)
            elif config["parallel_attn_type"] == "ring":
                ring_dist_wrap.parallelize_hunyuan(self)
            else:
                raise Exception(f"Unsuppotred parallel_attn_type")

        # 1.4 是否使用cpu
        if self.config["cpu_offload"]:
            self.to_cpu()

    # 1.2 指定推理类
    def _init_infer_class(self):
        self.pre_infer_class = HunyuanPreInfer
        if self.config["feature_caching"] == "NoCaching":
            self.transformer_infer_class = HunyuanTransformerInfer
        elif self.config["feature_caching"] == "TaylorSeer":
            self.transformer_infer_class = HunyuanTransformerInferTaylorCaching
        elif self.config["feature_caching"] == "Tea":
            self.transformer_infer_class = HunyuanTransformerInferTeaCaching
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {self.config['feature_caching']}")
        self.post_infer_class = HunyuanPostInfer

    # 1.3 实例化三个权重实例
    def _init_weights(self):
        # 1.2.1 加载weight_dict，weight_dict是一个字典，有856个键值对
        weight_dict = self._load_ckpt()
        # 1.2.2 实例化三个权重类，传入config
        self.pre_weight = self.pre_weight_class(self.config)
        self.transformer_weights = self.transformer_weight_class(self.config)
        self.post_weight = self.post_weight_class(self.config)
        # 1.2.3 调用权重类的.load_weights(weight_dict)加载权重
        self.pre_weight.load_weights(weight_dict)
        self.transformer_weights.load_weights(weight_dict)
        self.post_weight.load_weights(weight_dict)

    # 1.3.1 初始化权重，加载模型的transformers权重文件
    # 1.3.1 对于hunyuan来说，i2v只有一个26G的mp_rank_00_model_states.pt
    # 1.3.1 对于hunyuan来说，t2v只有三个24G的mp_rank_00_model_states.pt, 13G的mp_rank_00_model_states_fp8.pt, 102K的mp_rank_00_model_states_fp8_map.pt
    def _load_ckpt(self):
        # 1.2.1. 获取transformers权重文件的路径
        if self.config.task == "t2v":
            ckpt_path = os.path.join(self.model_path, "hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt")
        else:
            ckpt_path = os.path.join(self.model_path, "hunyuan-video-i2v-720p/transformers/mp_rank_00_model_states.pt")

        # 1.2.2 transformers权重文件mp_rank_00_model_states.pt在torch加载好之后是一个字典
        # 1.2.2 字典中只有键值:weights，而且由于weights_only=True，也只会加载{ "module" : ... }这个键值
        # 1.2.2 然后拿出其中"module"的键值，"module"对应的值也是一个字典，一共有856个键值对
        # 1.2.2 存储在weight_dict中
        weight_dict = torch.load(ckpt_path, map_location=self.device, weights_only=True)["module"]
        return weight_dict

    # 1.4 实例化三个推理实例
    def _init_infer(self):
        self.pre_infer = self.pre_infer_class(self.config)
        self.transformer_infer = self.transformer_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)

    ################################################## 2. 两个工具: to_cpu, to_cuda ##################################################
    def to_cpu(self):
        self.pre_weight.to_cpu()
        self.post_weight.to_cpu()
        self.transformer_weights.to_cpu()

    def to_cuda(self):
        self.pre_weight.to_cuda()
        self.post_weight.to_cuda()
        self.transformer_weights.to_cuda()

    ############################################################ 3. 推理 ############################################################
    # 1. 设置调度器
    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        self.pre_infer.set_scheduler(scheduler)
        self.transformer_infer.set_scheduler(scheduler)
        self.post_infer.set_scheduler(scheduler)
        
    # 2. infer计算
    @torch.no_grad()
    def infer(self, inputs):
        # 2.1 移动到GPU
        if self.config["cpu_offload"]:
            self.pre_weight.to_cuda()
            self.post_weight.to_cuda()

        # 2.2. pre, transformer, post轮流infer
        # 2.2.1 inputs首先是一个runner的属性，是一个字典，包含图像编码输出，文本编码输出
        # 2.2.1. pre_infer.infer()之后inputs为返回的一系列张量
        inputs = self.pre_infer.infer(self.pre_weight, inputs)

        # 2.2.2 pre_infer().infer()返回的一系列张量传递给下一层
        inputs = self.transformer_infer.infer(self.transformer_weights, *inputs)

        # 2.2.3 transformer_infer.infer()返回的一系列张量传递给下一层
        self.scheduler.noise_pred = self.post_infer.infer(self.post_weight, *inputs)

        # 2.3 移动回CPU
        if self.config["cpu_offload"]:
            self.pre_weight.to_cpu()
            self.post_weight.to_cpu()

        # 2.4 缓存复用
        if self.config["feature_caching"] == "Tea":
            self.scheduler.cnt += 1
            if self.scheduler.cnt == self.scheduler.num_steps:
                self.scheduler.cnt = 0
