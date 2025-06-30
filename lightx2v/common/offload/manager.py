import torch


# 1. transformer权重缓存
# 1.1 一共60层
class WeightStreamManager(object):
    # 1. 初始化
    def __init__(self):
        # 1.1 用于缓存权重
        # 1.1 active_weights[0]和active_weights[1]
        self.active_weights = [None for _ in range(2)]

        # 1.2 torch.cuda.Stream()
        # 1.2 torch.cuda.Stream()就是CPU和GPU之间的消息队列，CPU往消息队列里写东西，GPU去执行
        # 1.2.1 torch.cuda.Stream(priority=-1)和torch.cuda.Stream(priority=0)是两个分开的消息队列，-1的优先级更高，但并不意味值-1这个队列中的任务就一定先被执行，只是大概率先被执行
        self.compute_stream = torch.cuda.Stream(priority=-1)
        self.load_stream = torch.cuda.Stream(priority=0)

    # 2. 加载权重
    # 2.1 提供block_idx和blocks权重列表
    def prefetch_weights(self, block_idx, blocks_weights):
        # 2.1 放进优先级更低的消息队列
        with torch.cuda.stream(self.load_stream):
            # 2.2 如果active_weights[1]有东西，先放回cpu
            if self.active_weights[1] is not None:
                self.active_weights[1].to_cpu_sync()
            # 2.3 把新的权重迁移到gpu，然后赋值给active_weights[1]
            new_weights = blocks_weights[block_idx]
            new_weights.to_cuda_sync()
            self.active_weights[1] = new_weights

    # 3. 交换权重
    def swap_weights(self):
        # 3.1 同步等待
        # 3.1.1 通过.synchronize()阻塞等待
        # 3.1.2 直到compute_stream和load_stream中的任务都执行完，才会继续之后的代码
        self.compute_stream.synchronize()
        self.load_stream.synchronize()

        # 3.2 交换一下active_weights[0]和active_weights[1]
        self.active_weights[0], self.active_weights[1] = (
            self.active_weights[1],
            self.active_weights[0],
        )
