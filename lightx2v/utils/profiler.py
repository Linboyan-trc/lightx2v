import time
import torch
from contextlib import ContextDecorator
from lightx2v.utils.envs import *

########################################################################################################################
# 1. _ProfilingContext类
# 1.1 仅本文件内部使用
# 1.2 使用方法有两种，with ProfilingContext("...")包裹一段代码段，代码段开头调用__enter__()开始计时，代码段结束调用__exit__结束计时并打印输出
# 1.3 使用方法有两种，@ProfilingContext("...")装饰一个函数，函数执行前调用__enter__()开始计时，函数执行完调用__exit__结束计时并打印输出
class _ProfilingContext(ContextDecorator):
    # 1.1 初始化，获得属性name
    def __init__(self, name):
        self.name = name

    # 1.2 
    def __enter__(self):
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start_time
        print(f"[Profile] {self.name} cost {elapsed:.6f} seconds")
        return False


class _NullContext(ContextDecorator):
    # Context manager without decision branch logic overhead
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

# 2. 导出类
# 2.1 使用with ProfilingContext("...")或@ProfilingContext("...")，则无论是否需要性能分析都会打印运行耗时
# 2.2 使用with ProfilingContext4Debug("...")或@ProfilingContext4Debug("...")，仅在需要性能分析时打印运行耗时
ProfilingContext = _ProfilingContext
ProfilingContext4Debug = _ProfilingContext if CHECK_ENABLE_PROFILING_DEBUG() else _NullContext
