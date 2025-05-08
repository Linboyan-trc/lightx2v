########################################################################################################################
# 1. Register类
# 1.1 用于存储一系列函数名 -> 函数的映射
# 1.2 用于存储一系列类名 -> 类的映射
class Register(dict):
    # 1.1 初始化，设置_dict属性
    def __init__(self, *args, **kwargs):
        super(Register, self).__init__(*args, **kwargs)
        self._dict = {}

    # 1.2 重写了__setitem__，使得self[key] = ..., 等价于self._dict[key] = ...
    # 1.2.1 可以直接使用self[key]来设置键值对
    def __setitem__(self, key, value):
        self._dict[key] = value

    # 1.3 重写了__getitem__，使得self[key]等价于self._dict[key]
    # 1.3.1 可以直接使用self[key]来获取键值
    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __str__(self):
        return str(self._dict)
    
    # 1.4 使用@RUNNER_REGISTER开始注册函数/类
    def __call__(self, target_or_name):
        # 1.4.1 对于@RUNNER_REGISTER则进入if分支，因为直接callable
        if callable(target_or_name):
            return self.register(target_or_name)
        
        # 1.4.2 对于@RUNNER_REGISTE('str')则进入else分支，因为'str'不是callable，此时是用自定义键值名称，建立'str' -> 函数/类的映射
        # 1.4.3 lambda x是为了可以获取到@RUNNER_REGISTE('str')后面跟的函数，作为x，然后调用self.register()方法传入函数，和自定义的名字
        else:
            return lambda x: self.register(x, key=target_or_name)

    # 1.5 注册一组键值对映射
    # 1.5.1 值必须要是callable
    # 1.5.2 没有指定自定义的key的时候使用函数/类的.__name__
    # 1.5.3 注册键值不能重名
    # 1.5.4 添加到本self的._dict中
    # 1.5.5 返回callable，也就是返回@RUNNER_REGISTE('str')后面这个类或函数本身
    def register(self, target, key=None):
        if not callable(target):
            raise Exception(f"Error: {target} must be callable!")

        if key is None:
            key = target.__name__

        if key in self._dict:
            raise Exception(f"{key} already exists.")

        self[key] = target
        return target

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

########################################################################################################################
# 1. 通过注册器['func_name']来调用函数
# 1.1 共有注册器MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER, LN_WEIGHT_REGISTER, 分别是addmm, 均方根, 线性层
MM_WEIGHT_REGISTER = Register()
RMS_WEIGHT_REGISTER = Register()
LN_WEIGHT_REGISTER = Register()

# 1.2 共有注册器CONV2D_WEIGHT_REGISTER, CONV3D_WEIGHT_REGISTER
CONV3D_WEIGHT_REGISTER = Register()
CONV2D_WEIGHT_REGISTER = Register()

# 1.3 共有注册器RUNNER_REGISTER
# 1.3.1 自定义了@RUNNER_REGISTER("hunyuan") class HunyuanRunner()，使用RUNNER_REGISTER['hunyuan']()可调用这个类并构造一个实例化对象
# 1.3.2 自定义了@RUNNER_REGISTER("wan2.1") class WanRunner()，使用RUNNER_REGISTER['wan2.1']()可调用这个类并构造一个实例化对象
RUNNER_REGISTER = Register()
