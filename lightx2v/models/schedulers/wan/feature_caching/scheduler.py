from lightx2v.models.schedulers.wan.scheduler import WanScheduler


class WanSchedulerTeaCaching(WanScheduler):
    def __init__(self, config):
        super().__init__(config)

    def clear(self):
        self.transformer_infer.clear()


class WanSchedulerTaylorCaching(WanScheduler):
    def __init__(self, config):
        super().__init__(config)

        pattern = [True, False, False, False]
        self.caching_records = (pattern * ((config.infer_steps + 3) // 4))[: config.infer_steps]
        self.caching_records_2 = (pattern * ((config.infer_steps + 3) // 4))[: config.infer_steps]

    def clear(self):
        self.transformer_infer.clear()


class WanSchedulerAdaCaching(WanScheduler):
    def __init__(self, config):
        super().__init__(config)

    def clear(self):
        self.transformer_infer.clear()


class WanSchedulerCustomCaching(WanScheduler):
    def __init__(self, config):
        super().__init__(config)

    def clear(self):
        self.transformer_infer.clear()


class WanSchedulerCustomCachingV2(WanScheduler):
    def __init__(self, config):
        super().__init__(config)

        steps = config.infer_steps
        warmup = int(steps * 0.1)
        pattern = [True, False, False, False]
        remain = steps - warmup
        pattern_seq = (pattern * ((remain + 3) // 4))[:remain]
        self.caching_records = [True] * warmup + pattern_seq
        self.caching_records_2 = [True] * warmup + pattern_seq

    def clear(self):
        self.transformer_infer.clear()
