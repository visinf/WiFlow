import numpy as np
import torch


def measure_time(model, warmup_it, measure_it, *args, **kwargs):
    times = []
    for _ in range(measure_it + warmup_it):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = model( **kwargs)

        end.record()
        torch.cuda.synchronize()

        times.append(start.elapsed_time(end))

    times = times[warmup_it:]

    mean = np.mean(times).item()
    std = np.std(times).item()

    return {"mean": mean, "std": std, "measurements": times}
