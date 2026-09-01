import torch


def measure_memory(model, *args, **kwargs):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
    ) as prof:
        _ = model(*args, **kwargs)

    events = prof.events()
    memory = sum([int(evt.device_memory_usage) for evt in events])

    return {
        "B": memory,
        "KB": memory / 1024,
        "MB": memory / 1024 / 1024,
        "GB": memory / 1024 / 1024 / 1024,
        "TB": memory / 1024 / 1024 / 1024 / 1024,
    }


def measure_peak_memory(model, *args, **kwargs):
    torch.cuda.reset_peak_memory_stats()
    _ = model(*args, **kwargs)

    memory = torch.cuda.max_memory_allocated()

    return {
        "B": memory,
        "KB": memory / 1024,
        "MB": memory / 1024 / 1024,
        "GB": memory / 1024 / 1024 / 1024,
        "TB": memory / 1024 / 1024 / 1024 / 1024,
    }
