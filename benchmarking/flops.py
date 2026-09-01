import torch


def measure_flops(model, *args, **kwargs):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_flops=True,
    ) as prof:
        _ = model(*args, **kwargs)

    events = prof.events()
    flops = sum([int(evt.flops) for evt in events])

    return {
        "FLOPS": flops,
        "KFLOPS": flops / 1000,
        "MFLOPS": flops / 1000 / 1000,
        "GFLOPS": flops / 1000 / 1000 / 1000,
        "TFLOPS": flops / 1000 / 1000 / 1000 / 1000,
    }
