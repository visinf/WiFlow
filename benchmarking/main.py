from benchmarking import measure_flops,measure_memory,measure_peak_memory,measure_time
from WIFlow.WIFlow_combined import WiFlowCombined
from WIFlow.WIFlow_rnn import RNNArgs
from WIFlow.csi_preprocessor import CSIQuotientPreprocessor
import torch
import json
import os
from datetime import datetime


if __name__ == "__main__":
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    model = WiFlowCombined( RNNArgs(antenna_count=16,
                        iters=4,# original 4
                        dim=32,
                        image_size= [128,168],
                        preprocessor=CSIQuotientPreprocessor, # TODO replace it through measurements from the best result from the preprocessor
                        align_output=3)).to(device)
    model.to(device)
    model.eval()
    
    # Create random CSI samples (batch_size, channels, height, width)
    # Adjust dimensions based on your model's expected input
    batch_size = 1
    csi1 = torch.randn(batch_size, model.args.antenna_count, model.args.carrier_count, 100, dtype=torch.complex64).to(device)
    csi2 = torch.randn(batch_size, model.args.antenna_count, model.args.carrier_count, 100, dtype=torch.complex64).to(device)
    results = {}
    # Measure time
    results["time"] = measure_time(model, warmup_it=100, measure_it=1000, csi1=csi1, csi2=csi2)
    results["flops"] = measure_flops(model, csi1, csi2)
    results["memory"] = measure_memory(model, csi1, csi2)
    results["peak_memory"] = measure_peak_memory(model, csi1, csi2)

    print(f"Mean time: {results['time']['mean']:.2f} ms")
    print(f"Std time: {results['time']['std']:.2f} ms")
    print(f"FLOPs: {results['flops']}")
    print(f"Memory: {results['memory']}")
    print(f"Peak Memory: {results['peak_memory']}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model.__class__.__name__
    report_dir = "benchmarking/reports"
    os.makedirs(report_dir, exist_ok=True)
    
    filename = f"{report_dir}/{model_name}_{timestamp}_iter{model.args.iters}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to: {filename}")
