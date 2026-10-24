"""Consistent analytical FLOP accounting for the project.

Convention:
- batch size 1, forward pass only, input 1x5x256x256, FP32
- fvcore convention for supported operators (one fused multiply-add = one FLOP)
- FFT pair estimate per spectral block = 10*C*N*log2(N)
- adjusted FLOPs = fvcore supported FLOPs + analytical FFT estimate
This is an analytical estimate, not an exact count of GPU instructions.
"""
import math
from fvcore.nn import FlopCountAnalysis


def fft_estimate(config, height=256, width=256):
    m = config["model"]
    if m["name"] not in {"fno", "ufno"}:
        return 0.0
    c = int(m["width"])
    depth = int(m["depth"])
    padding = int(m["padding"])
    n = (height + padding) * (width + padding)
    return depth * 10.0 * c * n * math.log2(n)


def measure_model(model, config, example_input):
    analysis = FlopCountAnalysis(model, example_input)
    analysis.unsupported_ops_warnings(False)
    analysis.uncalled_modules_warnings(False)
    supported = float(analysis.total())
    fft = fft_estimate(config, example_input.shape[-2], example_input.shape[-1])
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "parameters": int(parameters),
        "supported_flops": supported,
        "fft_estimate": fft,
        "adjusted_flops": supported + fft,
        "unsupported_ops": dict(analysis.unsupported_ops()),
    }
