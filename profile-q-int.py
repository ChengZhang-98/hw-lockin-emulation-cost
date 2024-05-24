import torch
import transformers
from quantize.quantized_functions.matmul import matmul_flexible
from quantize.quantized_layers.linear import LinearLoQER

import bitsandbytes as bnb


def set_layer_by_name(module, name, new_layer):
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = module
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_layer)
    else:
        setattr(module, name, new_layer)


def profile_matmul(model_name, batch_size, seq_len, num_iters, dtype, device):
    q_config = {
        "w_quantizer": {
            "name": "integer",
            "fp_min": -4.0,
            "fp_max": 4.0,
            "n_bits": 8,
            "is_affine": True,
        },
        "x_quantizer": {
            "name": "integer",
            "fp_min": -4.0,
            "fp_max": 4.0,
            "n_bits": 8,
            "is_affine": True,
        },
    }
    config = transformers.AutoConfig.from_pretrained(model_name)

    input1 = torch.randn(batch_size * seq_len, config.hidden_size, dtype=dtype).to(device)
    input2 = torch.randn(config.hidden_size, batch_size * seq_len, dtype=dtype).to(device)

    flops_per_iter = (batch_size * seq_len) ** 2 * (2 * config.hidden_size - 1)

    cuda_starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    cuda_ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    flops = flops_per_iter * num_iters

    for i in range(num_iters):
        cuda_starts[i].record()
        _ = matmul_flexible(input1, input2, q_config)
        cuda_ends[i].record()

    torch.cuda.synchronize()

    emulated_total_time = sum([cuda_starts[i].elapsed_time(cuda_ends[i]) for i in range(num_iters)])
    emulated_flops_per_sec = flops / (emulated_total_time / 1000)

    # int8

    for i in range(num_iters):
        cuda_starts[i].record()
        bnb.matmul(input1, input2.t(), threshold=10.0)
        cuda_ends[i].record()

    torch.cuda.synchronize()

    int8_total_time = sum([cuda_starts[i].elapsed_time(cuda_ends[i]) for i in range(num_iters)])
    int8_flops_per_sec = flops / (int8_total_time / 1000)

    results = {
        "model_name": model_name,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_iters": num_iters,
        "dtype": dtype,
        "device": device,
        "emulated total_time (ms)": emulated_total_time,
        "emulated flops_per_sec (TFLOPS)": emulated_flops_per_sec / 1e12,
        "int8_total_time (ms)": int8_total_time,
        "int8_flops_per_sec (TFLOPS)": int8_flops_per_sec / 1e12,
    }

    return results


@torch.no_grad()
def profile_opt_infer(model_name, batch_size, seq_len, num_iters, dtype, device):
    q_config = {
        "w_quantizer": {
            "name": "integer",
            "fp_min": -4.0,
            "fp_max": 4.0,
            "n_bits": 8,
            "is_affine": True,
        },
        "x_quantizer": {
            "name": "integer",
            "fp_min": -4.0,
            "fp_max": 4.0,
            "n_bits": 8,
            "is_affine": True,
        },
    }
    config = transformers.AutoConfig.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name).to(device=device, dtype=dtype)
    model.eval()

    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Linear):
            emulated_layer = LinearLoQER(
                layer.in_features,
                layer.out_features,
                bias=layer.bias is not None,
                q_config=q_config,
                device=next(layer.parameters()).device,
                dtype=next(layer.parameters()).dtype,
            )
            set_layer_by_name(model, name, emulated_layer)

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device=device)
    tokens_per_iter = batch_size * seq_len
    total_tokens = tokens_per_iter * num_iters

    cuda_starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    cuda_ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        cuda_starts[i].record()
        _ = model(input_ids)
        cuda_ends[i].record()

    torch.cuda.synchronize()

    emulated_int_total_time = sum([cuda_starts[i].elapsed_time(cuda_ends[i]) for i in range(num_iters)])
    emulated_int_tokens_per_sec = total_tokens / (emulated_int_total_time / 1000)

    del model

    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, load_in_8bit=True)

    for i in range(num_iters):
        cuda_starts[i].record()
        _ = model(input_ids)
        cuda_ends[i].record()

    torch.cuda.synchronize()

    int8_total_time = sum([cuda_starts[i].elapsed_time(cuda_ends[i]) for i in range(num_iters)])
    int8_tokens_per_sec = total_tokens / (int8_total_time / 1000)

    results = {
        "model_name": model_name,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_iters": num_iters,
        "dtype": dtype,
        "device": device,
        "emulated total_time (ms)": emulated_int_total_time,
        "emulated tokens_per_sec": emulated_int_tokens_per_sec,
        "int8_total_time (ms)": int8_total_time,
        "int8_tokens_per_sec": int8_tokens_per_sec,
    }

    return results


if __name__ == "__main__":
    from pprint import pprint

    model_name = "facebook/opt-2.7b"
    batch_size = 1
    seq_len = 2048 * 20
    num_iters = 100
    dtype = torch.float32
    device = "cuda"

    # warmup
    x = torch.randn(4096, 4096).to(device)
    for i in range(100):
        _ = torch.matmul(x, x)

    results = profile_matmul(model_name, batch_size, seq_len, num_iters, dtype, device)
    pprint(results, indent=4, sort_dicts=False)

    results = profile_opt_infer(model_name, batch_size, seq_len, num_iters, dtype, device)
    pprint(results, indent=4, sort_dicts=False)
