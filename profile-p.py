import torch
import transformers

from loqer.utils import get_layer_by_name, set_layer_by_name, get_layer_name


@torch.no_grad()
def profile_matmul(model_name, batch_size, seq_len, num_iters, device="cuda", dtype=torch.float16, sparsity=0.995):
    config = transformers.AutoConfig.from_pretrained(model_name)

    input1 = torch.randn(batch_size * seq_len, config.hidden_size, dtype=dtype).to(device)
    input2 = torch.randn(config.hidden_size, batch_size * seq_len, dtype=dtype).to(device)

    flops_per_iter = (batch_size * seq_len) ** 2 * (2 * config.hidden_size - 1)
    total_flops = flops_per_iter * num_iters

    cuda_starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    cuda_ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        cuda_starts[i].record()
        _ = torch.matmul(input1, input2)
        cuda_ends[i].record()

    torch.cuda.synchronize()

    fp16_total_time = sum([cuda_starts[i].elapsed_time(cuda_ends[i]) for i in range(num_iters)])
    fp16_flops_per_sec = total_flops / (fp16_total_time / 1000)

    input_1_sparse = input1.clone().flatten()
    mask = torch.randperm(input_1_sparse.numel())
    num_zeros = int(sparsity * input_1_sparse.numel())
    num_non_zeros = input_1_sparse.numel() - num_zeros
    mask = mask[:num_zeros]
    input_1_sparse[mask] = 0
    input_1_sparse = input_1_sparse.reshape(batch_size * seq_len, config.hidden_size).to_sparse_csr()

    cuda_starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    cuda_ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        cuda_starts[i].record()
        _ = torch.sparse.mm(input_1_sparse, input2)
        cuda_ends[i].record()

    torch.cuda.synchronize()

    fp16_sparse_total_time = sum([cuda_starts[i].elapsed_time(cuda_ends[i]) for i in range(num_iters)])
    fp16_sparse_flops_per_sec = total_flops / (fp16_sparse_total_time / 1000)

    cuda_starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    cuda_ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    input1_clone = input1.clone().flatten()
    input1_clone[mask] = 0
    input1_clone = input1_clone.reshape(batch_size * seq_len, config.hidden_size)
    zeros = torch.zeros_like(input1)

    for i in range(num_iters):
        cuda_starts[i].record()
        input1_tmp = torch.where(input1_clone == 0, zeros, input1)
        _ = torch.matmul(input1_tmp, input2)
        cuda_ends[i].record()

    torch.cuda.synchronize()

    fp16_sparse_emulated_total_time = sum([cuda_starts[i].elapsed_time(cuda_ends[i]) for i in range(num_iters)])
    fp16_sparse_emulated_flops_per_sec = total_flops / (fp16_sparse_emulated_total_time / 1000)

    results = {
        "what's this": "Dense FP16 Matmul vs Sparse FP16 Matmul (FC1 in MLP)",
        "device": device,
        "sparsity": sparsity,
        "dtype": f"{dtype}",
        "num_iters": num_iters,
        "num_non_zeros": num_non_zeros,
        "num_zeros": num_zeros,
        "model_name": model_name,
        "total_flops": total_flops,
        "fp16_dense_total_time": fp16_total_time,
        "fp16_sparse_total_time": fp16_sparse_total_time,
        "fp16_sparse_emulated_total_time": fp16_sparse_emulated_total_time,
        "fp16_dense_TFLOPS": fp16_flops_per_sec / 1e12,
        "fp16_sparse_TFLOPS": fp16_sparse_flops_per_sec / 1e12,
        "fp16_sparse_emulated_TFLOPS": fp16_sparse_emulated_flops_per_sec / 1e12,
    }

    return results


class SparseLinear(torch.nn.Linear):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, sparsity=None
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        assert sparsity is not None

        sparse_weight = self.weight.clone().flatten()
        mask = torch.randperm(sparse_weight.numel())
        mask = mask[: int(sparsity * sparse_weight.numel())]
        sparse_weight[mask] = 0
        sparse_weight = sparse_weight.reshape(out_features, in_features).to_sparse_csr()

        self.weight = torch.nn.Parameter(sparse_weight, requires_grad=False)

    def forward(self, input: torch.Tensor):
        input.squeeze_(0)
        output = torch.sparse.mm(input, self.weight.t())
        # if self.bias is not None:
        # output = torch.sparse.mm(input, self.weight.t()) + self.bias
        # else:
        # output = torch.sparse.mm(input, self.weight.t())
        input.unsqueeze_(0)
        return output

    @classmethod
    def from_fp16(cls, linear: torch.nn.Linear, sparsity=None):
        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            sparsity=sparsity,
        )


class EmulatedSpareLinear(torch.nn.Linear):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, sparsity=None
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        assert sparsity is not None

        mask = torch.randperm(self.weight.numel())
        mask = mask[: int(sparsity * self.weight.numel())].to(self.weight.device)

        self.mask = self.register_buffer("mask", mask)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight.flatten()
        weight[self.mask] = 0
        weight = weight.reshape(self.weight.shape)

        return torch.nn.functional.linear(input, weight, self.bias)

    @classmethod
    def from_fp16(cls, linear: torch.nn.Linear, sparsity):
        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            sparsity=sparsity,
        )


@torch.no_grad()
def profile_model(model_name, batch_size, seq_len, num_iters, device="cuda", dtype=torch.float16, sparsity=0.995):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, _attn_implementation="eager"
    ).to(device)
    model.eval()

    inputs = torch.randint(0, model.config.vocab_size, (batch_size, seq_len)).to(device)

    tokens_per_iter = batch_size * seq_len
    total_tokens = tokens_per_iter * num_iters

    cuda_starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    cuda_ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        cuda_starts[i].record()
        _ = model(inputs)
        cuda_ends[i].record()

    torch.cuda.synchronize()

    fp16_total_time = sum([cuda_starts[i].elapsed_time(cuda_ends[i]) for i in range(num_iters)])

    def prune_model(model, sparsity):
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.Linear):
                sp_linear = SparseLinear.from_fp16(layer, sparsity)
                set_layer_by_name(model, name, sp_linear)

    prune_model(model, sparsity)
    model.to(device)

    cuda_starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    cuda_ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        cuda_starts[i].record()
        _ = model(inputs)
        cuda_ends[i].record()

    torch.cuda.synchronize()

    fp16_sparse_total_time = sum([cuda_starts[i].elapsed_time(cuda_ends[i]) for i in range(num_iters)])

    def prune_model_emulated(model, sparsity):
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.Linear):
                sp_linear = EmulatedSpareLinear.from_fp16(layer, sparsity)
                set_layer_by_name(model, name, sp_linear)

    prune_model_emulated(model, sparsity)

    cuda_starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    cuda_ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        cuda_starts[i].record()
        _ = model(inputs)
        cuda_ends[i].record()

    torch.cuda.synchronize()

    fp16_sparse_emulated_total_time = sum([cuda_starts[i].elapsed_time(cuda_ends[i]) for i in range(num_iters)])

    results = {
        "what's this": "Dense FP16 vs Sparse FP16 (OPT inference)",
        "device": device,
        "sparsity": sparsity,
        "dtype": f"{dtype}",
        "model_name": model_name,
        "total_tokens": total_tokens,
        "fp16_dense_total_time": fp16_total_time,
        "fp16_sparse_total_time": fp16_sparse_total_time,
        "fp16_sparse_emulated_total_time": fp16_sparse_emulated_total_time,
        "fp16_dense_tokens_per_sec": total_tokens / (fp16_total_time / 1000),
        "fp16_sparse_tokens_per_sec": total_tokens / (fp16_sparse_total_time / 1000),
        "fp16_sparse_emulated_tokens_per_sec": total_tokens / (fp16_sparse_emulated_total_time / 1000),
    }
    return results


def test_sparse_linear():
    num_iters = 100

    ln = torch.nn.Linear(2048, 2048, device="cuda")

    ln_sparse = SparseLinear(2048, 2048, sparsity=0.99, device="cuda")

    x = torch.randn(4, 2048).cuda()

    cuda_starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    cuda_ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        cuda_starts[i].record()
        _ = ln(x)
        cuda_ends[i].record()

    torch.cuda.synchronize()

    fp16_total_time = sum([cuda_starts[i].elapsed_time(cuda_ends[i]) for i in range(num_iters)])

    cuda_starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    cuda_ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        cuda_starts[i].record()
        _ = ln_sparse(x)
        cuda_ends[i].record()

    torch.cuda.synchronize()

    fp16_sparse_total_time = sum([cuda_starts[i].elapsed_time(cuda_ends[i]) for i in range(num_iters)])

    print(f"FP16 Dense: {fp16_total_time}")
    print(f"FP16 Sparse: {fp16_sparse_total_time}")


if __name__ == "__main__":
    from pprint import pprint

    # test_example()

    model_name = "facebook/opt-2.7b"
    batch_size = 1
    seq_len = 2048
    num_iters = 80
    device = "cuda"
    sparsity = 0.9995
    dtype = torch.float32

    # matmul
    results = profile_matmul(model_name, batch_size, seq_len, num_iters, dtype=dtype, sparsity=sparsity, device=device)
    pprint(results, indent=4, sort_dicts=False)

    # model
    # test_sparse_linear()

    results = profile_model(model_name, batch_size, seq_len, num_iters, device, dtype=dtype, sparsity=sparsity)
    pprint(results, indent=4, sort_dicts=False)
