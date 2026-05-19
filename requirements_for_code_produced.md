## Requirements for Code Produced

You must implement the required neural-network, optimization, and RLHF components from scratch.

In this document:

1. **RLHF** means reinforcement learning from human feedback.
2. **PPO** means proximal policy optimization.
3. **Submitted implementation** means the code path that computes model forward passes, losses, gradients, optimizer updates, inference, generation, sampling, log-probabilities, entropy, KL divergence, or RLHF/PPO objectives. Data preprocessing, logging, plotting, and independent tests are governed separately below.

Test files, sanity-check scripts, and numerical-equivalence checks may use otherwise-forbidden PyTorch utilities only for independent comparison, provided that the submitted implementation does not import, call, or depend on them.

### Main Rule

You may use PyTorch as:

1. a tensor library,
2. an automatic differentiation engine,
3. a device and dtype management system,
4. a module and parameter registration framework,
5. a performance, tracing, checkpointing, and mixed-precision framework.

You may not use PyTorch or another machine-learning library to provide a component that the assignment asks you to implement.

If the assignment requires a layer, activation, loss, normalization operation, dropout, attention mechanism, initializer, optimizer, learning-rate schedule, sampling method, log-probability computation, entropy computation, KL-divergence computation, or RLHF/PPO computation, you must implement it manually using elementary tensor operations.

Ordinary Python control flow, Python math, type annotations, dataclasses, containers, and scalar bookkeeping are allowed. The restrictions apply to library calls that compute tensor operations or provide assignment-required machine-learning behavior.

When in doubt, implement the operation yourself.

### Import and Alias Rule

Restrictions apply to the underlying object, not to the spelling used to access it.

For example, all of the following are forbidden if they refer to the same forbidden prebuilt function:

```python
torch.nn.functional.softmax(...)
F.softmax(...)
from torch.nn.functional import softmax
```

Allowed objects may be imported directly or accessed through aliases. For example, these are equivalent and allowed:

```python
torch.nn.Parameter
nn.Parameter
from torch.nn import Parameter
```

### Allowed `torch.nn` Objects

You may use only the following objects from `torch.nn`:

1. `torch.nn.Parameter`
2. `torch.nn.Module`
3. `torch.nn.ModuleList`
4. `torch.nn.ModuleDict`
5. `torch.nn.ParameterList`
6. `torch.nn.ParameterDict`
7. `torch.nn.Sequential`

You may use normal `torch.nn.Module` methods, including:

1. `parameters()`
2. `named_parameters()`
3. `register_parameter()`
4. `register_buffer()`
5. `buffers()`
6. `named_buffers()`
7. `train()`
8. `eval()`
9. `to()`
10. `state_dict()`
11. `load_state_dict()`

Initializer exception:

1. `torch.nn.init.trunc_normal_` may be used for assignment-required truncated
   normal parameter initialization.

### Forbidden `torch.nn` and `torch.nn.functional` Objects

You may not use any other `torch.nn` class, function, or object.

In particular, do not use:

1. `torch.nn.Linear`
2. `torch.nn.Embedding`
3. `torch.nn.Conv1d`, `torch.nn.Conv2d`, or `torch.nn.Conv3d`
4. `torch.nn.RNN`, `torch.nn.LSTM`, or `torch.nn.GRU`
5. `torch.nn.MultiheadAttention`
6. `torch.nn.ReLU`, `torch.nn.GELU`, `torch.nn.Sigmoid`, `torch.nn.Tanh`, or `torch.nn.Softmax`
7. `torch.nn.Dropout`
8. `torch.nn.BatchNorm1d`, `torch.nn.BatchNorm2d`, `torch.nn.LayerNorm`, or `torch.nn.GroupNorm`
9. `torch.nn.CrossEntropyLoss`, `torch.nn.NLLLoss`, `torch.nn.MSELoss`, or any other loss class
10. `torch.nn.Flatten`, `torch.nn.Identity`, or other convenience modules
11. `torch.nn.init` or any initializer from `torch.nn.init`, except for
    `torch.nn.init.trunc_normal_` as allowed above
12. `torch.nn.utils`, including but not limited to:
    - `torch.nn.utils.clip_grad_norm_`
    - `torch.nn.utils.clip_grad_value_`
    - `torch.nn.utils.weight_norm`
    - `torch.nn.utils.spectral_norm`
    - `torch.nn.utils.parametrize`

You may not use anything from `torch.nn.functional`.

This includes, but is not limited to:

1. `torch.nn.functional.linear`
2. `torch.nn.functional.embedding`
3. `torch.nn.functional.conv1d`, `torch.nn.functional.conv2d`, or `torch.nn.functional.conv3d`
4. `torch.nn.functional.relu`, `torch.nn.functional.gelu`, `torch.nn.functional.sigmoid`, or `torch.nn.functional.tanh`
5. `torch.nn.functional.softmax` or `torch.nn.functional.log_softmax`
6. `torch.nn.functional.dropout`
7. `torch.nn.functional.batch_norm`, `torch.nn.functional.layer_norm`, or `torch.nn.functional.group_norm`
8. `torch.nn.functional.cross_entropy`, `torch.nn.functional.nll_loss`, or `torch.nn.functional.mse_loss`
9. `torch.nn.functional.scaled_dot_product_attention`
10. any other function defined in `torch.nn.functional`

### Optimizer and Scheduler Restrictions

You may use `torch.optim.Optimizer` only as a base class for writing your own optimizer.

You may use its infrastructure, including:

1. parameter groups,
2. optimizer state dictionaries,
3. `state_dict()`,
4. `load_state_dict()`,
5. `add_param_group()`,
6. `zero_grad()`.

You must manually implement the actual update rule in `step()` or in your training loop.

Do not use prebuilt optimizers or schedulers, including:

1. `torch.optim.SGD`
2. `torch.optim.Adam`
3. `torch.optim.AdamW`
4. `torch.optim.RMSprop`
5. `torch.optim.Adagrad`
6. `torch.optim.lr_scheduler` or any learning-rate scheduler provided by PyTorch

Learning-rate schedules must be implemented manually.

### Distribution and RLHF Restrictions

You may not use `torch.distributions` in submitted code.

Do not use:

1. `torch.distributions.Categorical`
2. `torch.distributions.Distribution`
3. `torch.distributions.kl_divergence`
4. `distribution.sample()`
5. `distribution.log_prob()`
6. `distribution.entropy()`

For RLHF, PPO, policy-gradient, sampling, log-probability, entropy, or KL-divergence computations, you must implement the math manually using elementary tensor operations.

### Forbidden High-Level PyTorch Functions

Do not use high-level PyTorch functions that directly implement required neural-network, optimization, or RLHF components.

In particular, do not use:

1. `torch.softmax`
2. `Tensor.softmax`
3. `torch.log_softmax`
4. `Tensor.log_softmax`
5. `torch.logsumexp`
6. `Tensor.logsumexp`
7. `torch.dropout`
8. `torch.special.softmax`
9. `torch.special.log_softmax`
10. `torch.special.logsumexp`
11. `torch.special.entr`

The same rule applies to aliases, wrappers, private APIs, and equivalent functions in other namespaces.

### Private and Low-Level PyTorch APIs

Do not use private, internal, generated, or low-level PyTorch namespaces to bypass these rules.

Do not call forbidden operations through:

1. `torch.ops`
2. `torch.ops.aten`
3. `torch.ops.prims`
4. `torch._C`
5. `torch._VF`
6. `torch._refs`
7. `torch._prims`
8. `torch._decomp`
9. any other underscore-prefixed PyTorch namespace

Private or low-level aliases of forbidden operations are forbidden even if they are not explicitly named elsewhere in this document.

### Allowed Elementary Tensor Operations

You may use top-level `torch` functions and Tensor methods for elementary tensor computation.

The list below is the allowlist for PyTorch tensor operations in the submitted implementation. Entries written without a namespace, such as `reshape` or `sum`, may be used either as Tensor methods or as equivalent top-level `torch` functions when PyTorch provides both forms, unless the operation is explicitly forbidden elsewhere in this document.

Allowed categories are:

1. Tensor creation:
   - `torch.tensor`
   - `torch.empty`
   - `torch.zeros`
   - `torch.ones`
   - `torch.full`
   - `torch.arange`
   - `torch.linspace`
   - `torch.rand`
   - `torch.randn`
   - `torch.randint`
   - `torch.zeros_like`
   - `torch.ones_like`
   - `torch.empty_like`
   - `torch.full_like`
   - `torch.rand_like`
   - `torch.randn_like`

2. Shape operations:
   - `reshape`
   - `view`
   - `Tensor.T`
   - `transpose`
   - `permute`
   - `movedim`
   - `unsqueeze`
   - `squeeze`
   - `expand`
   - `expand_as`
   - `repeat`
   - `repeat_interleave`
   - `split`
   - `chunk`
   - `unbind`
   - Tensor-method `flatten`
   - `contiguous`
   - `torch.cat`
   - `torch.stack`

3. Shape, dtype, device, and parameter introspection:
   - `Tensor.shape`
   - `Tensor.size()`
   - `Tensor.dim()`
   - `Tensor.ndim`
   - `Tensor.numel()`
   - `Tensor.dtype`
   - `Tensor.device`
   - `Tensor.requires_grad`
   - `Parameter.grad`
   - `Parameter.requires_grad`

4. Indexing and masking:
   - slicing
   - integer indexing
   - boolean masking
   - boolean operators `&`, `|`, and `~`
   - `gather`
   - `scatter`
   - `index_select`
   - `where`
   - `masked_fill`

5. Arithmetic and linear algebra:
   - `+`, `-`, `*`, `/`, `**`, and `@`
   - comparison operators such as `==`, `!=`, `<`, `<=`, `>`, and `>=`
   - `torch.matmul`
   - `torch.mm`
   - `torch.bmm`
   - `torch.einsum`
   - `torch.dot`
   - `torch.outer`

6. Reductions:
   - `sum`
   - `mean`
   - `var`
   - `std`
   - `max`
   - `min`
   - `all`
   - `any`
   - `argmax`
   - `argmin`

7. Elementary math:
   - `exp`
   - `log`
   - `sqrt`
   - `rsqrt`
   - `sin`
   - `cos`
   - `abs`
   - `clamp`
   - `maximum`
   - `minimum`

8. Explicitly allowed scalar nonlinearities:
   - `torch.tanh`
   - `Tensor.tanh`
   - `torch.sigmoid`
   - `Tensor.sigmoid`

   These are allowed as elementary scalar functions unless the assignment specifically asks you to implement tanh or sigmoid themselves.

9. Random and sampling primitives:
   - `torch.rand`
   - `torch.randn`
   - `torch.randint`
   - `torch.bernoulli`
   - `torch.multinomial`

   `torch.multinomial` may only be used to draw indices from probabilities that you computed manually.

10. Sorting, ranking, and cumulative operations:
   - `torch.sort`
   - `torch.argsort`
   - `torch.topk`
   - `torch.cumsum`
   - `torch.cumprod`

   These may be used to implement manual top-k or top-p filtering. They may not be used through a higher-level sampling, distribution, or generation helper.

11. Mask construction, debugging, and logging helpers:
    - `torch.eye`
    - `torch.tril`
    - `torch.triu`
    - causal and padding masks built from `torch.arange`, comparison operators, `torch.tril`, `torch.triu`, and broadcasting
    - `torch.isfinite`
    - `torch.isnan`
    - `torch.isinf`
    - `Tensor.item`, for logging only
    - `Tensor.tolist`, for logging only

12. Device, dtype, and autograd operations:
    - `to`
    - `cuda`
    - `cpu`
    - `Tensor.float()`
    - `Tensor.long()`
    - `Tensor.bool()`
    - dtype constants: `torch.float`, `torch.float16`, `torch.float32`, `torch.float64`, `torch.double`, `torch.bfloat16`, `torch.long`, `torch.int`, `torch.int32`, `torch.int64`, and `torch.bool`
    - `torch.device`
    - `torch.cuda.is_available`
    - `detach`
    - `clone`
    - `backward`
    - `requires_grad_`
    - `torch.no_grad`
    - `torch.enable_grad`
    - `torch.autograd.grad`

13. In-place assignment, initialization, and update primitives:
    - `copy_`
    - `zero_`
    - `fill_`
    - `uniform_`
    - `normal_`
    - `add_`
    - `sub_`
    - `mul_`
    - `div_`
    - `clamp_`
    - `torch.manual_seed`
    - `torch.Generator`

Use in-place parameter initialization and manual parameter updates under `torch.no_grad()`. Do not use `.data` to mutate parameters.

PyTorch functions, Tensor methods, and PyTorch library objects not permitted by this document are not allowed in the submitted implementation unless the assignment or instructor explicitly permits them. If a desired helper is not listed, write the operation from listed primitives instead.

### Notes on Normalization and Reductions

`torch.mean`, `Tensor.mean`, `torch.var`, `Tensor.var`, `torch.std`, and `Tensor.std` are permitted in manual normalization implementations because they are elementary reductions, not normalization primitives.

PyTorch's `torch.var` defaults to `unbiased=True`. Layer normalization and batch normalization use the biased population variance, so pass the appropriate argument when implementing these manually.

### Gradient Clipping

If gradient clipping is required, implement it manually.

You may use `Tensor.norm`, `torch.norm`, or `torch.linalg.vector_norm` only to compute gradient norms.

You must manually implement:

1. threshold comparison,
2. clipping coefficient calculation,
3. in-place gradient rescaling or clamping.

Do not use `torch.nn.utils.clip_grad_norm_` or `torch.nn.utils.clip_grad_value_`.

### Required Manual Implementations

If required by the assignment, implement the following manually:

1. Linear layer:
   Use explicit weight and bias parameters with matrix multiplication.

2. Embedding layer:
   Use explicit embedding parameters and indexing.

3. ReLU:
   Implement using elementary operations such as comparison, multiplication, `where`, or `maximum`.

4. GELU:
   Implement the formula specified by the assignment. If no variant is specified, use the tanh approximation: `0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x ** 3)))`.

5. Softmax:
   Implement using max-subtraction, exponentiation, summation, and division.

6. Log-softmax:
   Implement manually. Do not use `log_softmax` or `logsumexp`.

7. Cross-entropy loss:
   Implement manually from logits and labels. Do not use prebuilt cross-entropy, negative-log-likelihood, softmax, or log-softmax functions.

8. Normalization:
   Implement manually using mean, variance, epsilon stabilization, scale, and shift.

9. Dropout:
   Implement manually by sampling a mask and applying train-time scaling.

10. Attention:
    Implement manually using matrix multiplication, masking, manual softmax, and weighted summation.

11. Parameter initialization:
    Implement manually using random tensor operations and in-place assignment under `torch.no_grad()`.

12. Optimizer updates:
    Implement manually inside your own optimizer class or training loop.

13. Learning-rate schedules:
    Implement manually.

14. Categorical sampling:
    Manually compute logits, temperature scaling, masks, top-k/top-p filtering if required, probabilities, and sampled indices.

15. Log-probabilities:
    Implement manually using indexing or `gather` and a manually implemented log-softmax or equivalent.

16. Entropy:
    Implement manually from probabilities and log-probabilities.

17. KL divergence:
    Implement manually from probabilities and log-probabilities.

18. PPO/RLHF objectives:
    Implement manually from the mathematical definitions required by the assignment.

### Permitted Infrastructure

You may use the following PyTorch infrastructure as long as it does not replace a required implementation:

1. `torch.utils.data.Dataset`
2. `torch.utils.data.DataLoader`
3. `torch.save`
4. `torch.load`
5. `torch.compile`
6. `torch.jit`
7. `torch.fx`
8. `torch.profiler`
9. `torch.amp`
10. `torch.cuda.amp`

Performance, tracing, compilation, profiling, checkpointing, and mixed-precision utilities are permitted, but they must not replace or hide the implementation of any required mathematical component.

Compilation, tracing, and mixed-precision utilities operate on your from-scratch source code at execution time and are permitted. The source code, not the compiled artifact, must satisfy the from-scratch requirement.

`torch.amp.autocast`, `torch.cuda.amp.autocast`, `torch.amp.GradScaler`, and `torch.cuda.amp.GradScaler` are permitted as mixed-precision infrastructure.

Gradient scaling and unscaling for numerical stability are permitted. However, `GradScaler` must not be used to hide or replace a required manual optimizer update or manual gradient-clipping implementation. If gradient clipping is used with AMP, unscale gradients before manually clipping them.

### Other Libraries

You may use Python standard-library modules, NumPy, pandas, matplotlib, or similar non-machine-learning libraries for:

1. data processing,
2. logging,
3. testing,
4. plotting,
5. numerical checking,
6. experiment management.

You may not use another machine-learning or numerical framework to implement submitted model, loss, optimizer, gradient-clipping, sampling, log-probability, entropy, KL-divergence, PPO, or RLHF computations.

Do not implement submitted mathematical components in:

1. NumPy,
2. JAX,
3. TensorFlow,
4. scikit-learn,
5. Hugging Face Transformers,
6. TRL,
7. Accelerate,
8. any other external machine-learning framework.

These libraries may be used only outside the submitted training or inference code path for preprocessing, logging, plotting, or independent numerical checks.

### Examples

Allowed:

```python
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.empty(out_features))
        with torch.no_grad():
            self.weight.normal_(0.0, 0.02)
            self.bias.zero_()

    def forward(self, x):
        return x @ self.weight.T + self.bias
```

Forbidden:

```python
self.linear = torch.nn.Linear(in_features, out_features)
```

Allowed:

```python
def softmax(x, dim=-1):
    shifted = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(shifted)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)
```

Forbidden:

```python
torch.softmax(x, dim=-1)
torch.nn.functional.softmax(x, dim=-1)
```

Allowed:

```python
with torch.no_grad():
    for p in params:
        if p.grad is not None:
            p.add_(p.grad, alpha=-lr)
```

Forbidden:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
```

### Final Edge-Case Rule

A function, class, method, or library call is forbidden if it directly implements something the assignment asks you to build from scratch.

Namespace loopholes are not allowed. Moving from `torch.nn.functional.softmax` to `torch.softmax`, `Tensor.softmax`, `torch.distributions.Categorical`, or an equivalent wrapper does not make the operation permitted.

If you are unsure whether a helper is allowed, do not use it. Implement the operation directly from elementary tensor operations.
