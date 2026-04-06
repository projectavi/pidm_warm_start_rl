# Implementation Brief: Predictive Inverse Dynamics as a Structured State Space Model

---

## 0. Overview

This document specifies a complete implementation of a Predictive Inverse Dynamics Model
(PIDM) framed as a continuous-time structured state space model (SSM), discretised via
zero-order hold, and trained end-to-end as a sequence model. The implementation follows
the mathematical framework developed in the accompanying derivation, with all identified
holes explicitly resolved.

Stack: Python 3.11+, PyTorch 2.x, einops, scipy (for matrix exponential initialisation).

---

## 1. Mathematical Contract

The model implements the following system exactly. Do not deviate from these equations.

### 1.1 Augmented Input

```
u(t)       ∈ R^{d_u}     executed state at time t
u*(t+h)    ∈ R^{d_u}     reference state at horizon h, known at time t
ũ(t)       ∈ R^{2d_u}    augmented input: [u(t); u*(t+h)]
```

**Causality contract**: The reference trajectory `u*(t+h)` must be provided as a
pre-planned, fully known signal at time t. This is a hard precondition of the model.
The data pipeline (Section 4) is responsible for enforcing it. The model itself makes
no assumptions about how the reference is generated.

### 1.2 Continuous-Time SSM

```
x'(t) = A x(t) + B ũ(t)
y(t)  = C x(t) + D ũ(t)
```

with:
```
A  ∈ R^{N x N}
B  = [B_e | B_f]  ∈ R^{N x 2d_u}
C  ∈ R^{d_y x N}
D  = [D_e | D_f]  ∈ R^{d_y x 2d_u}
```

### 1.3 Zero-Order Hold Discretisation

Given step size Δ:

```
Ā = exp(ΔA)
B̄ = (∫_0^Δ exp(As) ds) B
```

The integral is computed as the matrix power series (do NOT use A^{-1} form):

```
∫_0^Δ exp(As) ds = Σ_{k=0}^{∞} A^k Δ^{k+1} / (k+1)!
```

In practice, compute via `scipy.linalg.expm` applied to the augmented matrix:

```
M = [A, I; 0, 0] * Δ   →   exp(M) = [Ā, B_int; 0, I]
```

where `B_int = ∫_0^Δ exp(As) ds` is read off the top-right block. Then `B̄ = B_int @ B`.
This is numerically stable and requires no invertibility assumption on A.

### 1.4 Discrete Recurrence

```
x_k = Ā x_{k-1} + B̄ ũ_k          (state update)
y_k = C x_k + D ũ_k               (output)
```

### 1.5 Convolution Kernel

The output over a sequence of length L is equivalently:

```
y_k = Σ_{j=0}^{k} K_{k-j} ũ_j
```

where the kernel is:

```
K_0     = C B̄ + D          ← lag-0 term: C B̄ and D are NOT separated
K_j     = C Ā^j B̄          for j ≥ 1
```

**Note**: K_0 = CB̄ + D, not D alone. Implement accordingly. The full output sequence
is a causal convolution computable in O(L log L) via FFT.

### 1.6 Stability Requirement

A must be Hurwitz: all eigenvalues must have strictly negative real part. This is
enforced by parameterisation (see Section 3.1), not by post-hoc clamping.

### 1.7 Horizon h

h is a model hyperparameter. It governs which future reference index is fed as input
at each timestep. It is NOT a parameter of A, B, C, D but it is a hyperparameter of
the overall model and must be tracked in the model config. h is specified in units of
timesteps in the discrete setting.

---

## 2. Architecture

### 2.1 Module Hierarchy

```
PIDMModel
├── SSMLayer (one or more, stacked)
│   ├── SSMCore          ← implements the discrete recurrence and convolution
│   ├── InputProjection  ← projects ũ_k into model dimension if needed
│   └── OutputProjection ← projects SSM output to d_y
└── LossHead             ← MSE loss with correct convention
```

### 2.2 SSMCore

This is the central module. It holds:

- `log_neg_real_A`: parameterised storage of A ensuring Hurwitz (see 3.1)
- `B_e`: R^{N x d_u}
- `B_f`: R^{N x d_u}
- `C`: R^{d_y x N}
- `D_e`: R^{d_y x d_u}
- `D_f`: R^{d_y x d_u}
- `delta`: scalar or learned log-step Δ

It exposes two forward modes:

```python
def forward_recurrent(self, u, u_ref):
    # u:     (batch, seq_len, d_u)
    # u_ref: (batch, seq_len, d_u)  — u_ref[:, k, :] = u*(t_{k+h})
    # returns y: (batch, seq_len, d_y)
    # Used at inference. Iterates the recurrence step by step.

def forward_convolve(self, u, u_ref):
    # Same signature.
    # Used at training. Builds the full kernel and computes output via FFT conv.
    # Must produce identical output to forward_recurrent (test this).
```

Both modes must be tested to produce numerically identical output (within floating
point tolerance) on the same input. Write a dedicated unit test for this.

### 2.3 Stacking

Multiple SSMCore layers can be stacked with residual connections:

```
z_0 = InputProjection(ũ)
z_l = SSMCore_l(z_l-1) + z_{l-1}    for l = 1, ..., L
y   = OutputProjection(z_L)
```

For the first implementation, use a single layer. Stacking is a later extension.

---

## 3. Parameterisation

### 3.1 Enforcing Hurwitz Stability

A must have all eigenvalues with negative real part. Use one of two strategies
depending on whether you want a diagonal or full A:

**Option A — Diagonal A (simpler, recommended for first implementation):**

Store `log_neg_lambda ∈ R^N` and reconstruct:

```python
lambda_ = -torch.exp(self.log_neg_lambda)   # strictly negative real
A = torch.diag(lambda_)
```

This trivially ensures all eigenvalues are negative real.

**Option B — Full A with NPLR structure (for S4 compatibility):**

Parameterise A = Λ - P Q^T where Λ is diagonal with negative real entries
(as above) and P, Q ∈ R^{N x r} are low-rank factors. This recovers the S4
structured parameterisation. Implement this only after Option A is working.

### 3.2 Initialisation of A

Initialise with HiPPO-LegS:

```
A_{nk} = -(2n+1)^{1/2} (2k+1)^{1/2}   for n > k
        = -(n+1)                         for n = k
        = 0                              for n < k
```

For diagonal Option A, initialise `log_neg_lambda` from the diagonal of HiPPO-LegS.
For Option B, factorise HiPPO-LegS as Λ - PQ^T before initialising.

Provide a standalone `hippo_legs(N)` function that returns the N x N matrix.
Do not hardcode N; it is a hyperparameter.

### 3.3 Initialisation of B, C, D

- B_e, B_f: initialise from N(0, 1/N)
- C: initialise from N(0, 1/N)
- D_e, D_f: initialise to zero. D_f=0 means no initial feedforward bias;
  the model learns this from data.

### 3.4 Step Size Δ

Store as `log_delta` (log of Δ) to ensure positivity:

```python
delta = torch.exp(self.log_delta)
```

Initialise `log_delta` such that Δ ≈ 1/seq_len. Make it a learned parameter.

---

## 4. Data Pipeline

### 4.1 Dataset Contract

The dataset must return tuples:

```
(u, u_ref, y_target)
```

where:
```
u:        (seq_len, d_u)   executed state trajectory
u_ref:    (seq_len, d_u)   u_ref[k] = u*(t_{k+h}) for all k
y_target: (seq_len, d_y)   ground truth actions
```

The horizon shift must be applied inside the dataset, not inside the model. The model
receives `u_ref` already shifted. This keeps the model's causal contract clean.

### 4.2 Horizon Shift Implementation

```python
def apply_horizon_shift(reference_trajectory, h):
    # reference_trajectory: (total_len, d_u)
    # h: int, number of timesteps
    # returns: (total_len - h, d_u)
    # u_ref[k] = reference_trajectory[k + h]
    return reference_trajectory[h:]
```

Truncate the executed trajectory correspondingly so all sequences are aligned:

```python
u     = executed[:total_len - h]
u_ref = reference[h:]
y     = actions[:total_len - h]
```

### 4.3 Validation

Assert inside the dataset:
- `u.shape == u_ref.shape`
- `u.shape[0] == y.shape[0]`
- No NaNs or Infs in any tensor
- h >= 1 and h < seq_len

---

## 5. Forward Pass: Detailed Implementation

### 5.1 Build Discrete Matrices

Called once per forward pass (or cached if Δ and A are not being updated):

```python
def discretise(self):
    A = self.get_A()          # (N, N)
    B = self.get_B()          # (N, 2*d_u)  — concatenate B_e and B_f
    delta = torch.exp(self.log_delta)   # scalar

    # Compute Ā and B̄ via augmented matrix exponential
    N = A.shape[0]
    M = torch.zeros(2*N, 2*N)
    M[:N, :N] = delta * A
    M[:N, N:] = delta * torch.eye(N)
    # M[N:, N:] = 0 (already zero)

    eM = torch.linalg.matrix_exp(M)
    A_bar = eM[:N, :N]          # (N, N)
    B_int = eM[:N, N:]          # (N, N), this is ∫_0^Δ exp(As)ds
    B_bar = B_int @ B           # (N, 2*d_u)

    return A_bar, B_bar
```

### 5.2 Recurrent Forward (Inference)

```python
def forward_recurrent(self, u, u_ref):
    # u:     (B, L, d_u)
    # u_ref: (B, L, d_u)
    B_batch, L, _ = u.shape
    A_bar, B_bar = self.discretise()

    u_tilde = torch.cat([u, u_ref], dim=-1)   # (B, L, 2*d_u)
    D = torch.cat([self.D_e, self.D_f], dim=-1)  # (d_y, 2*d_u)

    x = torch.zeros(B_batch, self.N, device=u.device, dtype=u.dtype)
    ys = []
    for k in range(L):
        x = x @ A_bar.T + u_tilde[:, k, :] @ B_bar.T   # (B, N)
        y_k = x @ self.C.T + u_tilde[:, k, :] @ D.T    # (B, d_y)
        ys.append(y_k)

    return torch.stack(ys, dim=1)   # (B, L, d_y)
```

### 5.3 Convolutional Forward (Training)

Build the kernel sequence and apply via FFT:

```python
def forward_convolve(self, u, u_ref):
    # u:     (B, L, d_u)
    # u_ref: (B, L, d_u)
    B_batch, L, _ = u.shape
    A_bar, B_bar = self.discretise()

    u_tilde = torch.cat([u, u_ref], dim=-1)   # (B, L, 2*d_u)
    D = torch.cat([self.D_e, self.D_f], dim=-1)

    # Build kernel K ∈ R^{L x d_y x 2*d_u}
    K = []
    A_pow = torch.eye(self.N, device=u.device)
    for j in range(L):
        if j == 0:
            K_j = self.C @ B_bar + D      # (d_y, 2*d_u) — correct lag-0 term
        else:
            K_j = self.C @ A_pow @ B_bar  # (d_y, 2*d_u)
        K.append(K_j)
        A_pow = A_pow @ A_bar

    K = torch.stack(K, dim=0)   # (L, d_y, 2*d_u)

    # Causal convolution via FFT
    # u_tilde: (B, L, 2*d_u) → (B, 2*d_u, L) for conv
    # K:       (L, d_y, 2*d_u)
    # Output:  (B, L, d_y)
    # Use torch.fft or implement as batched conv1d

    y = causal_conv_fft(u_tilde, K)  # implement separately — see 5.4
    return y
```

**Note on kernel construction efficiency**: The loop above is O(L * N^2). For large
L and N, replace with the S4 Cauchy kernel trick. For the first implementation, the
loop is acceptable.

### 5.4 causal_conv_fft

```python
def causal_conv_fft(u_tilde, K):
    # u_tilde: (B, L, 2*d_u)
    # K:       (L, d_y, 2*d_u)
    # returns: (B, L, d_y)

    B, L, d_in = u_tilde.shape
    d_y = K.shape[1]
    fft_len = 2 * L   # zero-pad to avoid circular aliasing

    # FFT of input: (B, 2*d_u, fft_len)
    U_f = torch.fft.rfft(u_tilde.transpose(1, 2), n=fft_len)

    # FFT of kernel: (d_y, 2*d_u, fft_len)
    K_f = torch.fft.rfft(K.permute(1, 2, 0), n=fft_len)

    # Multiply and sum over input dimension: (B, d_y, fft_len)
    Y_f = torch.einsum('bid,oid->bod', U_f, K_f)

    # IFFT and trim: (B, d_y, L)
    y = torch.fft.irfft(Y_f, n=fft_len)[..., :L]

    return y.transpose(1, 2)   # (B, L, d_y)
```

---

## 6. Loss Function

The training loss is MSE with the correct convention (model output vs ground truth):

```python
def loss(y_pred, y_target):
    # y_pred:   (B, L, d_y)  — model output
    # y_target: (B, L, d_y)  — observed ground truth
    return torch.mean((y_pred - y_target) ** 2)
```

`y_pred` is the model output. `y_target` is the ground truth. Do not swap these.

---

## 7. Model Configuration

All hyperparameters must be stored in a single config dataclass:

```python
@dataclass
class PIDMConfig:
    d_u:      int           # input dimension per stream (executed and reference)
    d_y:      int           # output (action) dimension
    N:        int           # SSM state dimension
    h:        int           # prediction horizon in timesteps
    delta_init: float       # initial step size Δ (e.g. 1/seq_len)
    n_layers: int = 1       # number of stacked SSM layers
    hippo_init: bool = True # initialise A from HiPPO-LegS
    diagonal_A: bool = True # use diagonal A (Option A) vs NPLR (Option B)
```

The horizon `h` is logged as a top-level model attribute. It must be saved with
model checkpoints and restored on load.

---

## 8. Testing Requirements

The following tests must all pass before training:

### 8.1 Recurrent/Convolutional Equivalence
```
max |forward_recurrent(u, u_ref) - forward_convolve(u, u_ref)| < 1e-5
```
Test on random inputs, batch size 4, seq_len 64, with both random A and HiPPO init.

### 8.2 Discretisation Correctness
Verify that `Ā = exp(ΔA)` and that `B̄` satisfies `dB̄/dΔ = exp(ΔA) B` numerically
(finite difference check).

### 8.3 Lag-0 Kernel Term
Assert that `K[0] == C @ B_bar + D` exactly (not `D` alone).

### 8.4 Stability
After initialisation, assert all eigenvalues of A have negative real part.

### 8.5 Causality
Assert that `y_pred[:, k, :]` does not depend on `u[:, j, :]` for `j > k`.
Test by zeroing out future inputs and verifying outputs up to k are unchanged.

### 8.6 Horizon Shift
Assert that `u_ref[k] == reference[k + h]` in the dataset for several k values.

### 8.7 Loss Convention
Assert that `loss(y, y) == 0` and `loss(y, y + 1) > 0`.

---

## 9. Training Loop

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

for epoch in range(n_epochs):
    for u, u_ref, y_target in dataloader:
        optimizer.zero_grad()
        y_pred = model.forward_convolve(u, u_ref)   # use conv mode at training
        l = loss(y_pred, y_target)
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
```

Switch to `forward_recurrent` at inference time only. Gradient clipping is important
because gradients flow through long convolution kernels.

---

## 10. Known Limitations and Future Extensions

| Limitation | Resolution |
|---|---|
| Diagonal A limits expressivity | Implement NPLR Option B |
| Kernel loop is O(L·N²) | Replace with Cauchy kernel (S4 §3) |
| Fixed h across training | Curriculum over h; random h sampling |
| Single-layer | Stack with residual connections (Section 2.3) |
| ZOH assumes piecewise-constant input | Replace with bilinear (Tustin) discretisation |
| No uncertainty estimate on y | Add output variance head |
