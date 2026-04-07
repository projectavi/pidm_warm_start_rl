# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
from typing import Tuple

import torch
from torch import Tensor, nn

from pidm_imitation.utils.valid_controller_actions import ValidControllerActions


def hippo_legs(N: int) -> Tensor:
    """
    Return the HiPPO-LegS matrix used to initialize the stable continuous-time dynamics.
    """
    if N <= 0:
        raise ValueError(f"HiPPO dimension must be positive, got {N}.")

    A = torch.zeros(N, N, dtype=torch.float32)
    for n in range(N):
        for k in range(N):
            if n > k:
                A[n, k] = -math.sqrt(2 * n + 1) * math.sqrt(2 * k + 1)
            elif n == k:
                A[n, k] = -(n + 1)
    return A


def causal_conv_fft(u_tilde: Tensor, K: Tensor) -> Tensor:
    """
    Compute the causal convolution y_k = sum_{j=0}^k K_{k-j} u_j via FFT.

    Args:
        u_tilde: (batch, seq_len, d_in)
        K:       (seq_len, d_out, d_in)
    Returns:
        (batch, seq_len, d_out)
    """
    batch_size, seq_len, d_in = u_tilde.shape
    _, d_out, kernel_d_in = K.shape
    if d_in != kernel_d_in:
        raise ValueError(
            f"Kernel input dim mismatch, got input dim {d_in} and kernel dim {kernel_d_in}."
        )

    fft_len = 1 << (2 * seq_len - 1).bit_length()
    U_f = torch.fft.rfft(u_tilde.transpose(1, 2), n=fft_len)
    K_f = torch.fft.rfft(K.permute(1, 2, 0), n=fft_len)
    Y_f = torch.einsum("bif,oif->bof", U_f, K_f)
    y = torch.fft.irfft(Y_f, n=fft_len)[..., :seq_len]
    return y.transpose(1, 2)


def get_block_nonlinearity(name: str) -> nn.Module:
    name = name.lower()
    if name == "none":
        return nn.Identity()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(
        f"Unsupported SSIDM block nonlinearity '{name}'. Expected one of: none, silu, gelu."
    )


class StructuredSSMCore(nn.Module):
    """
    Continuous-time linear state space model with ZOH discretisation.

    This implements the exact diagonal-A version from the implementation brief:
        x'(t) = A x(t) + B u_tilde(t)
        y(t)  = C x(t) + D u_tilde(t)
    with recurrent discrete inference and convolutional training.
    """

    def __init__(
        self,
        stream_dim: int,
        action_dim: int,
        ssm_state_dim: int = 64,
        delta_init: float = 1.0,
        hippo_init: bool = True,
        diagonal_A: bool = True,
    ):
        super().__init__()
        if stream_dim <= 0:
            raise ValueError(f"stream_dim must be positive, got {stream_dim}.")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}.")
        if ssm_state_dim <= 0:
            raise ValueError(f"ssm_state_dim must be positive, got {ssm_state_dim}.")
        if delta_init <= 0:
            raise ValueError(f"delta_init must be positive, got {delta_init}.")
        if not diagonal_A:
            raise NotImplementedError(
                "Only the exact diagonal-A SSIDM core is implemented in this phase."
            )

        self.N = ssm_state_dim
        self.stream_dim = stream_dim
        self.action_dim = action_dim
        self.diagonal_A = diagonal_A

        if hippo_init:
            hippo_diag = -torch.diag(hippo_legs(self.N))
            init_log_neg_lambda = torch.log(hippo_diag)
        else:
            init_log_neg_lambda = torch.zeros(self.N, dtype=torch.float32)
        self.log_neg_lambda = nn.Parameter(init_log_neg_lambda)

        b_scale = 1.0 / math.sqrt(self.N)
        self.B_e = nn.Parameter(torch.randn(self.N, self.stream_dim) * b_scale)
        self.B_f = nn.Parameter(torch.randn(self.N, self.stream_dim) * b_scale)
        self.C = nn.Parameter(torch.randn(self.action_dim, self.N) * b_scale)
        self.D_e = nn.Parameter(torch.zeros(self.action_dim, self.stream_dim))
        self.D_f = nn.Parameter(torch.zeros(self.action_dim, self.stream_dim))
        self.log_delta = nn.Parameter(
            torch.tensor(math.log(delta_init), dtype=torch.float32)
        )

        self._cached_state: Tensor | None = None

    def get_A(self) -> Tensor:
        lambda_ = -torch.exp(self.log_neg_lambda)
        return torch.diag(lambda_)

    def get_B(self) -> Tensor:
        return torch.cat([self.B_e, self.B_f], dim=-1)

    def get_D(self) -> Tensor:
        return torch.cat([self.D_e, self.D_f], dim=-1)

    def discretise(self) -> Tuple[Tensor, Tensor]:
        A = self.get_A()
        B = self.get_B()
        delta = torch.exp(self.log_delta).to(device=A.device, dtype=A.dtype)

        N = A.shape[0]
        M = torch.zeros(2 * N, 2 * N, device=A.device, dtype=A.dtype)
        M[:N, :N] = delta * A
        M[:N, N:] = delta * torch.eye(N, device=A.device, dtype=A.dtype)

        eM = torch.linalg.matrix_exp(M)
        A_bar = eM[:N, :N]
        B_int = eM[:N, N:]
        B_bar = B_int @ B
        return A_bar, B_bar

    def build_kernel(self, seq_len: int) -> Tensor:
        A_bar, B_bar = self.discretise()
        D = self.get_D()

        kernel = [self.C @ B_bar + D]
        A_pow = A_bar
        for _ in range(1, seq_len):
            kernel.append(self.C @ A_pow @ B_bar)
            A_pow = A_pow @ A_bar
        return torch.stack(kernel, dim=0)

    def _get_initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype, use_cached: bool
    ) -> Tensor:
        if (
            use_cached
            and self._cached_state is not None
            and self._cached_state.shape[0] == batch_size
        ):
            return self._cached_state.to(device=device, dtype=dtype)
        return torch.zeros(batch_size, self.N, device=device, dtype=dtype)

    def forward_recurrent(
        self,
        u: Tensor,
        u_ref: Tensor,
        use_cached_state: bool = False,
        update_cache: bool = False,
    ) -> Tensor:
        batch_size, seq_len, _ = u.shape
        A_bar, B_bar = self.discretise()
        D = self.get_D()
        u_tilde = torch.cat([u, u_ref], dim=-1)

        x = self._get_initial_state(
            batch_size=batch_size,
            device=u.device,
            dtype=u.dtype,
            use_cached=use_cached_state,
        )
        ys = []
        for k in range(seq_len):
            x = x @ A_bar.T + u_tilde[:, k, :] @ B_bar.T
            y_k = x @ self.C.T + u_tilde[:, k, :] @ D.T
            ys.append(y_k)

        if update_cache:
            self._cached_state = x.detach()
        return torch.stack(ys, dim=1)

    def forward_convolve(self, u: Tensor, u_ref: Tensor) -> Tensor:
        batch_size, seq_len, _ = u.shape
        del batch_size
        u_tilde = torch.cat([u, u_ref], dim=-1)
        kernel = self.build_kernel(seq_len)
        return causal_conv_fft(u_tilde, kernel)

    def step(self, u: Tensor, u_ref: Tensor) -> Tensor:
        if u.ndim == 2:
            u = u.unsqueeze(1)
        if u_ref.ndim == 2:
            u_ref = u_ref.unsqueeze(1)
        return self.forward_recurrent(
            u,
            u_ref,
            use_cached_state=True,
            update_cache=True,
        )

    def reset(self) -> None:
        self._cached_state = None


class StructuredSSMBlock(nn.Module):
    """
    Residual SSIDM block that wraps the exact linear SSM core with optional
    prenorm, pointwise nonlinearity, and dropout.

    The core remains unchanged and linear. Any nonlinearity is applied pointwise
    to the block output so the sequence/recurrent dual execution paths remain
    consistent while adding network-level expressiveness.
    """

    def __init__(
        self,
        d_model: int,
        ssm_state_dim: int = 64,
        delta_init: float = 1.0,
        hippo_init: bool = True,
        diagonal_A: bool = True,
        block_nonlinearity: str = "none",
        prenorm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.norm: nn.Module = nn.LayerNorm(d_model) if prenorm else nn.Identity()
        self.core = StructuredSSMCore(
            stream_dim=d_model,
            action_dim=d_model,
            ssm_state_dim=ssm_state_dim,
            delta_init=delta_init,
            hippo_init=hippo_init,
            diagonal_A=diagonal_A,
        )
        self.activation = get_block_nonlinearity(block_nonlinearity)
        self.dropout: nn.Module = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.block_nonlinearity = block_nonlinearity
        self.prenorm = prenorm

    def _postprocess(self, delta: Tensor) -> Tensor:
        return self.dropout(self.activation(delta))

    def forward_convolution(self, hidden: Tensor, reference: Tensor) -> Tensor:
        normalized_hidden = self.norm(hidden)
        delta = self.core.forward_convolve(normalized_hidden, reference)
        return hidden + self._postprocess(delta)

    def forward_recurrent(self, hidden: Tensor, reference: Tensor) -> Tensor:
        normalized_hidden = self.norm(hidden)
        delta = self.core.forward_recurrent(
            normalized_hidden,
            reference,
            use_cached_state=False,
            update_cache=False,
        )
        return hidden + self._postprocess(delta)

    def forward_step(self, hidden: Tensor, reference: Tensor) -> Tensor:
        normalized_hidden = self.norm(hidden)
        delta = self.core.step(normalized_hidden, reference)
        return hidden + self._postprocess(delta)

    def reset(self) -> None:
        self.core.reset()

    @property
    def _cached_state(self) -> Tensor | None:
        return self.core._cached_state


class SSIDMPolicyNetwork(nn.Module):
    """
    Shared SSIDM implementation for both PSSIDM and LSSIDM.

    The backbone is a residual stack of structured SSM blocks. PSSIDM uses only a
    minimal per-timestep linear lift from raw executed/reference state streams into
    a common hidden width d_model. LSSIDM uses a shared timestep-wise latent encoder
    instead. Training uses the convolutional sequence path through every block;
    inference uses recurrent stepping with one cached state per block.
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        action_type: str,
        d_model: int | None = None,
        num_ssm_layers: int = 1,
        ssm_state_dim: int = 64,
        delta_init: float = 1.0,
        hippo_init: bool = True,
        diagonal_A: bool = True,
        block_nonlinearity: str = "none",
        prenorm: bool = False,
        dropout: float = 0.0,
        latent_encoder_dim: int = 128,
        use_latent_encoder: bool = False,
    ):
        super().__init__()
        if action_type != ValidControllerActions.LEFT_STICK:
            raise ValueError(
                f"Unsupported action type for SSIDMPolicyNetwork: {action_type}"
            )
        if input_dim != 2 * state_dim:
            raise ValueError(
                "SSIDMPolicyNetwork currently expects only the strict state_only input "
                f"contract [state_history; state_lookahead], got input_dim={input_dim} "
                f"and state_dim={state_dim}."
            )

        self.input_dim = input_dim
        self.state_dim = state_dim
        self.action_type = action_type
        self.use_latent_encoder = use_latent_encoder
        self.action_dim = ValidControllerActions.get_actions_dim(action_type)
        self.d_model = d_model if d_model is not None else (
            latent_encoder_dim if use_latent_encoder else state_dim
        )
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}.")
        if num_ssm_layers <= 0:
            raise ValueError(
                f"num_ssm_layers must be positive, got {num_ssm_layers}."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        self.num_ssm_layers = num_ssm_layers
        self.block_nonlinearity = block_nonlinearity
        self.prenorm = prenorm
        self.dropout = dropout

        if self.use_latent_encoder:
            self.shared_latent_encoder: nn.Module = nn.Sequential(
                nn.Linear(state_dim, self.d_model),
                nn.ReLU(),
            )
            self.executed_lift = None
            self.reference_lift = None
        else:
            self.shared_latent_encoder = nn.Identity()
            if self.d_model == state_dim:
                self.executed_lift = nn.Identity()
                self.reference_lift = nn.Identity()
            else:
                self.executed_lift = nn.Linear(state_dim, self.d_model)
                self.reference_lift = nn.Linear(state_dim, self.d_model)

        self.blocks = nn.ModuleList(
            [
                StructuredSSMBlock(
                    d_model=self.d_model,
                    ssm_state_dim=ssm_state_dim,
                    delta_init=delta_init,
                    hippo_init=hippo_init,
                    diagonal_A=diagonal_A,
                    block_nonlinearity=block_nonlinearity,
                    prenorm=prenorm,
                    dropout=dropout,
                )
                for _ in range(self.num_ssm_layers)
            ]
        )
        self.output_head = nn.Linear(self.d_model, self.action_dim)

    def _split_streams(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        executed = inputs[..., : self.state_dim]
        reference = inputs[..., self.state_dim :]
        return executed, reference

    def _encode_streams(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        executed, reference = self._split_streams(inputs)
        if self.use_latent_encoder:
            return self.shared_latent_encoder(executed), self.shared_latent_encoder(
                reference
            )
        assert self.executed_lift is not None and self.reference_lift is not None
        return self.executed_lift(executed), self.reference_lift(reference)

    def forward_convolution(self, inputs: Tensor) -> Tensor:
        hidden, reference = self._encode_streams(inputs)
        for block in self.blocks:
            hidden = block.forward_convolution(hidden, reference)
        return self.output_head(hidden)

    def forward_recurrent(self, inputs: Tensor) -> Tensor:
        hidden, reference = self._encode_streams(inputs)
        for block in self.blocks:
            hidden = block.forward_recurrent(hidden, reference)
        return self.output_head(hidden)

    def forward_step(self, inputs: Tensor) -> Tensor:
        hidden, reference = self._encode_streams(inputs)
        for block in self.blocks:
            hidden = block.forward_step(hidden, reference)
        return self.output_head(hidden)

    def forward(self, inputs: Tensor) -> Tensor:
        if self.training:
            return self.forward_convolution(inputs)
        if inputs.shape[1] == 1:
            return self.forward_step(inputs)
        return self.forward_recurrent(inputs)

    def reset(self) -> None:
        for block in self.blocks:
            block.reset()

    @property
    def collapse_sequence(self) -> bool:
        return False

    @property
    def is_recurrent(self) -> bool:
        return True

    @property
    def out_dim(self) -> int:
        return self.action_dim
