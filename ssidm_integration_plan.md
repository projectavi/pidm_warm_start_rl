# PSSIDM/LSSIDM Integration Plan

## Goal

Add two new algorithm modes, `pssidm` and `lssidm`, to this repository so they can be trained, evaluated, and compared against the existing `bc` and `idm` modes without creating a parallel training stack.

## Verification Summary

The existing `pidm_ssm_implementation_brief.md` is mathematically useful, but it does not match several structural realities of this codebase. The plan below is based on the current implementation, not on the brief's standalone architecture.

### What the brief gets right

- A structured state-space formulation is a reasonable way to implement a predictive inverse-dynamics policy.
- A stable diagonal `A` parameterization is the right first version.
- The executed-state plus future-reference-state contract is compatible with the current IDM-style data path.
- Training and inference should share one model implementation, with a recurrent rollout path available for evaluation.

### What must be corrected for this repo

1. The repo is not organized around a standalone `PIDMModel -> SSMLayer -> LossHead` stack.
   It is organized around:
   - YAML config parsing
   - `DataModuleFactory`
   - `ModelFactory`
   - `SingleHeadActionRegressor`
   - submodel factories and inference-agent factories

2. The dataset does not return `(u, u_ref, y_target)` tuples.
   It returns dictionaries with keys such as:
   - `state_history`
   - `action_history`
   - `action_target`
   - `state_lookahead`
   - `action_lookahead`
   - `lookahead_k_onehot`

   For strict `pssidm`, only `state_history`, `state_lookahead`, and `action_target`
   are required. The `lookahead_k_*` keys exist in the repo dataset contract, but they
   should not be part of the strict fixed-horizon SSIDM core input.

3. The brief's horizon variable `h` should not be introduced as a second, separate shifting mechanism.
   This repo already encodes future reference selection through `data.lookahead` and `data.lookahead_slice_specs`.
   Important convention: in the current dataset, `lookahead_k = 0` already means the next future state, not the current one.

   For strict `pssidm`, this should be fixed rather than variable:
   - use a single fixed lookahead slice
   - start with `lookahead_k = 0`, i.e. next-state reference
   - do not train over multiple horizons in version 1

4. The brief assumes a custom config dataclass like `PIDMConfig`.
   This repo stores model/data settings in YAML and serializes them through `OfflinePLConfigFile`, `ModelConfig`, and datamodule hyperparameters in checkpoints.

5. The brief names `scipy` and `einops` as required dependencies, but this repo currently depends on neither.
   The first implementation should use `torch.linalg.matrix_exp` and standard PyTorch tensor ops instead of adding SciPy unless a later benchmark shows a concrete need.

6. The brief is sequence-to-sequence by default.
   That should be preserved for `pssidm` and `lssidm` training. The adjustment required by this repo is narrower: rollout still expects a next-action interface, so these algorithms need an adapter from sequence outputs to the current one-step agent API instead of collapsing the model itself to final-action-only.

## Recommended Design

Implement `pssidm` and `lssidm` as new policy-model types inside the existing single-head supervised-learning stack.

That means:

- Keep using `SingleHeadActionRegressor`.
- Keep using `PolicyHead`.
- Add a new policy network implementation, for example `SSIDMPolicyNetwork`, under `pidm_imitation/agents/models/`.
- Route `model.algorithm: pssidm` and `model.algorithm: lssidm` through the same training entrypoint and checkpoint format as `bc` and `idm`.
- Reuse the IDM inference path and planner contract for evaluation, because both algorithms still need a future reference state at rollout time.

This is the lowest-friction path and preserves comparability.

## Two Supported Algorithms

To accommodate both strict mathematical fidelity and a practical latent-feature extension, the implementation plan should support two explicit algorithms built on the same SSIDM framework.

### `pssidm`: Pure Structured SSM IDM

This algorithm preserves the brief literally:

- executed input is the raw state `u(t)`
- future-reference input is the raw state `u*(t+h)`
- the augmented input is `[u(t); u*(t+h)]`
- any projection used is only the brief's own `InputProjection` inside the SSIDM model
- the SSM equations act directly on those state-coordinate inputs

This is the reference implementation and should be the default.

### `lssidm`: Latent-Encoded Structured SSM IDM

This algorithm keeps the same SSM equations, but applies them in learned latent coordinates:

- `z(t) = phi(u(t))`
- `z_ref(t+h) = phi(u*(t+h))`
- the augmented input is `[z(t); z_ref(t+h)]`
- the same SSM core, discretization, convolutional training path, and recurrent inference path are used

This is not literally the brief's raw-state contract, but it is still faithful to the framework as an SSM over executed/reference coordinates in a learned representation space.

Intended training behavior for this variant:

- encode the executed and reference sequences in parallel
- then apply the SSIDM convolutional sequence operator to the encoded sequences

In other words: `encode in parallel, then convolve`.

Recommendation:

- register `pssidm` and `lssidm` separately in every algorithm and agent registry
- require them to share the same underlying SSIDM implementation, while exposing them as distinct comparable modes

## Scope For Version 1

Version 1 should intentionally be narrow, but it should still preserve the SSM training/inference split:

- Algorithm names:
  - `pssidm`
  - `lssidm`
- Evaluation agent names:
  - `toy_pssidm`
  - `toy_lssidm`
- Input format support: `state_only`
- Fixed-horizon training for both variants
- Default horizon choice: next-state reference (`lookahead_k = 0`)
- Training mode: seq-to-seq prediction over the supervised window
- Inference mode: recurrent/stateful next-action prediction using the existing IDM-style planner path to supply the current future reference

Avoid supporting `state_and_action` and `state_and_action_history` in the first pass. Those can be added after the base comparison is working.

## Mapping The Brief To Current Repo Semantics

### Inputs

Use the current IDM-style tensors as the structured input contract:

- executed stream: `state_history`
- future-reference stream: `state_lookahead`
- supervision target: `action_target`

For `state_only`, the policy head already knows how to concatenate these inputs into one tensor. `SSIDMPolicyNetwork` can split that tensor internally into executed/reference components or consume it through explicit projected channels.

Variant-specific interpretation:

- `pssidm`: `state_history` and `state_lookahead` are used directly as `u` and `u_ref`
- `lssidm`: a shared encoder `phi` first maps those tensors to latent executed/reference coordinates before they enter the SSM core

For strict fixed-horizon `pssidm`:

- `lookahead_k_onehot` should not be part of the SSM core input
- the realized horizon comes entirely from the dataset configuration
- version 1 should use the next future state as the reference target

### Horizon

Do not add a new top-level model hyperparameter named `h` in parallel to the data config.

Instead:

- treat the repo's existing lookahead configuration as the source of truth
- for fixed-horizon training, use a single lookahead slice
- for version 1, use the next-state future reference, i.e. the repo setting that yields `lookahead_k = 0`
- if needed, save the resolved lookahead slice in checkpoint hyperparameters for analysis

Concretely, this means:

- do not add a separate model-level `h` field in version 1
- enforce a single fixed dataset lookahead slice during `pssidm` and `lssidm` training
- use the existing repo configuration to realize the horizon
- disable `include_k` for the strict fixed-horizon `pssidm` path

### Training Objective

Use the existing supervised action sequence target and current Lightning training loop.

This keeps comparison with `bc` and `idm` fair:

- same dataset
- same action targets
- same optimizer and scheduler plumbing
- same evaluation harness

Important clarification:

- `pssidm` and `lssidm` should train against the full `action_target` sequence for the sampled window
- rollout should consume only the current/last predicted action at each environment step
- the model itself should therefore expose both:
  - a sequence training path
  - a recurrent or one-step inference path

## Implementation Work Plan

Execution discipline for this work:

- keep `IMPLEMENTATION_PROGRESS_HANDOFF.md` updated at the start and end of each substantive phase
- commit at each stable checkpoint rather than waiting for the full feature to land
- favor scaffold-first integration: registrations and signatures first, then basic data-flow implementations, then real component math

The practical build sequence should therefore be:

1. lay out the framework with method signatures, registry entries, factory routing, and rollout-adapter seams
2. add minimal shared implementations so batches can flow through training and inference without the full SSM math yet
3. replace the placeholders component by component with the real shared SSIDM implementation
4. validate each checkpoint before moving on

### Phase 1: Register `pssidm` and `lssidm` everywhere the repo enumerates algorithms

Update the registries so `pssidm` and `lssidm` are treated as first-class modes:

- `pidm_imitation/utils/valid_models.py`
- `pidm_imitation/agents/supervised_learning/inference_agents/pytorch_valid_agents.py`
- `pidm_imitation/evaluation/toy_valid_agents.py`
- docs and example configs

Required outcomes:

- `ValidModels.ALL` contains `pssidm` and `lssidm`
- `ValidPytorchAgents.ALL` contains `pssidm` and `lssidm`
- `ValidToyAgents.ALL` contains `toy_pssidm` and `toy_lssidm`
- `toy_pssidm` maps to `pssidm` and `toy_lssidm` maps to `lssidm` through the existing prefix-stripping logic

### Phase 2: Reuse IDM data wiring for `pssidm` and `lssidm`

Add `pssidm` and `lssidm` handling to the input-routing code so they receive the same future-reference inputs as IDM.

Files:

- `pidm_imitation/agents/supervised_learning/inputs_factory.py`
- `pidm_imitation/agents/supervised_learning/model_factory.py`

Recommendation:

- make `pssidm` and `lssidm` reuse the IDM input-key contract for version 1
- require `state_only`
- support two explicit branches:
  - `pssidm`: no external encoder and no `lookahead_k_onehot` input
  - `lssidm`: allow a tightly constrained shared encoder branch

Constraint for the latent-encoder branch:

- the encoder must be shared between executed and future-reference streams
- it should be applied pointwise or timestep-wise, not as a history-conditioned sequence summarizer
- the SSM must still operate on per-timestep executed/reference coordinates after encoding

Training-benefit requirement for the latent-encoder branch:

- the encoder stage must preserve sequence structure so the downstream SSIDM still trains seq-to-seq
- the encoder should run in parallel over the full supervised window
- the convolutional speed benefit should still apply to the SSM stage after encoding

What should still be rejected:

- using the existing generic repo encoder as an unconstrained sequence summarizer ahead of the SSM
- using different encoders for executed and future-reference streams unless explicitly justified and tested
- any encoder configuration that collapses the sequence before the SSM stage
- any `pssidm` or `lssidm` configuration that trains over multiple lookahead horizons

### Phase 3: Add the actual SSIDM model family

Create a new policy model implementation under `pidm_imitation/agents/models/`, for example:

- `pidm_imitation/agents/models/ssidm.py`

Suggested components:

- `hippo_legs(N)`
- `StructuredSSMCore`
- `SSIDMPolicyNetwork`

Implementation rule:

- `pssidm` and `lssidm` must share the same underlying SSIDM core implementation
- the only intended architectural difference is whether a shared latent encoder is applied before the common SSM core
- avoid duplicating the SSM logic into separate pure/latent codepaths

`SSIDMPolicyNetwork` should:

- accept the concatenated policy-head input tensor
- split or project the executed and reference streams
- optionally apply a shared latent encoder to each stream before the SSM core
- build discrete SSM parameters from learnable continuous parameters
- use a stable diagonal `A` first
- expose a convolutional sequence-training path and a recurrent inference path
- expose:
  - `collapse_sequence`
  - `is_recurrent`
  - `reset()`

Recommended initial behavior:

- `collapse_sequence = False` during training, so the standard head/loss path supervises the whole action sequence
- implement an internal recurrent state for rollout
- provide a way for inference wrappers to request or extract the current/last action only
- support:
  - a pure raw-state path for `pssidm`
  - a latent-encoder path for `lssidm`

For `lssidm` specifically:

- encode executed/reference sequences in parallel first
- feed the encoded sequences into the convolutional SSIDM training path
- at rollout, encode the current executed/reference timestep and advance the recurrent SSIDM state

That preserves the point of the SSM:

- training uses sequence-to-sequence convolution for speed
- inference runs step-by-step like an RNN

### Phase 4: Integrate the new policy model into the existing factories

Files:

- `pidm_imitation/agents/models/policy_models.py`
- `pidm_imitation/agents/supervised_learning/submodel_factories.py`

Changes:

- register a new policy model class name, e.g. `SSIDMPolicyNetwork`
- extend `PolicyModelFactory.VALID_POLICY_MODELS`
- add argument extraction for SSIDM-specific init args
- keep `PolicyHead` unchanged if possible for training
- add an inference-facing adapter if needed so rollout only consumes the final/current action
- wire `lssidm` to a shared encoder module inside the SSIDM model rather than using the repo's generic state-encoder path as an unconstrained front-end
- keep `pssidm` and `lssidm` as thin configuration/registration differences over one shared implementation

The preferred shape is:

- `model.algorithm: pssidm` or `model.algorithm: lssidm`
- `model.submodels.policy_head.class_name: PolicyHead`
- `model.submodels.policy_head.init_args.policy_model.class_name: SSIDMPolicyNetwork`

This keeps training and checkpoint loading aligned with the current architecture.

### Phase 5: Add inference support by reusing the IDM agent path, but not the training output contract

Files:

- `pidm_imitation/agents/supervised_learning/inference_agents/pytorch_agents_factory.py`
- any small helper updates required by inference wrappers

Recommendation:

- make `pssidm` and `lssidm` follow the IDM branch for planner-backed evaluation
- use `PytorchIdmAgent` or `PytorchSlidingWindowIdmAgent` depending on `model.is_recurrent`
- add SSIDM-specific handling so a sequence-valued policy output is converted to the current action for rollout

Rationale:

- `pssidm` and `lssidm` still depend on a future reference state during rollout
- the current planner contract already provides that
- this preserves direct comparability between `idm`, `pssidm`, and `lssidm`
- training remains seq-to-seq even though rollout remains one-step

### Phase 6: Add configs and documentation

Add at least two example configs:

- `configs/supervised_learning/pssidm_example.yaml`
- `configs/supervised_learning/lssidm_example.yaml`

Base them on `configs/supervised_learning/pidm_example.yaml`, but change:

- `experiment_name`
- `agent.type: toy_pssidm` / `toy_lssidm`
- `model.algorithm: pssidm` / `lssidm`
- policy model class to `SSIDMPolicyNetwork`
- `input_format: state_only`
- remove `state_encoder` for `pssidm`
- add the constrained internal latent encoder config for `lssidm`
- add SSIDM-specific `init_args`

Also update:

- `README.md`
- `pidm_imitation/agents/README.md`

Document `pssidm` and `lssidm` as additional comparable modes beside `bc` and `idm`.

### Phase 7: Add tests before benchmarking

This repo currently has no visible `tests/` tree, so create one rather than relying on ad hoc checks.

Recommended initial tests:

- `tests/test_ssidm_registry.py`
  - verifies `pssidm` and `lssidm` are accepted by all relevant registries

- `tests/test_ssidm_inputs.py`
  - verifies `pssidm` routes `state_history` and `state_lookahead` as the core inputs
  - verifies `pssidm` does not require `lookahead_k_onehot`
  - verifies `pssidm` enforces a single fixed lookahead slice
  - verifies `lssidm` uses the same core sequence inputs before encoding
  - verifies unsupported input formats fail clearly

- `tests/test_ssidm_model.py`
  - verifies the training path outputs a full action sequence matching `action_target`
  - verifies `A` is stable after initialization
  - verifies discretization runs with `torch.linalg.matrix_exp`
  - verifies fixed inputs give deterministic outputs
  - verifies convolutional training output matches recurrent output on the same window within tolerance
  - verifies `pssidm` and `lssidm` both satisfy the same recurrence/convolution equivalence checks

- `tests/test_ssidm_inference_factory.py`
  - verifies `toy_pssidm` and `toy_lssidm` resolve to the IDM-style inference agent path
  - verifies rollout consumes only the current/last action from a sequence-valued SSIDM model

- `tests/test_ssidm_encoder_constraints.py`
  - verifies `lssidm` uses a shared executed/reference encoder
  - verifies the encoder is pointwise/timestep-wise rather than a sequence-collapsing summary model
  - verifies `lssidm` preserves the sequence dimension into the SSM stage

Only after those pass should rollout comparisons start.

## Suggested SSIDM Config Shape

```yaml
data:
  lookahead: 1
  lookahead_slice_specs:
    class_name: LinearSlicer
    num_samples: 1
  include_k: false

model:
  algorithm: pssidm
  input_format: state_only
  submodels:
    policy_head:
      class_name: PolicyHead
      init_args:
        input_keys:
          - state_history
          - state_lookahead
        policy_model:
          class_name: SSIDMPolicyNetwork
          init_args:
            state_dim: null        # inferred in factory
            d_model: 128
            ssm_state_dim: 64
            diagonal_A: true
            hippo_init: true
            delta_init: 1.0
            train_mode: convolutional_sequence
            inference_mode: recurrent_step
        action_loss:
          continuous_loss: l1
```

Notes:

- `state_dim` should be inferred by the factory, matching existing patterns.
- the model should predict the full action sequence during training, not just the final action
- for `pssidm`, use fixed next-state lookahead from the existing repo config and omit `lookahead_k_onehot` from the core model input
- the strict version should set `data.lookahead = 1`, use a single lookahead slice, and disable `include_k`
- for `lssidm`, add a shared internal latent encoder inside the SSIDM model boundary
- `pssidm` and `lssidm` should share the same underlying `SSIDMPolicyNetwork` / `StructuredSSMCore` implementation and differ only in whether the internal latent encoder is active
- for `lssidm`, the intended training decomposition is: encode in parallel, then convolve
- Keep optimizer, scheduler, callbacks, and dataset settings aligned with the comparison baselines.

## Concrete File-Level Change List

### Must change

- `pidm_imitation/utils/valid_models.py`
- `pidm_imitation/agents/supervised_learning/model_factory.py`
- `pidm_imitation/agents/supervised_learning/inputs_factory.py`
- `pidm_imitation/agents/supervised_learning/inference_agents/pytorch_valid_agents.py`
- `pidm_imitation/agents/supervised_learning/inference_agents/pytorch_agents_factory.py`
- `pidm_imitation/evaluation/toy_valid_agents.py`
- `pidm_imitation/agents/models/policy_models.py`
- `pidm_imitation/agents/supervised_learning/submodel_factories.py`
- `README.md`
- `pidm_imitation/agents/README.md`
- `configs/supervised_learning/pssidm_example.yaml`
- `configs/supervised_learning/lssidm_example.yaml`

### New files recommended

- `pidm_imitation/agents/models/ssidm.py`
- `tests/test_ssidm_registry.py`
- `tests/test_ssidm_inputs.py`
- `tests/test_ssidm_model.py`
- `tests/test_ssidm_inference_factory.py`

## Acceptance Criteria

`pssidm` and `lssidm` integration is complete when all of the following are true:

1. Configs with `model.algorithm: pssidm` and `model.algorithm: lssidm` train through `python -m pidm_imitation.agents.supervised_learning.train`.
2. Configs with `agent.type: toy_pssidm` and `agent.type: toy_lssidm` evaluate through the existing toy rollout path.
3. `pssidm` and `lssidm` checkpoints load through `PytorchAgentFactory`.
4. `pssidm` and `lssidm` can be run on the same datasets and with the same train/validation splits as `bc` and `idm`.
5. `pssidm` and `lssidm` train seq-to-seq over supervised windows using a convolutional SSM path.
6. `pssidm` and `lssidm` rollout run recurrently, consuming planner outputs from the current state and emitting one action per environment step.
7. `pssidm` matches the brief literally at the raw-state level.
8. `lssidm` preserves the same SSM properties in learned coordinates.
9. `pssidm` and `lssidm` share one underlying SSIDM implementation rather than maintaining duplicated model code.
10. The comparison only changes the model family, not the surrounding training/evaluation infrastructure.

## Explicit Non-Goals For Version 1

Do not block the first integration on these:

- multi-layer stacked SSM blocks
- NPLR / S4-style low-rank parameterization
- action-conditioned SSIDM inputs
- a brand-new dataset class
- a separate standalone training script
- adding SciPy just for matrix exponentials

Do not drop these from version 1:

- sequence-to-sequence SSIDM training
- recurrent rollout inference
- equivalence testing between convolutional and recurrent forward paths
- a strict raw-state implementation as `pssidm`

Optional in version 1 if time permits, but planned explicitly:

- latent-encoder SSIDM as `lssidm`

## Recommended Build Order

1. Update `IMPLEMENTATION_PROGRESS_HANDOFF.md` and keep it current throughout the work.
2. Register the new algorithm and agent names.
3. Route `pssidm` and `lssidm` through IDM-style inputs and inference selection.
4. Add a shared scaffold `SSIDMPolicyNetwork` and rollout adapter so data can flow end to end with sequence-shaped outputs.
5. Commit the scaffold checkpoint once registrations, construction, and basic data flow are validated.
6. Replace the scaffold internals with the strict raw-state SSIDM implementation for `pssidm`, preserving convolutional training and recurrent inference interfaces.
7. Add a working `pssidm` example config.
8. Add focused tests, especially convolution-vs-recurrence equivalence.
9. Extend the same shared implementation with the internal latent-encoder path for `lssidm`.
10. Add a second example config and encoder-constraint tests.
11. Run training and rollout smoke tests.
12. Commit each validated checkpoint and only then start benchmark comparisons against `bc`, `idm`, `pssidm`, and `lssidm`.

## Stacked Structured Block Addendum

This addendum supersedes the earlier version-1 non-goal that excluded multi-layer stacked SSM blocks. The implementation direction is now: preserve the SSIDM mathematics, and increase expressiveness primarily through a stack of structured blocks rather than through a large generic front-end network.

### Assumptions

The following assumptions are fixed for the stacked design:

1. `pssidm` remains the primary mathematical reference implementation.
2. `pssidm` uses raw state coordinates as the semantic variables of the brief, up to at most a per-timestep linear lift into a common model width.
3. `lssidm` remains a controlled extension in which the same SSIDM core operates in learned latent coordinates produced by a shared timestep-wise encoder.
4. The future-reference horizon remains fixed through the existing repo lookahead configuration:
   - single lookahead slice
   - one-step future (`lookahead_k = 0` in repo semantics)
   - `include_k = false`
5. Training remains sequence-to-sequence using the full supervised window.
6. Rollout remains recurrent and emits one action per environment step.
7. The reference/planner remains external to the SSIDM model and is queried from the current online state at each environment step.
8. For the strict `pssidm` path, no sequence-collapsing encoder, attention block, or generic pre-SSM history summarizer is allowed.
9. Any optional normalization, dropout, or feedforward sublayers are deferred until after the exact stacked structured model is working and validated.

### Why Stacking Is The Right Scaling Mechanism

The scientific goal is to test whether the SSIDM mathematics is expressive enough to compete with existing PIDM baselines. If most additional capacity is placed in a large generic encoder, then positive results become ambiguous:

- they may be caused by the generic encoder rather than by the structured inverse-dynamics formulation
- the performance claim becomes harder to attribute to the SSIDM core
- the comparison against PIDM shifts from "structured predictive inverse dynamics" to "another large neural network with an SSM head"

Stacking structured blocks avoids that ambiguity. It increases expressiveness by composing the same mathematical object multiple times, so the added capacity still belongs to the SSIDM family.

This also aligns with strong reference implementations of modern SSM models:

- S4 scales through repeated layers and a model width `d_model`, rather than by making one single SSM enormous.
- Mamba likewise treats the sequence mixer as one block inside a repeated residual backbone, with each block owning its own inference cache/state.

The important design lesson is not "copy S4 or Mamba literally." The lesson is:

- separate hidden model width from internal state-space dimension
- scale through repeated structured blocks
- keep recurrent state per block
- preserve the same train-parallel / infer-recurrently split at every layer

### Single-Block SSIDM Recap

For the strict raw-state version, define:

- executed state sequence: `u_t \in R^{d_u}`
- future reference sequence: `u^*_t \in R^{d_r}`
- combined input: `\tilde{u}_t = [u_t ; u^*_t] \in R^{d_u + d_r}`

The brief-level continuous-time structured model is:

`x'(t) = A x(t) + B \tilde{u}(t)`

`y(t) = C x(t) + D \tilde{u}(t)`

with `A` Hurwitz. Under zero-order hold with timestep `\Delta`, this becomes:

`\bar{A} = exp(A \Delta)`

`\bar{B} = \int_0^\Delta exp(A \tau) d\tau B`

and the discrete recurrence is:

`x_{t+1} = \bar{A} x_t + \bar{B} \tilde{u}_t`

`y_t = C x_t + D \tilde{u}_t`

If `x_0 = 0`, then the sequence output is the causal convolution:

`y_t = \sum_{k=0}^t K_k \tilde{u}_{t-k}`

where:

`K_0 = C \bar{B} + D`

`K_k = C \bar{A}^k \bar{B}` for `k >= 1`

This is the mathematical reason the same model admits:

- parallel sequence training through convolution/kernel application
- recurrent rollout inference through the state update

### Stacked Structured Blocks: Mathematical Construction

The stacked design should preserve this property block by block.

Introduce a hidden model width `d_model`. For `pssidm`, this width is reached by a minimal per-timestep linear lift:

`h_t^{(0)} = P_u u_t`

`r_t = P_r u^*_t`

where `P_u \in R^{d_model \times d_u}` and `P_r \in R^{d_model \times d_r}` are learned linear maps.

For `lssidm`, replace these linear lifts with a shared timestep-wise encoder `\phi`:

`h_t^{(0)} = \phi(u_t)`

`r_t = \phi(u^*_t)`

The shared encoder must satisfy the following:

- the same encoder is used for executed and reference streams
- it is applied independently at each timestep
- it does not collapse the sequence before the SSM stack

Now define a stack of `L` structured blocks. For each layer `\ell = 1, ..., L`, define the block input:

`z_t^{(\ell)} = [h_t^{(\ell-1)} ; r_t] \in R^{2 d_model}`

Each block has its own continuous-time structured dynamics:

`x'^{(\ell)}(t) = A^{(\ell)} x^{(\ell)}(t) + B^{(\ell)} z^{(\ell)}(t)`

`\Delta h^{(\ell)}(t) = C^{(\ell)} x^{(\ell)}(t) + D^{(\ell)} z^{(\ell)}(t)`

After discretization:

`x_{t+1}^{(\ell)} = \bar{A}^{(\ell)} x_t^{(\ell)} + \bar{B}^{(\ell)} z_t^{(\ell)}`

`\Delta h_t^{(\ell)} = C^{(\ell)} x_t^{(\ell)} + D^{(\ell)} z_t^{(\ell)}`

and the residual update is:

`h_t^{(\ell)} = h_t^{(\ell-1)} + \Delta h_t^{(\ell)}`

Finally, the action head is applied to the final hidden sequence:

`a_t = W_out h_t^{(L)} + b_out`

### Why This Still Preserves The Framework

This preserves the framework for `pssidm` in a stronger sense than a generic front-end MLP does.

1. Each layer is itself a structured inverse-dynamics operator of exactly the same form as the single-block SSIDM.
2. The recurrent rollout path is still explicit:
   - each block stores its own recurrent state `x_t^{(\ell)}`
   - each environment step applies the stack one block at a time
3. The training path remains parallel at the layer level:
   - for each block, the entire sequence can be processed through its convolutional/kernel form
   - then passed to the next block

For `pssidm`, if `P_u` and `P_r` are linear and there are no nonlinearities between blocks, then the entire stacked model remains a linear time-invariant system over an augmented state. To see this, define the global stacked state:

`X_t = [x_t^{(1)} ; x_t^{(2)} ; ... ; x_t^{(L)}]`

and note that each `h_t^{(\ell)}` is an affine linear function of:

- the current block state `x_t^{(\ell)}`
- the previous hidden `h_t^{(\ell-1)}`
- the current reference `r_t`

By substitution, the full stack can be written as a larger discrete linear system:

`X_{t+1} = \mathcal{A} X_t + \mathcal{B} [u_t ; u^*_t]`

`a_t = \mathcal{C} X_t + \mathcal{D} [u_t ; u^*_t]`

for suitably constructed block-triangular matrices `\mathcal{A}, \mathcal{B}, \mathcal{C}, \mathcal{D}`.

This matters because it shows that stacked `pssidm` is not merely "SSM-inspired." It is still exactly a structured state-space model, just with a larger composed state.

For `lssidm`, the latent encoder introduces nonlinearity before the structured stack. That means the end-to-end map is no longer linear in raw state coordinates. However, conditioned on the latent sequence, the SSIDM backbone still preserves:

- causal convolutional sequence training
- recurrent stepwise inference
- explicit per-block structured state evolution

So `lssidm` remains a controlled extension, while `pssidm` remains the exact mathematical reference model.

### Why The Reference Stream Must Be Injected At Every Layer

The future reference should not be injected only once at the bottom of the stack. Injecting `r_t` at every layer is the cleaner design for predictive inverse dynamics.

Reason:

- the reference signal is not incidental context; it is part of the control law itself
- deeper layers should be able to refine the action-relevant transformation while retaining direct access to the same future reference
- if the reference is only supplied to the first layer, higher layers only see a transformed surrogate of the reference rather than the original control target

So the block input should be `[hidden ; reference]` at every layer, not just at layer 1.

### Parameterization And Scaling Rules

The stacked design should introduce these explicit hyperparameters:

- `d_model`: hidden width carried between blocks
- `num_ssm_layers`: number of structured blocks
- `ssm_state_dim`: internal state dimension of each block
- optional `dropout`: default `0.0` in the exact first implementation
- optional `prenorm`: default `false` in the exact first implementation

The main scaling rule is:

- increase expressiveness first through `num_ssm_layers`
- then through `d_model`
- then through `ssm_state_dim`

Rationale:

- `num_ssm_layers` directly increases compositional depth while preserving the block semantics
- `d_model` increases the representational bandwidth of the hidden executed/reference interaction
- `ssm_state_dim` increases the internal memory capacity of each block

This separation mirrors successful SSM reference implementations, where model width and internal state size are not the same knob.

### Recommended Version-1 Block Structure

The first stacked implementation should stay mathematically strict:

1. Per-timestep input lift:
   - `pssidm`: linear only
   - `lssidm`: shared timestep-wise encoder
2. `L` repeated structured SSIDM blocks
3. Residual addition after each block
4. Final linear action head

Do not add in the first stacked version:

- feedforward MLP sublayers between blocks
- gating mechanisms copied from unrelated sequence architectures
- attention layers
- sequence pooling before the structured stack

These may improve performance later, but they would weaken the attribution of performance to the structured inverse-dynamics mathematics.

### Convolution vs Recurrent Equivalence In The Stacked Model

For a single block, equivalence is already required. For the stacked model, equivalence must hold at the full network level.

That means the test should be:

1. Run the full stacked model in sequence mode over a batch window.
2. Run the same stacked model one step at a time using the recurrent cache of every block.
3. Compare the full output sequences.

This must be tested for:

- `pssidm`
- `lssidm`
- multi-layer stacks, not just one-layer degenerate cases

Without this, the stack would not be validated as a correct sequence/recurrent dual implementation.

### Inference State Handling

Each block needs its own recurrent cache. So rollout state is no longer one tensor; it is an ordered collection:

`state_t = (x_t^{(1)}, x_t^{(2)}, ..., x_t^{(L)})`

Reset semantics must clear every block state. Step semantics must update every block in order. The rollout adapter must treat the full stack as recurrent even though the training graph is sequence-based.

### Reference-Informed Optimizer Note

Reference SSM implementations often treat sensitive continuous-time parameters specially, especially:

- `A`
- `B`
- step-size / `\Delta`-related parameters

The usual pattern is lower learning rate and zero weight decay for these parameters. This is not required for the first stacked implementation, but it should be an explicitly planned optimization refinement if training stability becomes an issue.

### Recommended Updated Build Order

This addendum changes the implementation order after the already-completed scaffold/basic-core work:

1. Refactor the current single-core implementation into a reusable `StructuredSSMBlock`.
2. Introduce `d_model` and `num_ssm_layers`.
3. Implement the stacked residual backbone with per-block recurrent caches.
4. Keep `pssidm` strict:
   - raw-state semantic inputs
   - linear timestep-wise lift only
   - no deep generic front-end encoder
5. Keep `lssidm` as the same stacked backbone with the shared timestep-wise latent encoder in front.
6. Add full-stack sequence-vs-recurrent equivalence tests.
7. Re-run train/load/evaluate smoke tests.
8. Only after the exact stacked version is correct, consider optional normalization/dropout and optimizer parameter groups.

### Acceptance Criteria Added By This Addendum

In addition to the earlier acceptance criteria, the stacked implementation is not complete until all of the following are true:

1. `pssidm` and `lssidm` expose `d_model` and `num_ssm_layers` as first-class config knobs.
2. `pssidm` and `lssidm` share the same stacked structured backbone implementation.
3. The recurrent rollout state is maintained per block and reset correctly.
4. Full-stack sequence-mode outputs match full-stack recurrent outputs within tolerance.
5. `pssidm` scaling is achieved primarily through structured block composition rather than through a large generic encoder.
6. `lssidm` differs from `pssidm` only by the presence of the shared latent encoder ahead of the same structured stack.

## Nonlinear Block Addendum

The next expressiveness step is to keep the internal `StructuredSSMCore` fixed and linear, but make the stacked backbone configurable with pointwise nonlinear wrappers around each structured block. This preserves the main per-block sequence-parallel benefit while allowing the overall stacked model to become nonlinear.

### Assumptions

1. The internal `StructuredSSMCore` remains unchanged and linear.
2. Nonlinearities are applied pointwise in time, not across the time dimension.
3. The same block wrapper applies to both `pssidm` and `lssidm`.
4. `pssidm` remains the raw-state reference path; `lssidm` remains the latent-encoder extension.
5. Full-sequence training and stepwise recurrent inference must still agree numerically for the full model, even when block nonlinearities are enabled.

### Configurable Nonlinear Choices

Expose the following SSIDM policy-model config knobs:

- `block_nonlinearity: none | silu | gelu`
- `prenorm: false | true`
- `dropout: float`

The intended first comparison set is:

1. `block_nonlinearity: none`, `prenorm: false`
2. `block_nonlinearity: silu`, `prenorm: false`
3. `block_nonlinearity: silu`, `prenorm: true`

These correspond to:

- exact stacked linear SSIDM baseline
- nonlinear stacked SSIDM with minimal pointwise activation
- nonlinear stacked SSIDM with the same activation plus a more literature-aligned prenorm wrapper

`gelu` should also be implemented as a supported config option, but it does not need to be part of the first comparison matrix.

### Block Form

For hidden stream `h^{(\ell-1)}` and reference stream `r`, each block should compute:

1. optional normalization:
   - `\hat{h}^{(\ell-1)} = Norm(h^{(\ell-1)})`
2. linear structured SSM update:
   - `\Delta h^{(\ell)} = SSM^{(\ell)}([\hat{h}^{(\ell-1)} ; r])`
3. optional pointwise nonlinearity:
   - `\widetilde{\Delta h}^{(\ell)} = \phi(\Delta h^{(\ell)})`
4. optional dropout:
   - `\bar{\Delta h}^{(\ell)} = Dropout(\widetilde{\Delta h}^{(\ell)})`
5. residual update:
   - `h^{(\ell)} = h^{(\ell-1)} + \bar{\Delta h}^{(\ell)}`

When `block_nonlinearity = none`, the block reduces to the exact stacked linear model. When `block_nonlinearity != none`, the overall stack becomes nonlinear, but each internal SSM remains mathematically exact.

### Why This Preserves The Speed Story

The per-block structured update is still a fixed linear SSM. Therefore each block still admits:

- sequence-mode training through the block's convolutional/kernel form
- recurrent rollout through the block's cached state update

The nonlinear transform does break the interpretation that the entire network is one single global linear system, but it does not break the blockwise sequence-parallel execution strategy. This is the intended compromise between mathematical structure and expressiveness.

### Required Validation

In addition to the earlier stacked tests, the nonlinear extension must validate:

1. full-stack convolution mode matches full-stack recurrent mode for:
   - `block_nonlinearity: none`
   - `block_nonlinearity: silu`
   - `block_nonlinearity: gelu`
2. eval-time repeated `forward_step` matches `forward_recurrent` for nonlinear `lssidm`
3. invalid block-nonlinearity values fail fast
4. comparison configs for the first three settings instantiate and train

### Comparison-Ready Configs

Provide smoke configs for:

- `pssidm_smoke.yaml` as option 1
- `pssidm_silu_smoke.yaml` as option 2
- `pssidm_silu_prenorm_smoke.yaml` as option 3
- `lssidm_smoke.yaml` as option 1
- `lssidm_silu_smoke.yaml` as option 2
- `lssidm_silu_prenorm_smoke.yaml` as option 3

These should share all non-SSIDM settings so the comparison isolates only the nonlinear block choice.
