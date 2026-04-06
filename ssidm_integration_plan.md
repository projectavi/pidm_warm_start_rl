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

1. Register the new algorithm and agent names.
2. Route `pssidm` and `lssidm` through IDM-style inputs and inference selection.
3. Implement the strict raw-state SSIDM policy network for `pssidm` with both convolutional training and recurrent inference paths.
4. Add the rollout adapter that converts SSIDM sequence semantics into the existing one-action agent loop.
5. Add a working `pssidm` example config.
6. Add focused tests, especially convolution-vs-recurrence equivalence.
7. Implement `lssidm` with shared pointwise/timestep-wise encoding inside the SSIDM model.
8. Add a second example config and encoder-constraint tests.
9. Run training and rollout smoke tests.
10. Only then start benchmark comparisons against `bc`, `idm`, `pssidm`, and `lssidm`.
