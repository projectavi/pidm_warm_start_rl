# Agents

This module contains agent definitions, neural network components, and imitation learning models.

## Directory Structure

- `agent.py` - Abstract `Agent` base class defining the interface for all agents
- `models/` - Neural network building blocks (layers, activations, schedulers, policy models)
- `supervised_learning/` - Supervised learning models and training infrastructure

## Network Components (`models/`)

Network building blocks for constructing models:
- `network_block.py` - Configurable network block builder
- `policy_models.py` - Policy network implementations
- `layer_types.py` - Custom layer types
- `activations.py`, `norms.py` - Activation functions and normalization layers
- `optimizers.py`, `schedulers.py` - Optimizers and learning rate schedulers

## Training Entry Point

The main training script is located at:

```
pidm_imitation/agents/supervised_learning/train.py
```

### Basic Usage

```bash
python -m pidm_imitation.agents.supervised_learning.train --config <path/to/config.yaml>
```

### CLI Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--config` | Yes | Path to the training config YAML file |
| `--new` | No | Overwrite any existing checkpoints and logs in the experiment directory. Without this flag, training will fail if checkpoints already exist to prevent accidental overwrites |

### Example

```bash
python -m pidm_imitation.agents.supervised_learning.train \
    --config configs/supervised_learning/pidm_example.yaml \
    --new \
```

## Supported Algorithms

The codebase supports the following algorithms, each mapping to a specific action regressor class:

| Algorithm | Config Value | Action Regressor Class | Description |
|-----------|--------------|------------------------|-------------|
| Behavioral Cloning | `bc` | `SingleHeadActionRegressor` | Predicts actions from state/ action history |
| Inverse Dynamics Model | `idm` | `SingleHeadActionRegressor` | Predicts actions given current state and a future goal state (lookahead) |

### Configuration Example

```yaml
model:
  algorithm: idm  # One of: bc, idm
  input_format: state_only
  submodels:
    state_encoder:
      # ...
    policy_head:
      # ...
```

## Input Formats

The `input_format` config option determines what inputs are provided to the model:

| Input Format | Config Value | Description |
|--------------|--------------|-------------|
| State Only | `state_only` | Only state observations for history and lookahead. Actions are not used as model input. |
| State and Action | `state_and_action` | Both state and action sequences for history and lookahead are processed through the state encoder. |
| State and Action History | `state_and_action_history` | States for history/lookahead through encoder, but action history is passed directly to the policy head (bypassing encoder). |

We recommend typically using the `state_only` format unless actions provide significant context. We have observed that action history can significantly harm performance with models learning spurious correlations between actions in the history and predicted target actions during training. These models achieve high training performance but fail to generalize at inference time when action distributions differ.

## Inference Agents

Agents for evaluation/rollout are in `supervised_learning/inference_agents/`. The appropriate agent type is selected based on the model architecture and algorithm.

### Agent Selection Logic

#### 1. BC Models (Behavioral Cloning)

| Model Type | Agent Class | Description |
|------------|-------------|-------------|
| Recurrent (RNN-based) | `PytorchAgent` | Passes single timestep inputs; model maintains internal state |
| Non-recurrent (stacked inputs) | `PytorchSlidingWindowAgent` | Wraps model with `SlidingWindowInferenceModel` to manage input history |

#### 2. IDM Models (Inverse Dynamics)

IDM models require a **planner** (also called state predictor) that provides likely future states as goal states for the IDM policy.

| Model Type | Agent Class | Description |
|------------|-------------|-------------|
| Recurrent (RNN-based) | `PytorchIdmAgent` | Passes single timestep + lookahead; model maintains internal state |
| Non-recurrent (stacked inputs) | `PytorchSlidingWindowIdmAgent` | Wraps model with `SlidingWindowInferenceIdmModel` to manage history and lookahead |

### Sliding Window Inference Models

For non-recurrent models that expect stacked/windowed inputs, the inference models handle input buffering:

- **`SlidingWindowInferenceModel`**: Maintains sliding windows for state and action history. Accepts single-timestep inputs and internally builds the required sequence.
- **`SlidingWindowInferenceIdmModel`**: Extends sliding window handling to also manage lookahead state and action sequences for IDM models.

### IDM Planners

The planner provides future state predictions for IDM inference:

```python
lookahead_state, lookahead_action, lookahead_k = planner.get_lookahead_state_action_and_k(
    current_state, current_action, current_step
)
```

Planner types are configured via `agent.idm_planner` in the config. See `inference_agents/utils/idm_planners.py` for available planners.

### Agent Factory

The `PytorchAgentFactory` automatically selects the respective agent class by algorithm name and whether the model is recurrent:

```python
from pidm_imitation.agents.supervised_learning.inference_agents.pytorch_agents_factory import PytorchAgentFactory

# Factory checks model.is_recurrent and agent type to select appropriate class
agent = PytorchAgentFactory.create_agent(config, model_path, ...)
```

## Reference for Further Directories

- `dataset/` - Dataset and datamodule implementations
- `submodels/` - Submodel components (policy heads, state encoders)
- `utils/` - Utility functions for training, evaluation, and model handling

## See Also

- `configs/supervised_learning/` - Example training configurations
