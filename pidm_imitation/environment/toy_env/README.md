# Toy Environment

This folder provides the implementation of a simple 2D navigation environment to run simple experiments in. The environment is based on the [gymnasium](https://gymnasium.farama.org/) interface and can be used with the same API.

All (current) tasks are represented by 2D navigation tasks with the player controlling the blue square that needs to be moved to any or all goals (red squares) and avoid walls (black). The actions are represented as continuous 2D movement.

---

## Environment Logic (Base Environment)

The base environment (`toy_environment_base.py`) provides core mechanics shared by all tasks:

### Movement
- **Action Space**: Continuous 2D actions in range `[-1, 1]` for x and y directions
- **Velocity**: Actions are scaled by `velocity_scale` to determine movement speed
- **Deadzone**: Small actions below `control_deadzone` are zeroed out (mimics gamepad behavior)
- **Transition Noise**: Optional Gaussian noise can be added to actions for stochastic transitions
- **Collision Resolution**: When colliding with walls, the agent bounces back; position is clipped to room boundaries

### Rendering
- The environment uses Pygame for rendering
- Agent is displayed as a blue square, walls as black rectangles
- Optional player trace ribbon shows movement history

### Layout System
Maps are defined via string layouts where:
- `W` = Wall, `A` = Agent spawn candidate, `G` = Goal candidate (goal tasks only), ` ` = Open space

---

## Goal-Reaching Task (`toy_environment_goal.py`)

Navigate the agent to reach one or more goal positions.

| Aspect | Description |
|--------|-------------|
| **Observations** | **Feature vector**: Agent position (2D) + per-goal: position (2D) + reached flag (1). Optional exogenous noise. **Image**: RGB rendering of the environment (resizable). |
| **State** | Agent position + all goal positions + reached status for each goal |
| **Actions** | Continuous 2D velocity `[-1, 1]` × `[-1, 1]` scaled by `velocity_scale` |
| **Rewards** | `reward_per_reached_goal` (configurable) when reaching a goal; 0 otherwise |
| **Termination** | `any_goal`: Episode ends when any goal is reached. `all_goals`: Episode ends when all goals are reached. |
| **Truncation** | Episode truncates after `max_steps` |

**Goal Ordering**: Goals must be reached in a specified order. Only the current active goal can be reached.

**Randomization Options**:
- `randomise_agent_spawn`: Spawn agent at random valid position vs. from layout candidates
- `randomise_goal_positions`: Spawn goals randomly (`anywhere` or from `candidates` in layout)

---

## Play and Data Collection (`play.py`)

The main entry point for interacting with the toy environment:

```bash
python -m pidm_imitation.environment.toy_env.play \
    --config <PATH/TO/CONFIG> \
    --agent <AGENT_TYPE> \
    --episodes 30
```

### Key CLI Arguments

| Argument | Description |
|----------|-------------|
| `--config` | **(Required)** Path to environment config YAML |
| `--agent` | **(Required)** Agent type: `human`, `random`  |
| `--record` | Enable trajectory recording |
| `--experiment` | Experiment name (required with `--record`) |
| `--output_dir` | Output directory for recorded data (default: `.`) |
| `--episodes` | Number of episodes to play (default: 30) |
| `--only_success` | Only save successful trajectories |

### Recording Example

```bash
python -m pidm_imitation.environment.toy_env.play \
    --config configs/toy_env/toy_env_two_room.yaml \
    --agent planning \
    --record \
    --experiment my_experiment \
    --episodes 50
```

## Agent Types

### Human Agent (`human`)
Control the agent using an Xbox controller (left joystick for movement). A supported controller must be connected to the system before starting the script.

### Random Agent (`random`)
Randomly samples actions from the action space.

## Environment Types

Example configs in `configs/toy_env/`:

| Config | Description |
|--------|-------------|
| `toy_env_two_room.yaml` | Two connected rooms |
| `toy_env_four_room.yaml` | Four-room grid layout |
| `toy_env_multiroom.yaml` | Multiple room maze |
| `toy_env_large_maze.yaml` | Large maze environment |
| `toy_env_zigzag.yaml` | Zigzag corridor |

## Configurability

### Layout

Define map layout with a simple string grid:
- `W` = Wall
- `G` = Goal position
- `A` = Agent spawn position
- ` ` (space) = Open area

```yaml
layout: |
  WWWWWWW
  W     W
  W  G  W
  W     W
  W  A  W
  WWWWWWW
```

### Key Config Options

```yaml
# Goals
num_goals: 1
goal_size: 20

# Termination
termination:
  max_steps: 500
  type: single_goal  # or all_goals

# Observations
observation:
  type: features  # or image
  num_noise_values: 0  # exogenous noise dimensions

# Actions
action:
  control_deadzone: 0.1
  velocity_scale: 5.0

# Rewards
reward:
  goal_reached: 10.0

# Randomization
randomize_agent_spawn: true
randomize_goal_positions: true
```

## Module Structure

- `play.py` - Main play/record script
- `toy_environment_base.py` - Base environment class
- `toy_environment_goal.py` - Goal-reaching environment
- `toy_factory.py` - Environment factory
- `data_collection_agents.py` - Agent implementations for data collection
- `toy_trajectory.py` - Trajectory recording/saving
- `configs/` - Environment configuration classes

## See Also

- `configs/toy_env/` - Example environment configurations
- `pidm_imitation/agents/` - Trained agent evaluation
