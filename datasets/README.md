# 2D Navigation Datasets

This directory contains navigation datasets collected from four different 2D navigation environments used for training and evaluating imitation learning models. For implementation of these datasets, see [pidm_imitation/environment/toy_env](pidm_imitation/environment/toy_env/README.md). Each environment contains trajectories from both A\* optimal planners and human demonstrations. Below is an overview of the datasets, their structure, and technical specifications.

| Dataset | Environment | Data Collection Method | Number of Trajectories |
|---------|-------------|------------------------|-----------------------|
| four_room_astar_data | Four-room | A\* | 50 |
| four_room_human_data | Four-room | Human | 50 |
| maze_astar_data | Maze | A\* | 50 |
| maze_human_data | Maze | Human | 50 |
| multiroom_astar_data | Multiroom | A\* | 50 |
| multiroom_human_data | Multiroom | Human | 50 |
| zigzag_astar_data | Zigzag | A\* | 50 |
| zigzag_human_data | Zigzag | Human | 50 |

## Dataset Overview

The datasets comprise navigation trajectories in grid-world environments where an agent (marked as 'A') must navigate to goal locations (marked as 'G') while avoiding walls (marked as 'W'). All environments use continuous control with joystick inputs and support visual observations.

### Environment Types

1. **Four-Room Environment** (`four_room_*_data/`)
4. **Zigzag Environment** (`zigzag_*_data/`)
2. **Maze Environment** (`maze_*_data/`)
3. **Multi-Room Environment** (`multiroom_*_data/`)

### Data Collection Methods

Each environment contains two types of trajectory data:

- **A\* Data** (`*_astar_data/`): Optimal trajectories generated using A\* pathfinding algorithm
- **Human Data** (`*_human_data/`): Trajectories collected from human demonstrations using game controllers

## Dataset Structure

Each dataset contains **50 trajectories** (numbered 00-49), with each trajectory comprising 7 files:

### File Types per Trajectory

1. **`{env}_{method}_{id}_env_config.yaml`** - Environment configuration
   - Layout definition (ASCII grid with walls 'W', agent start 'A', goals 'G')
   - Action space parameters (noise, velocity scaling, control deadzone)
   - Observation settings (image size, noise parameters, observation type)
   - Rendering configuration (agent/goal sizes, trace settings)

2. **`{env}_{method}_{id}_inputs.json`** - Input control data
   - Timestamped controller inputs (Xbox gamepad format)
   - Joystick positions, timing information
   - Raw human inputs or A\* converted to controller format

3. **`{env}_{method}_{id}_observations.npz`** - Observations
   - Compressed numpy arrays of environment observations for each timestep

4. **`{env}_{method}_{id}_states.npz`** - Environment state data
   - Compressed numpy arrays of ground truth state information for each timestep

5. **`{env}_{method}_{id}_trajectory_data.npz`** - Trajectory metadata
   - Episode information, rewards, done flags, additional trajectory data

6. **`{env}_{method}_{id}_video_ticks.json`** - Video timing data
   - Timestamp mapping for video frames
   - Synchronization data between video and other modalities

7. **`{env}_{method}_{id}_video.mp4`** - Rendered trajectory video
   - Visual recording of the complete trajectory

## Technical Specifications

### Action Space
- Continuous 2D control using normalized joystick inputs (-1 to 1)
- Velocity scaling applied (typically 20.0 for both X/Y axes)
- Gaussian noise injection for robustness (σ = 0.2)
- Control deadzone handling (typically 0.1)

### Observation Space
- Feature-based state observations
- Optional goal observation and relative positioning
- Configurable noise injection for domain randomization
