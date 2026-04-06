# PIDM Research Ideas and Implementation Plan

This document outlines six research directions for enhancing the Predictive Inverse Dynamics Model (PIDM) framework. Each idea is designed to be implemented as an optional configuration to allow for comparative testing against the baseline.

## 1. Hierarchical Implicit State Predictor (HINE)
**Concept**: Transform the forward dynamics model into a hierarchical implicit solver that refines predictions over time. Instead of predicting the next state from scratch, the predictor uses "coarse hints" (latent representations) of the future predicted in earlier steps to anchor and refine the current state prediction. This is inspired by **Hierarchical Implicit Neural Emulators (HINE)** ([arXiv:2506.04528](https://arxiv.org/abs/2506.04528)).

### Conceptual Framework
The state predictor $f_{pred}$ is modified to operate on an **augmented state** and produce **multi-scale outputs**:
1.  **Augmented Input**: At step $t$, the predictor receives the current state $s_t$ plus hierarchical "hints" $\hat{z}$ predicted in previous steps: $Input_t = (s_t, \hat{z}_{t+1}^{(1)}, \hat{z}_{t+2}^{(2)}, \dots)$.
2.  **Hierarchical Output**: The predictor outputs the full-resolution next state $\hat{s}_{t+1}$ and the *next* generation of coarse hints for further future horizons: $Output_t = (\hat{s}_{t+1}, \hat{z}_{t+2}^{(1)}, \hat{z}_{t+3}^{(2)}, \dots)$.
3.  **Hierarchical Compression**: As the lookahead distance $l$ increases, the representation $z^{(l)}$ becomes more compressed (e.g., via spatial downsampling or lower latent dimensionality). 
4.  **Implicit Refinement Loop**: This architecture effectively executes an implicit refinement process across the autoregressive rollout. A coarse "preview" of $s_{t+1}$ (predicted at $t-1$) serves as a boundary condition that $f_{pred}$ must satisfy when generating the high-resolution $\hat{s}_{t+1}$ at step $t$.

**Impact on Actor**: The actor $\pi(a_t | s_t, \hat{s}_{t+k})$ can now be conditioned on the entire hierarchy of future states/latents $(\hat{s}_{t+1}, \hat{z}_{t+2}^{(1)}, \dots, \hat{z}_{t+k}^{(l)})$ providing it with multi-scale "intent" that is temporally consistent.

### Relevant Files & Code Blocks
*   **[slicer.py](file:///home/martyna/Documents/Avi/pidm_warm_start_rl/pidm_imitation/agents/supervised_learning/dataset/slicer.py)**:
    *   **Modification**: Implement `HierarchicalSlicer`. This slicer will yield a sequence of indices at different downsampling levels (e.g., $t+1$ at resolution $R$, $t+2$ at resolution $R/8$, $t+3$ at resolution $R/32$).
*   **[sliding_window_dataset.py](file:///home/martyna/Documents/Avi/pidm_warm_start_rl/pidm_imitation/agents/supervised_learning/dataset/sliding_window_dataset.py)**:
    *   **Modification**: Update `_get_lookahead_data` to return ground-truth hierarchical future states (compressed versions) to supervise the predictor's auxiliary heads.
*   **[state_encoder_model.py](file:///home/martyna/Documents/Avi/pidm_warm_start_rl/pidm_imitation/agents/supervised_learning/submodels/state_encoder_model.py)**:
    *   **Modification**: Update `StatePredictorModel` (likely within or paired with `StateEncoderModel`) to handle multiple heads for different compression levels and augmented inputs.
*   **[policy_heads.py](file:///home/martyna/Documents/Avi/pidm_warm_start_rl/pidm_imitation/agents/supervised_learning/submodels/policy_heads.py)**:
    *   **Modification**: Create `HierarchicalPolicyHead` to fuse the tokens from various multiscale embeddings using a cross-attention or MLP-based refinement layer.

---

## 2. Future Rollouts (Horizon Prediction)
**Concept**: Instead of targeting a single state at $t+k$, the policy receives a sequence (horizon) of predicted future states. These states can be weighted or discounted based on their temporal distance.

In this setting we can predict any set of predicted future states. This can be a continuous sequence or some random assortment (k=1,3,7,10). For this we could also use different state predictor heads for each horizon prediction and maybe they share weights. This would give it an implicit heirarchical structure as well. If doing a continuous sequence then we can train across different sequences as well to get the same kind of robustness as the current PIDM which trains with different k values.

We could also feed back in the predicted states from the previous time step to predict the next time step. This would be like a recurrent state predictor. 

### Relevant Files & Code Blocks
*   **[sliding_window_dataset.py](file:///home/martyna/Documents/Avi/pidm_warm_start_rl/pidm_imitation/agents/supervised_learning/dataset/sliding_window_dataset.py)**:
    *   **Modification**: Redefine the dataset to return a `STATE_HORIZON_KEY` containing a tensor of shape `(batch, horizon_len, state_dim)`. Update `lookahead_indices` calculation in `_get_lookahead_data` (line 259).
*   **[policy_heads.py](file:///home/martyna/Documents/Avi/pidm_warm_start_rl/pidm_imitation/agents/supervised_learning/submodels/policy_heads.py)**:
    *   **Modification**: Create a new `HorizonPolicyHead` class. Its `forward` method should implement the discounting logic for the sequence of future goals.

---

## 3. Structural Lessons from S3M (Mamba/S4)
**Concept**: The PIDM structure (mapping $s_t, s_{t+k} \rightarrow a_t$) shares fundamental similarities with Structured State Space Models (S3M/S4). The input-output flowchart looks similar except S3Ms work on sequences all at once and I don't know if PIDMs do that - maybe they do, if they don't we could make them do that and then take advantage of convolution style training. This might already exist in the literature.

### Relevant Files & Code Blocks
*   **[base_models.py](file:///home/martyna/Documents/Avi/pidm_warm_start_rl/pidm_imitation/agents/supervised_learning/base_models.py)**:
    *   **Insight**: Research incorporating the S4 "parallel scan" during training to process state-goal pairs more efficiently or using the recurrent form for low-latency inference.
*   **[state_encoder_model.py](file:///home/martyna/Documents/Avi/pidm_warm_start_rl/pidm_imitation/agents/supervised_learning/submodels/state_encoder_model.py)**:
    *   **Modification**: Adapt the encoder to maintain a latent state that is "pushed" by the current state and "pulled" by the future lookahead, mimicking the transition matrices ($A, B$) in S3M.
*   **[layer_types.py](file:///home/martyna/Documents/Avi/pidm_warm_start_rl/pidm_imitation/agents/models/layer_types.py)**:
    *   **Modification**: Add `"s4"` or `"mamba"` support to allow the architecture to benefit from high-order temporal reasoning.

---

## 4. PIDM for RL Warm-Starting
**Concept**: Use a pre-trained PIDM as an initialization for RL. This provides the agent with a "warm start" on understanding environment transitions and goal-oriented behaviors before reward-based exploration. This also might exist in the literature, good to check, couldn't find at first search.

Unsure if this would be used as initialisation for actor or critic.

### Relevant Files & Code Blocks
*   **[model_factory.py](file:///home/martyna/Documents/Avi/pidm_warm_start_rl/pidm_imitation/agents/supervised_learning/model_factory.py)**:
    *   **Task**: Implement a weight-loading utility that extracts the `state_encoder` and `policy_head` from a supervised checkpoint into an RL-compatible policy network.
*   **[train_utils.py](file:///home/martyna/Documents/Avi/pidm_warm_start_rl/pidm_imitation/agents/supervised_learning/utils/train_utils.py)**:
    *   **Modification**: Support partial weight loading (e.g., loading only the encoder) to allow the RL agent to start with pre-tuned features while learning a new policy.

---

## 5. Co-training: Alternating PIDM and RL Steps
**Concept**: Train the same policy using both supervised PIDM losses and reinforcement learning losses in an alternating fashion. This prevents the "forgetting" of expert behavior while allowing the agent to surpass expert performance through RL. This also might exist in the literature, good to check, couldn't find at first search.

### Relevant Files & Code Blocks
*   **[base_models.py](file:///home/martyna/Documents/Avi/pidm_warm_start_rl/pidm_imitation/agents/supervised_learning/base_models.py)**:
    *   **Modification**: Update `training_step` (line 250) to accept a "step type" flag. This allows the trainer to alternate between a `supervised_step` (using offline data) and an `rl_step` (using environment rewards).
*   **[datamodule.py](file:///home/martyna/Documents/Avi/pidm_warm_start_rl/pidm_imitation/agents/supervised_learning/dataset/datamodule.py)**:
    *   **Modification**: Implement a hybrid `train_dataloader` that combines the `SlidingWindowDataset` with an online replay buffer.
*   **[train.py](file:///home/martyna/Documents/Avi/pidm_warm_start_rl/pidm_imitation/agents/supervised_learning/train.py)**:
    *   **Modification**: Integrate an environment rollout loop within the PyTorch Lightning training process to collect RL data on-the-fly.

---

## 6. Orthogonal Basis Trajectories & Linear Combination Planning
**Concept**: Address the limitation of the current inference-time "Closest Reference" lookup. Instead of selecting a single nearest demonstration, represent the current state as a linear combination of states across multiple basis-aligned trajectories.

### Conceptual Framework
1.  **Basis Alignment**: Structure the latent state space (via specialized encoders or loss functions) such that distinct expert demonstration trajectories are encouraged to be approximately orthogonal basis vectors.
2.  **Linear Combination Planning**: During inference, instead of a winner-take-all Euclidean lookup, project the current state $s_t$ onto the trajectory basis to find coefficients $\alpha_1, \alpha_2, \dots$. The target lookahead $z_{t+k}$ is then synthesized as $\sum \alpha_i \cdot s^{(i)}_{t+k}$.

### Literature & Theoretical Foundations
*   **Movement Primitives (DMPs/ProMPs)**: Foundational work in robotics that represents trajectories as a linear combination of basis functions (e.g., radial basis functions).
*   **Koopman Operator Theory**: A mathematical framework that linearizes nonlinear dynamics by lifting them into a high-dimensional functional space, effectively finding a "linear basis" for trajectory evolution.
*   **Dynamic Mode Decomposition (DMD)**: A data-driven method for discovering spatiotemporal basis "modes" within complex dynamical systems, allowing for future state prediction via linear combinations of these modes.
*   **Manifold Learning & Orthogonality**: Recent deep learning research into "Orthogonal Latent Spaces" and "Polar Decomposition-based Initialization" aims to enforce independence between learned features, which aligns with the goal of "Basis Alignment."

### Relevant Files & Code Blocks
*   **[idm_planners.py](file:///home/martyna/Documents/Avi/pidm_warm_start_rl/pidm_imitation/agents/supervised_learning/inference_agents/idm/idm_planners.py)**:
    *   **Modification**: Implement `BasisCombinationPlanner` subclassing `InstanceBasedPlanner`. Replace the `_get_lookahead_state_and_action` logic (line 260) with a weighted average over masked reference states.
*   **[state_distances.py](file:///home/martyna/Documents/Avi/pidm_warm_start_rl/pidm_imitation/agents/supervised_learning/inference_agents/idm/state_distances.py)**:
    *   **Modification**: Add a `BasisProjection` metric that returns coefficients for the linear combination rather than a single distance value.
*   **[model_factory.py](file:///home/martyna/Documents/Avi/pidm_warm_start_rl/pidm_imitation/agents/supervised_learning/model_factory.py)**:
    *   **Modification**: Update the factory to support new architecture variants that enforce trajectory orthogonality (e.g., via a contrastive loss or specific layer constraints).

---

3.  **Warm-start**: The agent can be initialized with $\alpha$ values that perfectly reconstruct a specific expert trajectory (Idea 4), then optimized further.

### Relevant Files & Code Blocks
*   **[idm_planners.py](file:///home/martyna/Documents/Avi/pidm_warm_start_rl/pidm_imitation/agents/supervised_learning/inference_agents/idm/idm_planners.py)**:
    *   **Modification**: Create `RLBasisPlanner`. This planner will wrap the `BasisCombinationPlanner` and allow an external RL agent to provide the $\alpha$ coefficients at each step.
*   **[base_models.py](file:///home/martyna/Documents/Avi/pidm_warm_start_rl/pidm_imitation/agents/supervised_learning/base_models.py)**:
    *   **Modification**: Add a specific `RL_basis_loss` that encourages the agent to find sparse combinations of bases when rewards are similar, improving interpretability.
