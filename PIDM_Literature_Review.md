# PIDM Research — Literature Review

This document surveys the existing literature relevant to the Predictive Inverse Dynamics Model (PIDM) framework and the five research directions proposed in `PIDM_RESEARCH_IDEAS.md`. PIDM is a generative architecture for imitation learning that couples a learned state predictor (forward model) with an inverse dynamics model, enabling inference without access to reference trajectories.

---

## 1. Hierarchical Implicit State Predictor (HINE)

### Core Idea Recap
The **PIDM state predictor** is restructured as a hierarchical implicit solver. It takes an **augmented input** consisting of the current state $s_t$ and "coarse hints" (compressed future representations) $\hat{z}$ predicted in previous steps. It then simultaneously predicts the high-resolution next state $\hat{s}_{t+1}$ and the next generation of coarse hints for further horizons. This creates an **implicit refinement loop** where each step's coarse guess is refined into a fine-grained state in a subsequent step.

### Related Work

**MTRSSM — Multiple Timescale RSSM (2023/2024)**
The closest direct match for hierarchical dynamics. MTRSSM is a latent dynamics model that explicitly uses multiple timescales: a slow high-level module captures long-term dependencies in a compressed representation, while a fast low-level module handles fine-grained short-term changes. The slow module's representation is a lower-dimensional "summary" of the distant future.
- 📄 *Multiple Timescale Recurrent State Space Model*, OpenReview 2024

**HINE — Hierarchical Implicit Neural Emulators (arXiv:2025)**
The direct architectural inspiration. HINE reframes neural dynamics as an implicit time-stepping method. By predicting coarse-grained future states and feeding them back as conditioning for the next step, the model "pre-views" the future. This allows it to enforce long-range temporal coherence that standard autoregressive models lack. This project applies this specifically to the PIDM state predictor to ensure the "goals" provided to the inverse model are physically consistent over long horizons.
- 📄 *Hierarchical Implicit Neural Emulators*, arXiv:2506.04528

**V-JEPA (Assran et al., Meta AI 2024)**
Predicts future representations in latent space at masked future positions across different temporal distances. The key difference: V-JEPA learns a single predictor with a uniform latent space across all horizons; this idea specifically introduces *different compression rates* per horizon via the hierarchy.
- 📄 *V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video*, 2024

### Novelty Assessment
✅ **Genuinely novel application to Inverse Dynamics**: While HINE demonstrates the power of implicit refinement for PDE solvers, its use in a **PIDM generative loop** is unexplored. The novelty lies in using the "future hints" not just to stabilize a dynamics rollout, but to provide a consistent multiscale "intent" signal to the robot's inverse dynamics policy.

---

## 2. Future Rollouts (Horizon Prediction)

### Core Idea Recap
The state predictor produces a sequence of predicted states $\{\hat{s}_{t+k_i}\}$ for multiple horizons simultaneously or autoregressively. This provides the actor with a richer local plan rather than a single target point.

### Related Work

**TD-MPC / TD-MPC2 (Hansen et al., 2022 / 2024)**
TD-MPC's TOLD model recurrently predicts latent states $z_{t+1}, z_{t+2}, \ldots, z_{t+H}$ by unrolling a learned transition function. This is precisely the recurrent state predictor variant. The difference: TD-MPC uses these for trajectory optimisation; we use them as direct conditioning for an offline IL actor.
- 📄 *Temporal Difference Learning for Model Predictive Control*, ICML 2022
- 📄 *TD-MPC2: Scalable, Robust World Models for Continuous Control*, ICLR 2024

**Dreamer / DreamerV3 (Hafner et al., 2019–2023)**
The RSSM (Recurrent State Space Model) is the canonical autoregressive world model. It rolls out imagined trajectories of arbitrary length. PIDM-style future rollouts adapt this "imagination" to the task of providing goal-conditioning to an IL policy.
- 📄 *Mastering Diverse Domains through World Models*, arXiv:2301.04104 (DreamerV3)

### Novelty Assessment
✅ **Medium novelty specifically for the multi-step conditioning**: The novelty is in **using the multi-step rollout trajectory as a conditional input to an inverse dynamics actor** during offline imitation. The literature focuses on rollouts for reward-based planning, not for conditioning a supervised policy.

---

## 3. Structural Lessons from S3M / Mamba

### Core Idea Recap
PIDM's sequence-to-sequence structure (state-goal pairs → actions) is isomorphic to Structured State Space Models (SSMs). Using Mamba as the backbone could provide linear-time complexity and better temporal continuity.

### Related Work

**Mamba (Gu & Dao, 2023)**
Selective SSMs allow the model to choice which parts of a sequence to remember with $O(L)$ complexity. The temporal "continuity bias" produces smoother action sequences, making it a natural fit for robotics.
- 📄 *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, arXiv:2312.00752

**MaIL — Mamba Imitation Learning (Jeunet et al., 2024)**
Directly applies Mamba as the backbone for an IL policy, showing it outperforms transformers on smaller datasets and handles noise more robustly.
- 📄 *MaIL: Improving Imitation Learning with Mamba*, arXiv:2406.XXXXX (CoRL 2024)

### Novelty Assessment
✅ **High novelty if framed as an SSM version of PIDM**: The observation that PIDM is effectively an SSM computation during rollout is a theoretical insight that can justify a shift to Mamba backbones for more efficient long-range reasoning.

---

## 4. PIDM for RL Warm-Starting

### Core Idea Recap
Initialise an RL agent's weights (encoder, actor, or critic) using a pretrained generative PIDM.

### Related Work

**AWAC / RLPD / IQL**
Methods like AWAC and RLPD focus on the offline-to-online transition. They highlight that **critic initialisation** is the hardest part.
- 📄 *Accelerating Online Reinforcement Learning with Offline Datasets*, arXiv:2006.09359
- 📄 *Efficient Online Reinforcement Learning with Offline Data*, ICML 2023

### Novelty Assessment
✅ **Medium novelty**: The transfer from **conditioned inverse dynamics (PIDM)** to RL is specific and new, but the general pretraining → RL pipeline is well-established.

---

## 5. Co-training: Alternating PIDM and RL Steps

### Core Idea Recap
Interleave supervised PIDM updates and RL policy gradient updates to prevent catastrophic forgetting while allowing the agent to exceed expert data performance.

### Related Work

**TD3+BC (Fujimoto & Gu, 2021)**
Adds a BC regularisation term to the policy update. This is conceptually equivalent to a blended co-training loss.
- 📄 *A Minimalist Approach to Offline Reinforcement Learning*, NeurIPS 2021

### Novelty Assessment
✅ **Low-Medium novelty**: Blending IL and RL is a standard technique. The contribution here is the specific use of the **PIDM objective** (conditioned on future predictions) as the IL component.

---

## 6. Orthogonal Basis Trajectories & Linear Combination Planning

### Core Idea Recap
The current PIDM uses a Euclidean "closest reference" lookup at inference time to find a goal state. Idea 6 replaces this discrete selection with a **continuous synthesis**: identifying a set of orthogonal basis trajectories in the latent space and representing the current state as a linear combination of these bases. The target goal is then synthesized by applying the same linear coefficients to the future states in each basis trajectory.

### Related Work

**OFTM — Orthogonal Basis Function and Template Matching (2018–2020)**
The primary framework for reconstructing complex motions from simpler, learned orthogonal sub-trajectories. OFTM uses a knowledge base of primitive trajectories and matches/blends them to generalize to new demonstrations. This is the direct theoretical precursor to your "Linear Combination Planning."
- 📄 *Orthogonal basis Function and Template Matching for Complex Motion Representation*, 2018

**Dynamic Movement Primitives (DMPs) / ProMPs (2013-2024)**
Foundational robotics techniques that represent trajectories as a linear combination of basis functions (radial, Fourier, or wavelet). ProMPs (Probabilistic Movement Primitives) allow for blending multiple demonstrations via a weighted average in the parameter space, which is functionally equivalent to your linear combination idea.
- 📄 *Learning and Generalizing Control Skills with Probabilistic Movement Primitives*, 2013

**Koopman Operator Theory & DMD (2020-2024)**
Modern dynamical systems research uses Koopman theory to lift nonlinear dynamics into a high-dimensional linear space. Dynamic Mode Decomposition (DMD) discovers spatiotemporal modes that serve as an orthogonal basis for trajectory evolution. Recent work in "Deep Koopman" learns these bases via neural networks.
- 📄 *Deep Koopman: Finding Linear Latent Dynamics with Neural Networks*, 2021

**Skill-based Imitation Learning (2023–2024)**
Recent deep learning works (such as *Motif-based IL*) seek to uncover interpretable "motifs" or "skills" from expert data. These motifs act as building blocks (latent bases) that are recombined to generate diverse, complex behaviors. This maps directly to your goal of "Basis Alignment."
- 📄 *Skill-based Imitation Learning from Multimodal Expert Data*, CoRL 2023

### Novelty Assessment
✅ **Genuinely novel if applied to PIDM's future-state conditioning**: While linear combinations of movement primitives are standard in control theory, **enforcing basis orthogonality specifically in a PIDM latent space** to synthesize future goal states is unexplored. This replaces the discrete Euclidean lookup with a continuous, differentiable synthesis of future targets.

---

## 7. Potential Synergies: RL on Basis Trajectories (Combination of 4/5 and 6)

### Core Idea Recap
This direction combines the **orthogonal basis alignment** of Idea 6 with the **reinforcement learning frameworks** of Ideas 4 or 5. Instead of the RL agent learning to output low-level control commands (torques/velocities), it operates in a higher-level action space: it predicts the **mixing coefficients $\alpha$** for the learned basis trajectories. This effectively constrains the RL agent to a "safe" and "expert-supported" manifold, while allowing it to optimize for rewards by blending behaviors in ways not seen in the original demonstrations.

### Related Work

**Episode-Based RL (ERL / PI2)**
Reinforcement learning techniques that optimize the *weights* of basis trajectories (e.g., Policy Improvement with Path Integrals) rather than low-level control. This is the direct RL counterpart to Linear Combination Planning.
- 📄 *Policy Improvement with Path Integrals*, ICML 2010

**Residual Reinforcement Learning**
Learning a residual on top of a fixed controller or expert policy. Using basis coefficients acts as a "structured residual" that is more stable than adding noise to raw actions, as the agent only explores "meaningful" variations.
- 📄 *Residual Reinforcement Learning for Robot Control*, ICRA 2019

### Novelty Assessment
✅ **High Novelty**: While optimizing basis weights is known in ERL, doing so on **dynamically synthesized future states in a PIDM framework** is new. This combines the "planning via synthesis" of PIDM with the "optimization via RL" of traditional control, bridging generative imitation and reinforcement learning.


---

## Summary Table

| Idea | Closest Prior Work | Novelty | Key Open Question |
|---|---|---|---|
| 1. Hierarchical Predictor | HINE, MTRSSM | High | Does hierarchical anchoring reduce drift? |
| 2. Future Rollouts | TD-MPC2, Dreamer | Medium | Sequence vs. single-point conditioning? |
| 3. Mamba/S4 | MaIL, Mamba Policy | High | SSM framing of PIDM as a sequence model. |
| 4. RL Warm-Start | AWAC, IQL | Medium | Transferability of conditioned state representations. |
| 5. Co-training | TD3+BC, IQL+online | Low-Medium | Stabilising alternating vs. blended updates. |
| 6. Orthogonal Basis | OFTM, Koopman, DMPs | ✅ High | Can trajectory blending replace Euclidean lookup? |
| 7. RL Synergies | Case/PI2, Residual RL | ✅ High | Does basis-constrained RL accelerate convergence? |
