## MAPPO Algorithm Design — Parameters & Reasoning

### Approach decision
- Action space: DISCRETE (not continuous) for this phase
- Observation space: Option B — LIDAR-based sensing only, NOT ground-truth
  relative positions to obstacles
  
### Why LIDAR-only (Option B), not ground-truth positions (Option A)
- Reference paper (Yu et al. 2021) uses MPE/SMAC/Football environments where
  agents get exact ground-truth relative positions to obstacles/agents,
  handed directly by the simulator — no real sensing involved
- This doesn't transfer to real robots: a real JetBot has no built-in
  knowledge of obstacle positions unless someone manually pre-measures and
  hard-codes a static map (fragile, breaks if environment changes, defeats
  autonomy)
- This project's core research gap is specifically about REALISTIC sensor-
  based perception (LIDAR) vs. prior work's idealized/ground-truth channels
- Using Option A would reproduce the exact idealized assumption this
  project is meant to move beyond — so Option B (LIDAR) is the only choice
  consistent with the actual research contribution

---

### Core PPO/MAPPO hyperparameters (from Yu et al. 2021)

| Parameter | Value | Meaning / Why |
|---|---|---|
| Discount factor (γ) | 0.99 | How much future rewards matter vs. immediate ones. 0.99 = future rewards barely discounted — needed because reaching a distant goal requires many correct sequential steps |
| GAE λ | 0.95 | Generalized Advantage Estimation smoothing factor. Balances bias vs. variance when estimating "how much better was this action than average" — 0.95 is the exact value Yu et al. used in their multi-agent experiments |
| PPO clip ε | 0.2 | Limits how much the policy is allowed to change in a single update. Prevents one bad batch of data from destructively overwriting a decent policy |
| Learning rate | 5e-4 | Step size for gradient updates — how aggressively the network weights adjust each update. Matches Yu et al.'s reported value |
| RNN hidden dim | 64 | Size of the recurrent network's internal memory. 64 is small/efficient — appropriate given the observation space isn't huge |
| FC layer dim | 64 | Size of the fully-connected layers in the network. Matches paper; sized appropriately for observation complexity |

---

### Task-specific parameters (decided for this project, not from the paper)

| Parameter | Value | Meaning / Why |
|---|---|---|
| Number of agents | 2 | Matches current 2-JetBot spawn setup |
| Episode length | 200 steps | One episode = 200 simulation steps before reset. Long enough to cross the ~3m platform at JetBot speed, short enough to keep training fast |
| Action space | 4 discrete actions | 0=forward, 1=turn-left, 2=turn-right, 3=stop |
| Reward: goal reached | +10 | Strong positive signal for success, dominates small per-step penalties |
| Reward: collision | -10 | Symmetric penalty — makes collision as undesirable as reaching goal is desirable |
| Reward: per-step | -0.01 | Small time penalty — encourages efficient (shorter) paths without overwhelming the signal early in training |
| Min goal-reach distance | 0.15m | "Close enough to count as arrived" — matches robot's own chassis half-length |

---

### Observation space (per agent) — 16 values total

| # | Component | Range/Format | Meaning / Why |
|---|---|---|---|
| 1–12 | LIDAR sectors (downsampled) | 12 values, normalized [0,1] | Raw LIDAR has 360 samples (1° resolution, real RPLIDAR A1). Downsampled into 12 sectors of 30° each — each value = minimum range reading in that arc, divided by max range (12m). Gives obstacle-awareness without overwhelming the network with 360 raw values |
| 13 | Distance to goal | normalized by platform diagonal (~3.2m) | Relative, scale-invariant — tells the robot how far the goal is, in a 0-1 range regardless of platform size |
| 14 | Angle to goal | normalized to [-1, 1] (from -π to π) | Tells the robot which direction to turn to face the goal |
| 15 | Linear velocity (vx) | current value | Gives the policy awareness of its own current motion state |
| 16 | Angular velocity (vz) | current value | Same reasoning — current turning state |

### Why 12 LIDAR sectors specifically (not 8, not 16, not all 360)
- 360 raw samples: too high-dimensional for a small network to learn
  efficiently from scratch; also highly redundant (adjacent degrees rarely
  differ meaningfully)
- 8 sectors (45° each): too coarse — an obstacle directly ahead vs.
  slightly off-center would look nearly identical, risking missed
  detnatural detections at this platform's small scale (~3m)
- 16 sectors (22.5° each): more precise, but added complexity/training
  time likely unnecessary for a first working version
- 12 sectors (30° each): reasonable middle ground — commonly used at
  this robot/environment scale

### Why normalize all observation values to roughly [-1,1] or [0,1]
Neural networks train far more reliably when inputs are on similar small
scales. Mixing raw meters, raw radians, and raw velocities (very different
natural magnitudes) can destabilize early learning — same principle that
fixed the original reward-stagnation bug (simplifying/normalizing the
observation space unblocked training previously).