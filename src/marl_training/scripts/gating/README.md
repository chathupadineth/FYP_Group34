# gating/ -- Objective 2, standalone

## Status

Not wired into anything. Nothing in `jetbot_env.py`, `networks.py`, `ppo_update.py`,
`train_mappo_curriculum.py`, or any other existing file imports from this folder, and
nothing here imports ROS2/Gazebo. That's deliberate: Objective 1's curriculum training
is still running and being edited (see `training_log.csv`, the various
`checkpoints_*` snapshots, and `claude/odometry-frame-bug.md`), so this folder exists
to let the gate network get designed and shaped up *without* touching any file that
training run depends on.

Per your instruction: this stays disconnected until Objective 1 finishes across every
goal-distance band in the curriculum. Only then does it get attached, and only for
jointly fine-tuning the "full brain" (actor + critic + gate together) -- not before.

## What's here

| file | what it does |
|---|---|
| `gate_network.py` | `GateNet` -- the gate itself. Small feedforward net, 4 inputs -> 2 logits `[silent, talk]`. Untrained (random init). |
| `features.py` | Pure functions that turn two robots' world-frame pose + speed into the 4-value input `GateNet` expects. No ROS2/torch dependency -- works fully offline. |
| `test_gate_network.py` | Sanity check, same style as `test_networks.py` -- run it with `python3 gating/test_gate_network.py` from `scripts/`. No training happens here, just shape/plumbing checks. |

No training script yet, on purpose -- you asked to hold off until Objective 1 produces
real trajectory logs, rather than pretrain the gate on synthetic/rule-based data. Once
those logs exist, a `train_gate.py` slots in here alongside these two files without
changing either of their interfaces.

## How your notes map to this code

- Message `[x, y, theta, v]` -> `features.AgentState(x, y, yaw, v)`. `v` is forward
  speed, i.e. `odom.twist.twist.linear.x` -- already sitting in the existing
  observation as `obs[-2]` (see `jetbot_env.py`'s `sectors + [dist, angle_to_goal, vx, vz]`).
- Gate inputs (relative distance / velocity / heading / valid flag) -> the 4 numbers
  `features.gate_features()` returns, in that order.
- "Separate neural network in each agent" -> `GateNet` is its own `nn.Module`, entirely
  apart from `Actor`/`Critic` in `networks.py`. (Read as: separate *from the
  actor-critic*, not necessarily separate weights per robot -- Objective 1 already
  shares Actor/Critic weights across `jb_0`/`jb_1`; flag if you actually want two
  independently-weighted gates.)
- 0 = silent, 1 = talk -> `gate_network.SILENT = 0`, `gate_network.TALK = 1`, and
  `GateNet.decide()` returns exactly that.
- "If received no messages, slots are empty" -> `gate_features(ego, other_or_none=None, ...)`
  returns an explicit all-zero-but-flagged vector (see the "zero-value trap" note below).

## Open question your notes don't settle

Your notes describe 3 message slots. The current sim only has two robots (`jb_0`,
`jb_1`), so there's only ever one "other" robot -- at most 1 slot in use today. Worth
deciding whether slots 2 and 3 are for a planned scale-up past two robots, or mean
something else (e.g. keeping the last 3 received messages from the same neighbour).
`gate_features()` is written per-pair (one ego, one other) specifically so it doesn't
have to guess -- call it once per neighbour slot, however many end up existing.

## Already solved, don't rebuild

The frame problem that broke goal-distance (`claude/odometry-frame-bug.md`) would
break the gate's relative-distance/heading features the exact same way if they were
computed from raw exchanged odometry. `pose_source.PoseSource.world_pose(name, odom)`
already returns the correct world-frame `(x, y, yaw)` for any robot, ground-truth when
available and odom-anchored otherwise -- `features.py` assumes its inputs came from
there. No new pose-tracking code needed.

## Integration plan for later (documented now, not applied)

When Objective 1 is done and this actually gets attached, in roughly this order:

1. **`jetbot_env.py`** -- in `_build_observation`, after computing `clearance` (which
   already calls `pose_source.world_pose` for the other robot), build
   `gate_features()` for that pair, run `GateNet.decide()`, and populate (or zero out)
   a message slot in the observation depending on the result.
2. **`networks.py`** -- `OBS_DIM` grows to fit the new message slot(s) in the
   observation; `CENTRAL_OBS_DIM` in the critic follows since it's `2 * OBS_DIM`.
3. **`ppo_update.py`** -- add a second `Categorical(logits=...)` head for the gate's
   own action, log_prob, and entropy term, built the same way the existing per-agent
   actor loop already does it for the 4 nav actions -- this is why `GateNet` outputs
   2 logits instead of 1 sigmoid value.
4. **`train_mappo_curriculum.py`** -- this is where "not until all goal-distance
   ranges are done" literally gets enforced: load the actor/critic checkpoint from the
   final curriculum stage, attach a fresh (or by-then-pretrained) `GateNet`, and run
   fine-tuning as a distinct phase rather than folding it into the existing curriculum
   loop.

None of this is applied. It's here so the eventual integration is a checklist against
known files, not a redesign.

## Before training the gate for real

- Pick normalisation constants for `relative_distance`/`relative_velocity` from actual
  logged ranges -- don't guess, `LIDAR_NORM`'s history in `jetbot_env.py` is the
  cautionary tale for why (unscaled inputs 15x apart in magnitude got ignored by the
  policy entirely).
- Decide the label/reward for "dangerous situation" (your Object 2 description) --
  e.g. distance below a threshold while closing, or reuse the existing
  `wall_clearance`/`CONTACT_DIST` collision logic as a template for a robot-to-robot
  version.
- Confirm the slots-vs-neighbours question above before generalising past 2 robots.
