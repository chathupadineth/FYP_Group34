"""
pretrain_bc.py  --  Behavioural Cloning from the scripted controller.

Turns nav_dataset.npz (collected by collect_dataset.py, where the A* scripted
controller drives) into a starting point for MAPPO, so PPO does not have to
discover "steer toward the goal, stop short of walls" from random weights.

    python3 pretrain_bc.py                       # train and write checkpoints
    python3 pretrain_bc.py --epochs 40
    python3 pretrain_bc.py --dry-run             # inspect the data, write nothing

WHY THIS IS NOT "TRAINING ON THE LOG"
-------------------------------------
PPO cannot ingest scripted data directly: every sample needs the log-prob of
the action UNDER THE POLICY THAT CHOSE IT, and the script is not a policy with
log-probs. So the scripted data is used the only way it validly can be -- as
supervised learning on (observation -> action) pairs, which is Behavioural
Cloning. The result is an actor, not a value function and not a PPO update.

SEQUENCES, NOT SHUFFLED SAMPLES
-------------------------------
The Actor contains a GRU: its output depends on hidden state carried from
earlier steps. Training it on shuffled independent rows would teach it to act
with a hidden state it will never actually have. So the dataset is regrouped
into per-(episode, agent) trajectories and each is rolled through in order,
exactly as jetbot_env will run it.

AFTER THIS
----------
The BC actor is written as checkpoints/actor_latest.pt with a FRESH critic and
last_update = 0, so `python3 train_mappo.py` picks it up as its starting point.
The critic starts random, which is why train_mappo has CRITIC_WARMUP_UPDATES:
without it, the first PPO updates would compute advantages from a meaningless
value function and wreck the cloned policy in a handful of steps.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from networks import Actor, Critic, OBS_DIM, ACTION_DIM, OWN_OBS_DIM

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, 'nav_dataset.npz')
CHECKPOINT_DIR = os.path.join(HERE, 'checkpoints')

ACTION_NAMES = ['forward', 'left', 'right', 'backward']


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_trajectories(path, success_only=True):
    """Regroup flat rows into ordered per-(episode, agent) sequences."""
    d = np.load(path, allow_pickle=True)
    obs = d['obs'].astype(np.float32)
    act = d['action'].astype(np.int64)
    ep = d['episode'].astype(np.int64)
    ag = d['agent'].astype(np.int64)
    step = d['step'].astype(np.int64)

    if obs.shape[1] != OBS_DIM:
        raise SystemExit(
            f"Dataset has {obs.shape[1]}-value observations but the network "
            f"expects {OBS_DIM}. The dataset was collected with a different "
            f"jetbot_env -- re-run collect_dataset.py.")

    keep = np.ones(len(obs), dtype=bool)
    if success_only and 'ep_success' in d and 'ep_agent' in d:
        # Only clone runs the script actually completed. Cloning its failures
        # teaches the policy to fail the same way.
        #
        # CAREFUL with the indexing. collect_dataset.py appends TWO rows to the
        # episode table per episode -- one per robot -- while the samples'
        # `episode` column counts EPISODES. So episode e, agent a lives at table
        # row (2*e + a), not at row e. Treating the row index as an episode id
        # silently kept only half the runs, alternating jb_0 / jb_1.
        succ = d['ep_success'].astype(np.int64)
        eag = d['ep_agent'].astype(np.int64)
        n_eps = int(ep.max()) + 1
        ok = set()
        if len(succ) == 2 * n_eps:
            for e in range(n_eps):
                for a in (0, 1):
                    row = 2 * e + a
                    if int(eag[row]) == a and succ[row]:
                        ok.add((e, a))
        else:
            # Unexpected layout -- fall back to matching on (row, its agent)
            # rather than guessing, and say so.
            print(f"  ! episode table has {len(succ)} rows for {n_eps} episodes; "
                  f"using all runs")
            ok = None

        if ok is not None:
            keep = np.array([(int(e), int(a)) in ok for e, a in zip(ep, ag)])
            dropped = len(obs) - int(keep.sum())
            if dropped:
                print(f"  dropped {dropped} samples from unsuccessful runs")
        if keep.sum() == 0:
            print("  ! no successful episodes found; using all of them instead")
            keep = np.ones(len(obs), dtype=bool)

    groups = {}
    for i in np.nonzero(keep)[0]:
        groups.setdefault((int(ep[i]), int(ag[i])), []).append(i)

    trajs = []
    for key, idx in groups.items():
        idx = sorted(idx, key=lambda j: step[j])
        trajs.append((obs[idx], act[idx]))
    return trajs, d


def describe(trajs, tag):
    n_steps = sum(len(a) for _, a in trajs)
    counts = np.zeros(ACTION_DIM, dtype=np.int64)
    for _, a in trajs:
        counts += np.bincount(a, minlength=ACTION_DIM)
    lens = [len(a) for _, a in trajs]
    print(f"  {tag}: {len(trajs)} trajectories, {n_steps} steps "
          f"(median length {int(np.median(lens))})")
    print("    expert action mix: " + ", ".join(
        f"{ACTION_NAMES[i]} {100*counts[i]/max(1,n_steps):.0f}%"
        for i in range(ACTION_DIM)))
    return counts, n_steps


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def run_epoch(actor, trajs, loss_fn, optimizer=None):
    total_loss, total_correct, total_steps = 0.0, 0, 0
    order = np.random.permutation(len(trajs))
    for k in order:
        o, a = trajs[k]
        obs_t = torch.from_numpy(o).view(1, -1, OBS_DIM)
        act_t = torch.from_numpy(a)
        hidden = actor.init_hidden(1)
        logits, _ = actor(obs_t, hidden)          # roll the whole trajectory
        logits = logits.view(-1, ACTION_DIM)
        loss = loss_fn(logits, act_t)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            optimizer.step()
        total_loss += loss.item() * len(a)
        total_correct += (logits.argmax(dim=1) == act_t).sum().item()
        total_steps += len(a)
    return total_loss / max(1, total_steps), total_correct / max(1, total_steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=DEFAULT_DATA)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--all-episodes', action='store_true',
                    help='clone failed scripted runs too (default: successes only)')
    ap.add_argument('--dry-run', action='store_true',
                    help='report on the dataset and exit without writing anything')
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(f"No dataset at {args.data}\n"
                         f"Run:  python3 collect_dataset.py --episodes 120")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    trajs, raw = load_trajectories(args.data, success_only=not args.all_episodes)
    if not trajs:
        raise SystemExit("Dataset contains no usable trajectories.")

    print(f"Loaded {args.data}")
    counts, n_steps = describe(trajs, 'all')

    # A LIDAR sanity check. If the dataset was collected before the
    # normalisation fix its sector values sit near 0.02-0.24 and the policy it
    # produces will be useless in the current environment.
    sect = np.concatenate([o[:, :12].ravel() for o, _ in trajs])
    print(f"    LIDAR sector values: min={sect.min():.3f} "
          f"median={np.median(sect):.3f} max={sect.max():.3f} std={sect.std():.3f}")
    if sect.std() < 0.05:
        print("    ! These are squashed (std < 0.05). This dataset predates the")
        print("      LIDAR_NORM fix -- re-collect it before training on it.")

    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(trajs))
    n_val = max(1, int(args.val_frac * len(trajs))) if len(trajs) > 4 else 0
    val = [trajs[i] for i in idx[:n_val]]
    train = [trajs[i] for i in idx[n_val:]]
    print(f"  split: {len(train)} train / {len(val)} validation trajectories")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    # The expert is unbalanced (mostly 'forward'). Without class weighting the
    # clone learns to answer 'forward' to everything, which scores well on this
    # data and steers straight into walls in the environment.
    #
    # An action the expert NEVER used gets weight 1.0, not a huge number. The
    # scripted controller never drives backward, so 'backward' has zero samples;
    # weighting it by n_steps/(4*0) would be meaningless, and it never appears
    # as a target anyway. PPO explores that action later via the entropy bonus.
    weights = torch.tensor(
        [1.0 if c == 0 else n_steps / (ACTION_DIM * c) for c in counts],
        dtype=torch.float32)
    unused = [ACTION_NAMES[i] for i, c in enumerate(counts) if c == 0]
    if unused:
        print(f"  note: expert never used {', '.join(unused)} -- the clone will")
        print(f"        not produce it, PPO has to discover it by exploration")
    print("  class weights: " + ", ".join(
        f"{ACTION_NAMES[i]} {weights[i]:.2f}" for i in range(ACTION_DIM)))

    actor = Actor()
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)

    best_acc, best_state = -1.0, None
    print()
    for epoch in range(1, args.epochs + 1):
        actor.train()
        tr_loss, tr_acc = run_epoch(actor, train, loss_fn, optimizer)
        if val:
            actor.eval()
            with torch.no_grad():
                va_loss, va_acc = run_epoch(actor, val, loss_fn, None)
        else:
            va_loss, va_acc = tr_loss, tr_acc
        flag = ''
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.detach().clone() for k, v in actor.state_dict().items()}
            flag = '  <- best'
        print(f"  epoch {epoch:>3}/{args.epochs}  train loss {tr_loss:.4f} "
              f"acc {100*tr_acc:5.1f}%  |  val loss {va_loss:.4f} "
              f"acc {100*va_acc:5.1f}%{flag}")

    actor.load_state_dict(best_state)
    print(f"\nBest validation accuracy: {100*best_acc:.1f}%")
    if best_acc < 0.55:
        print("  ! Below 55%. The clone barely agrees with the script, so it will")
        print("    not give PPO much of a head start. More episodes usually helps.")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    for name in ('actor_latest.pt', 'critic_latest.pt', 'last_update.txt',
                 'curriculum_stage.txt'):
        p = os.path.join(CHECKPOINT_DIR, name)
        if os.path.exists(p):
            raise SystemExit(
                f"\n{CHECKPOINT_DIR} already holds a run ({name} exists).\n"
                f"Archive it first so a finished run is never silently overwritten:\n"
                f"    mv checkpoints checkpoints_<something>\n")

    # ---- Zero the communication columns before saving -------------------
    # The dataset's message slot is all zeros on every sample (the scripted
    # expert never transmits), so the gradient reaching fc1.weight[:, 16:20] is
    # exactly zero on every step -- those four columns finish training still
    # holding their RANDOM INITIALISATION. nn.Linear(20, 64) initialises to
    # uniform(-0.224, 0.224), mean |w| ~= 0.112, which is the same order as the
    # weights the clone actually learned. So the moment the gate is switched on
    # and real messages arrive, four randomly-weighted inputs hit the network
    # and the cloned policy is corrupted by noise it was never trained against.
    #
    # Zeroing them makes the clone a true no-communication policy: messages
    # contribute exactly nothing until PPO decides otherwise. Same reasoning as
    # migrate_checkpoint.py's zero-init, for the same reason.
    with torch.no_grad():
        actor.fc1.weight[:, OWN_OBS_DIM:].zero_()
    print(f"  message columns [{OWN_OBS_DIM}:{OBS_DIM}] zeroed "
          f"(the clone is a no-communication policy)")

    torch.save(actor.state_dict(), os.path.join(CHECKPOINT_DIR, 'actor_latest.pt'))
    torch.save(Critic().state_dict(), os.path.join(CHECKPOINT_DIR, 'critic_latest.pt'))
    torch.save(actor.state_dict(), os.path.join(CHECKPOINT_DIR, 'actor_bc.pt'))
    with open(os.path.join(CHECKPOINT_DIR, 'last_update.txt'), 'w') as f:
        f.write('0')
    with open(os.path.join(CHECKPOINT_DIR, 'curriculum_stage.txt'), 'w') as f:
        f.write('0\n0\n')

    print(f"\nWrote {CHECKPOINT_DIR}/")
    print("  actor_latest.pt   the cloned policy -- train_mappo.py starts here")
    print("  actor_bc.pt       an untouched copy, so you can always compare against it")
    print("  critic_latest.pt  fresh (BC produces no value function)")
    print("\nNext:  python3 evaluate_policy.py --episodes 20 --seed 0")
    print("       python3 train_mappo.py")


if __name__ == '__main__':
    main()
