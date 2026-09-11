"""
migrate_checkpoint.py  --  carry a pre-gating checkpoint into the gated network.

    python3 migrate_checkpoint.py --from checkpoints_run2_lidarfix_terminal \
                                  --tag update350 --out checkpoints

THE PROBLEM
-----------
Adding the communication slot grew the observation from 16 values to 20, so
Actor.fc1 went from Linear(16, 64) to Linear(20, 64) and Critic.fc1 from
Linear(32, 64) to Linear(40, 64). A saved 16-input weight matrix is shape
(64, 16); the new layer wants (64, 20). torch's load_state_dict refuses it
outright -- so update 350's policy, the best one this project has produced,
cannot simply be loaded.

THE FIX
-------
Copy the old weights into the columns that still mean the same thing, and set
the new columns to EXACTLY ZERO:

    new_fc1.weight[:, :16] = old_fc1.weight        # unchanged meaning
    new_fc1.weight[:, 16:] = 0.0                   # the message slot
    new_fc1.bias            = old_fc1.bias
    (every other layer copies across unchanged -- their shapes never moved)

A zero column contributes nothing to the layer's output, so on the first step
after migration the network computes bit-for-bit what it computed before.
The migrated policy starts at update 350's success rate, not at zero, and then
learns what to do with the four new inputs from there.

This works because jetbot_env appends the message slot at the END of the
observation. If it were inserted anywhere else, the old columns would line up
against the wrong inputs and the weights would be meaningless.

WHAT IT DOES NOT DO
-------------------
It does not create a gate. GateNet starts random; train_mappo.py handles that.
It does not touch the source checkpoint directory -- everything is written to
--out, so the original stays intact as a rollback point.
"""

import argparse
import os
import shutil

import torch

from networks import Actor, Critic, OWN_OBS_DIM, OBS_DIM, MSG_DIM, CENTRAL_OBS_DIM

OLD_OBS_DIM = OWN_OBS_DIM              # 16
OLD_CENTRAL_DIM = 2 * OWN_OBS_DIM      # 32


def expand_linear(old_w, old_b, new_in, keep_in):
    """Return (weight, bias) for a wider input layer, old columns preserved
    and the rest zeroed."""
    out_features = old_w.shape[0]
    w = torch.zeros(out_features, new_in, dtype=old_w.dtype)
    w[:, :keep_in] = old_w[:, :keep_in]
    return w, old_b.clone()


def migrate_actor(state):
    a = Actor()
    new = a.state_dict()
    for k in new:
        if k == 'fc1.weight':
            continue
        if k == 'fc1.bias':
            new[k] = state['fc1.bias'].clone()
            continue
        if k not in state:
            raise SystemExit(f"actor checkpoint is missing {k}")
        if state[k].shape != new[k].shape:
            raise SystemExit(f"actor {k}: shape {tuple(state[k].shape)} vs "
                             f"{tuple(new[k].shape)} -- not just an input-width change")
        new[k] = state[k].clone()
    ow = state['fc1.weight']
    if ow.shape[1] != OLD_OBS_DIM:
        raise SystemExit(f"actor fc1 expects {OLD_OBS_DIM} inputs, checkpoint has "
                         f"{ow.shape[1]} -- wrong checkpoint for this migration")
    new['fc1.weight'], _ = expand_linear(ow, state['fc1.bias'], OBS_DIM, OLD_OBS_DIM)
    a.load_state_dict(new)
    return a


def migrate_critic(state):
    c = Critic()
    new = c.state_dict()
    for k in new:
        if k == 'fc1.weight':
            continue
        if k == 'fc1.bias':
            new[k] = state['fc1.bias'].clone()
            continue
        if k not in state:
            raise SystemExit(f"critic checkpoint is missing {k}")
        if state[k].shape != new[k].shape:
            raise SystemExit(f"critic {k}: shape {tuple(state[k].shape)} vs "
                             f"{tuple(new[k].shape)}")
        new[k] = state[k].clone()
    ow = state['fc1.weight']
    if ow.shape[1] != OLD_CENTRAL_DIM:
        raise SystemExit(f"critic fc1 expects {OLD_CENTRAL_DIM} inputs, checkpoint "
                         f"has {ow.shape[1]}")
    # The joint observation is jb_0's 20 values then jb_1's 20. The old 32 was
    # jb_0's 16 then jb_1's 16, so the two halves have to be placed separately
    # -- copying all 32 columns straight across would put jb_1's old weights
    # on top of jb_0's message slot.
    w = torch.zeros(ow.shape[0], CENTRAL_OBS_DIM, dtype=ow.dtype)
    w[:, :OWN_OBS_DIM] = ow[:, :OWN_OBS_DIM]                       # jb_0 own
    w[:, OBS_DIM:OBS_DIM + OWN_OBS_DIM] = ow[:, OWN_OBS_DIM:]      # jb_1 own
    new['fc1.weight'] = w
    c.load_state_dict(new)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='src', required=True,
                    help='checkpoint directory to read')
    ap.add_argument('--tag', default='update350',
                    help='which snapshot inside it (e.g. update350, latest)')
    ap.add_argument('--out', default='checkpoints',
                    help='directory to write the migrated checkpoint to')
    ap.add_argument('--start-update', type=int, default=0,
                    help='what to write into last_update.txt (0 restarts the '
                         'numbering for the gated run, which keeps the two '
                         'training logs from overlapping)')
    args = ap.parse_args()

    ap_path = os.path.join(args.src, f'actor_{args.tag}.pt')
    cp_path = os.path.join(args.src, f'critic_{args.tag}.pt')
    for p in (ap_path, cp_path):
        if not os.path.exists(p):
            raise SystemExit(f"not found: {p}")

    actor_state = torch.load(ap_path, map_location='cpu')
    critic_state = torch.load(cp_path, map_location='cpu')

    print(f"Reading {args.src} / {args.tag}")
    print(f"  actor  fc1.weight {tuple(actor_state['fc1.weight'].shape)} "
          f"-> ({actor_state['fc1.weight'].shape[0]}, {OBS_DIM})")
    print(f"  critic fc1.weight {tuple(critic_state['fc1.weight'].shape)} "
          f"-> ({critic_state['fc1.weight'].shape[0]}, {CENTRAL_OBS_DIM})")

    actor = migrate_actor(actor_state)
    critic = migrate_critic(critic_state)

    # Prove the migration is behaviour-preserving before writing anything:
    # with the message slot zeroed, the wide network must produce exactly what
    # the narrow one did.
    from networks import HIDDEN_DIM
    import torch.nn as nn

    class NarrowActor(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(OLD_OBS_DIM, HIDDEN_DIM)
            self.rnn = nn.GRU(HIDDEN_DIM, HIDDEN_DIM, batch_first=True)
            self.fc_out = nn.Linear(HIDDEN_DIM, actor.fc_out.out_features)

        def forward(self, o, h):
            x = torch.relu(self.fc1(o))
            x, h = self.rnn(x, h)
            return self.fc_out(x), h

    narrow = NarrowActor()
    narrow.load_state_dict(actor_state)
    narrow.eval(); actor.eval()
    torch.manual_seed(0)
    probe16 = torch.randn(1, 5, OLD_OBS_DIM)
    probe20 = torch.cat([probe16, torch.zeros(1, 5, MSG_DIM)], dim=-1)
    with torch.no_grad():
        a_old, _ = narrow(probe16, torch.zeros(1, 1, HIDDEN_DIM))
        a_new, _ = actor(probe20, torch.zeros(1, 1, HIDDEN_DIM))
    diff = (a_old - a_new).abs().max().item()
    print(f"  behaviour check: max output difference {diff:.3e}")
    if diff > 1e-6:
        raise SystemExit("migration changed the policy -- refusing to write")
    print("  -> identical. The migrated policy starts exactly where it left off.")

    os.makedirs(args.out, exist_ok=True)
    torch.save(actor.state_dict(), os.path.join(args.out, 'actor_latest.pt'))
    torch.save(critic.state_dict(), os.path.join(args.out, 'critic_latest.pt'))
    torch.save(actor.state_dict(), os.path.join(args.out, f'actor_migrated_{args.tag}.pt'))
    torch.save(critic.state_dict(), os.path.join(args.out, f'critic_migrated_{args.tag}.pt'))
    with open(os.path.join(args.out, 'last_update.txt'), 'w') as f:
        f.write(str(args.start_update))

    # Carry the curriculum rung across, so the gated run starts on the same
    # goal-distance band rather than silently dropping back to 1.0 m.
    src_stage = os.path.join(args.src, 'curriculum_stage.txt')
    dst_stage = os.path.join(args.out, 'curriculum_stage.txt')
    if os.path.exists(src_stage):
        shutil.copyfile(src_stage, dst_stage)
        with open(dst_stage) as f:
            print(f"  curriculum stage carried over: {f.readline().strip()}")
    else:
        print("  no curriculum_stage.txt in source -- the gated run will start "
              "at stage 0 (1.0 m). Set it by hand if that is wrong.")

    print(f"\nWrote {args.out}/")
    print("  actor_latest.pt, critic_latest.pt   <- train_mappo.py resumes from these")
    print(f"  actor_migrated_{args.tag}.pt        <- untouched copy for reference")
    print(f"  last_update.txt = {args.start_update}")
    print("\nThe gate is NOT created here; train_mappo.py starts it fresh.")


if __name__ == '__main__':
    main()
