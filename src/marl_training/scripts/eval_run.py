import torch
import time
from networks import Actor
from jetbot_env import MultiJetBotEnv, GOAL_REACHED_DIST
from spawn_utils import is_position_valid

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/actor_latest.pt"
MAX_EVAL_STEPS = 200

JB0_START, JB0_GOAL = (0.5, -1.0), (1.6, -1.0)
JB1_START, JB1_GOAL = (2.9, -1.0), (2.9, 0.3)

def check_positions():
    for name, pos in [("jb_0 start", JB0_START), ("jb_0 goal", JB0_GOAL),
                       ("jb_1 start", JB1_START), ("jb_1 goal", JB1_GOAL)]:
        if not is_position_valid(*pos):
            raise ValueError(f"{name} {pos} is invalid (wall/obstacle/out of bounds)")
    print("Fixed positions validated OK.")

def load_actor():
    actor = Actor().to(DEVICE)
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "actor_state_dict" in state:
        state = state["actor_state_dict"]
    actor.load_state_dict(state)
    actor.eval()
    print(f"Loaded weights from {CHECKPOINT_PATH}")
    return actor

def run_episode():
    check_positions()
    actor = load_actor()
    env = MultiJetBotEnv()

    env.step_count = 0
    env.agent_done = {'jb_0': False, 'jb_1': False}
    env.agent_colliding = {'jb_0': False, 'jb_1': False}
    env.prev_distance = {'jb_0': None, 'jb_1': None}
    env._teleport('jb_0', *JB0_START)
    env._teleport('jb_1', *JB1_START)
    env.goals['jb_0'] = JB0_GOAL
    env.goals['jb_1'] = JB1_GOAL
    time.sleep(0.3)
    env._spin_until_fresh()

    hidden = {name: actor.init_hidden(batch_size=1).to(DEVICE) for name in env.agents}

    print(f"\njb_0: {JB0_START} -> {JB0_GOAL} | jb_1: {JB1_START} -> {JB1_GOAL}\n")

    for t in range(MAX_EVAL_STEPS):
        actions = {}
        for name in env.agents:
            if env.agent_done[name]:
                continue
            obs, _, _ = env._build_observation(name)
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).view(1, 1, -1)
            with torch.no_grad():
                logits, hidden[name] = actor(obs_t, hidden[name])
            actions[name] = torch.argmax(logits[0, -1]).item()
            if t % 10 == 0:
                print(f"    -> {name} chose action {actions[name]} (0=fwd,1=left,2=right,3=back)")

        if not actions:
            break

        env.step(actions)

        if t % 10 == 0:
            for name in env.agents:
                _, dist, _ = env._build_observation(name)
                status = "DONE" if env.agent_done[name] else "active"
                print(f"  step {t:3d} | {name}: dist_to_goal={dist:.2f}m [{status}]")

        if all(env.agent_done.values()):
            print(f"\nEpisode ended at step {t}.")
            break
    else:
        print(f"\nHit max steps ({MAX_EVAL_STEPS}) without both finishing.")

    print()
    for name in env.agents:
        _, dist, _ = env._build_observation(name)
        print(f"  {name}: final dist = {dist:.2f}m — {'REACHED' if dist <= GOAL_REACHED_DIST else 'NOT REACHED'}")

    env.close()

if __name__ == "__main__":
    run_episode()
