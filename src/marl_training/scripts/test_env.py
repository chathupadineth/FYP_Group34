import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from jetbot_env import MultiJetBotEnv

def main():
    env = MultiJetBotEnv()

    print("Calling reset()...")
    obs = env.reset()
    for name, o in obs.items():
        print(f"{name} initial observation (len={len(o)}): {o}")

    print("\nRunning 10 random steps...")
    import random
    for step in range(10):
        actions = {name: random.choice([0, 1, 2, 3]) for name in env.agents}
        observations, rewards, dones = env.step(actions)
        print(f"Step {step}: actions={actions}, rewards={rewards}, dones={dones}")

    env.close()
    print("\nDone.")

if __name__ == '__main__':
    main()