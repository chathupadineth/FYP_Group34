from gae import compute_gae

rewards = [-0.01, -0.01, -0.01, 10.0]  # 3 normal steps, then reaching goal
values = [0.05, 0.06, 0.08, 0.5]
dones = [False, False, False, True]

advantages, returns = compute_gae(rewards, values, dones)
print("Advantages:", advantages)
print("Returns:", returns)