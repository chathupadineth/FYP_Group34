from buffer import RolloutBuffer

buf = RolloutBuffer()

fake_obs = {'jb_0': [0.1]*16, 'jb_1': [0.2]*16}
fake_joint_obs = [0.1]*16 + [0.2]*16
fake_actions = {'jb_0': 0, 'jb_1': 2}
fake_log_probs = {'jb_0': -1.3, 'jb_1': -1.4}
fake_rewards = {'jb_0': -0.01, 'jb_1': -0.01}
fake_value = 0.05
fake_done = False

buf.add(fake_obs, fake_joint_obs, fake_actions, fake_log_probs, fake_rewards, fake_value, fake_done)
buf.add(fake_obs, fake_joint_obs, fake_actions, fake_log_probs, fake_rewards, fake_value, fake_done)

print("Buffer length:", len(buf))
print("jb_0 rewards stored:", buf.rewards['jb_0'])
print("Joint obs count:", len(buf.joint_obs))