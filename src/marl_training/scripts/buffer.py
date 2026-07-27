class RolloutBuffer:
    def __init__(self):
        self.obs = {'jb_0': [], 'jb_1': []}
        self.joint_obs = []
        self.actions = {'jb_0': [], 'jb_1': []}
        self.log_probs = {'jb_0': [], 'jb_1': []}
        self.rewards = {'jb_0': [], 'jb_1': []}
        self.values = []
        self.dones = []

    def add(self, obs_dict, joint_obs, actions, log_probs, rewards, value, done):
        for name in ['jb_0', 'jb_1']:
            self.obs[name].append(obs_dict[name])
            self.actions[name].append(actions[name])
            self.log_probs[name].append(log_probs[name])
            self.rewards[name].append(rewards[name])
        self.joint_obs.append(joint_obs)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.dones)