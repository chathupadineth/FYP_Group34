class RolloutBuffer:
    """Stores one rollout of experience.

    Three different "done" style signals are kept, because they mean
    different things and are used in different places:

      dones        -- the EPISODE ended here (both robots finished). Used to
                      reset the GRU hidden state during the update, matching
                      how it was reset during collection.

      agent_dones  -- THIS robot finished here (reached its goal, crashed, or
                      timed out). Used for GAE, so value bootstrapping stops
                      at the robot's own terminal step instead of the shared
                      episode boundary.

      active       -- THIS robot was still running at this step. False for
                      every step after it finished. Those steps must be
                      EXCLUDED from the loss: the robot was stopped, its
                      action did nothing, and the 0.0 reward says nothing
                      about that action. Training on them teaches the actor
                      from meaningless samples, and gives the critic two
                      different value targets for the same observation.
    """

    AGENTS = ('jb_0', 'jb_1')

    def __init__(self):
        self.obs = {'jb_0': [], 'jb_1': []}
        self.joint_obs = []
        self.actions = {'jb_0': [], 'jb_1': []}
        self.log_probs = {'jb_0': [], 'jb_1': []}
        self.rewards = {'jb_0': [], 'jb_1': []}
        self.values = []
        self.dones = []
        self.agent_dones = {'jb_0': [], 'jb_1': []}
        self.active = {'jb_0': [], 'jb_1': []}

    def add(self, obs_dict, joint_obs, actions, log_probs, rewards, value, done,
            active=None, agent_dones=None):
        """`active` and `agent_dones` are optional dicts keyed by agent name.
        If omitted, every agent is assumed to be running and to share the
        episode's done flag -- the old behaviour."""
        for name in self.AGENTS:
            self.obs[name].append(obs_dict[name])
            self.actions[name].append(actions[name])
            self.log_probs[name].append(log_probs[name])
            self.rewards[name].append(rewards[name])
            self.active[name].append(True if active is None else bool(active[name]))
            self.agent_dones[name].append(
                bool(done) if agent_dones is None else bool(agent_dones[name]))
        self.joint_obs.append(joint_obs)
        self.values.append(value)
        self.dones.append(done)

    def active_counts(self):
        """How many steps each robot was actually running -- useful for seeing
        how much of a rollout is padding."""
        return {name: sum(self.active[name]) for name in self.AGENTS}

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.dones)
