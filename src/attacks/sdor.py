import random
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

LOG_STD_MIN = -5
LOG_STD_MAX = 2


class SDorActor(nn.Module):
    """GRU-based stochastic actor.

    Input:  concat(adv_obs, prev_adv_action)  [obs_dim + n_actions]
    Output: â^i in Â^i = {d ∈ [-1,1]^|A|, Σd_j = 0}
            via tanh squashing + zero-mean projection.
    """

    def __init__(self, obs_dim, n_actions, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(obs_dim + n_actions, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_head    = nn.Linear(hidden_dim, n_actions)
        self.log_std_head = nn.Linear(hidden_dim, n_actions)

    def init_hidden(self, n):
        return th.zeros(n, self.hidden_dim)

    def forward(self, x, hidden):
        x      = F.relu(self.fc1(x))
        hidden = self.gru(x, hidden)
        x      = F.relu(self.fc2(hidden))
        mean    = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std, hidden

    def sample(self, x, hidden):
        mean, log_std, h_new = self.forward(x, hidden)
        std  = log_std.exp()
        dist = th.distributions.Normal(mean, std)
        x_t  = dist.rsample()
        action = th.tanh(x_t)
        # log-prob with tanh squashing correction
        log_prob = dist.log_prob(x_t) - th.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        # zero-sum constraint: subtract per-agent mean
        action = action - action.mean(dim=-1, keepdim=True)
        return action, log_prob, h_new


class SDorCritic(nn.Module):
    """Twin Q-network (state-based, no GRU needed for critic)."""

    def __init__(self, obs_dim, n_actions, hidden_dim):
        super().__init__()
        in_dim = obs_dim + n_actions + n_actions  # obs + prev_action + adv_action
        def _mlp():
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        self.q1 = _mlp()
        self.q2 = _mlp()

    def forward(self, obs_prev, action):
        x = th.cat([obs_prev, action], dim=-1)
        return self.q1(x), self.q2(x)


class _ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer   = []
        self._idx     = 0

    def push(self, *transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self._idx % self.capacity] = transition
            self._idx += 1

    def sample(self, k):
        return random.sample(self.buffer, k)

    def __len__(self):
        return len(self.buffer)


class SDorAgent:
    """SAC-based stochastic adversary director.

    Trained offline against a frozen protagonist to learn which policy
    perturbation directions â^i cause maximal long-term damage to team reward.
    """

    def __init__(self, obs_dim, n_actions, n_agents, hidden_dim=64,
                 lr=5e-4, gamma=0.99, tau=0.005,
                 buffer_size=100_000, batch_size=256, device="cpu"):
        self.obs_dim    = obs_dim
        self.n_actions  = n_actions
        self.n_agents   = n_agents
        self.hidden_dim = hidden_dim
        self.gamma      = gamma
        self.tau        = tau
        self.batch_size = batch_size
        self.device     = device
        self._step      = 0

        self.actor          = SDorActor(obs_dim, n_actions, hidden_dim).to(device)
        self.critic         = SDorCritic(obs_dim, n_actions, hidden_dim).to(device)
        self.critic_target  = SDorCritic(obs_dim, n_actions, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.actor_opt  = th.optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = th.optim.Adam(self.critic.parameters(), lr=lr)

        # Auto-tuned entropy temperature
        self.target_entropy = float(-n_actions)
        self.log_alpha      = th.zeros(1, requires_grad=True, device=device)
        self.alpha_opt      = th.optim.Adam([self.log_alpha], lr=lr)

        self.replay  = _ReplayBuffer(buffer_size)
        self._hidden = None  # [n_agents, hidden_dim], set by init_episode()

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def init_episode(self):
        self._hidden = self.actor.init_hidden(self.n_agents).to(self.device)

    def select_action(self, obs_list, prev_adv_action, explore=True):
        """Returns â^i as numpy array [n_agents, n_actions]."""
        obs  = th.FloatTensor(np.stack(obs_list)).to(self.device)
        prev = th.FloatTensor(np.asarray(prev_adv_action, dtype=np.float32)).to(self.device)
        inp  = th.cat([obs, prev], dim=-1)
        with th.no_grad():
            if explore:
                action, _, self._hidden = self.actor.sample(inp, self._hidden)
            else:
                mean, _, self._hidden = self.actor.forward(inp, self._hidden)
                action = th.tanh(mean)
                action = action - action.mean(dim=-1, keepdim=True)
        return action.cpu().numpy()

    def store(self, obs_list, prev_adv, adv_action, reward, next_obs_list, done):
        """Store per-agent transitions; all share the same team reward."""
        for i in range(self.n_agents):
            self.replay.push(
                np.asarray(obs_list[i],    dtype=np.float32),
                np.asarray(prev_adv[i],    dtype=np.float32),
                np.asarray(adv_action[i],  dtype=np.float32),
                float(reward),
                np.asarray(next_obs_list[i], dtype=np.float32),
                np.asarray(adv_action[i],  dtype=np.float32),  # next prev = current action
                float(done),
            )

    def can_update(self):
        return len(self.replay) >= self.batch_size

    def update(self):
        self._step += 1
        batch = self.replay.sample(self.batch_size)
        obs_b, prev_b, act_b, rew_b, nobs_b, nprev_b, done_b = zip(*batch)

        def t(x):
            return th.FloatTensor(np.array(x)).to(self.device)

        obs_b  = t(obs_b);  prev_b  = t(prev_b);  act_b  = t(act_b)
        rew_b  = t(rew_b).unsqueeze(1)
        nobs_b = t(nobs_b); nprev_b = t(nprev_b)
        done_b = t(done_b).unsqueeze(1)

        obs_prev  = th.cat([obs_b,  prev_b],  dim=-1)
        nobs_prev = th.cat([nobs_b, nprev_b], dim=-1)
        # Use zeros for hidden state — standard approximation for recurrent off-policy RL
        h_zero = th.zeros(self.batch_size, self.hidden_dim, device=self.device)

        # Critic target
        with th.no_grad():
            na, nlp, _ = self.actor.sample(nobs_prev, h_zero)
            q1t, q2t  = self.critic_target(nobs_prev, na)
            backup    = rew_b + self.gamma * (1 - done_b) * (th.min(q1t, q2t) - self.alpha * nlp)

        q1, q2 = self.critic(obs_prev, act_b)
        critic_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update
        a, lp, _ = self.actor.sample(obs_prev, h_zero)
        q1a, q2a  = self.critic(obs_prev, a)
        actor_loss = (self.alpha * lp - th.min(q1a, q2a)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Alpha update
        alpha_loss = -(self.log_alpha * (lp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # Soft target update
        with th.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha":       self.alpha,
        }

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        th.save({
            "actor":         self.actor.state_dict(),
            "critic":        self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha":     self.log_alpha.data,
            "obs_dim":       self.obs_dim,
            "n_actions":     self.n_actions,
            "n_agents":      self.n_agents,
            "hidden_dim":    self.hidden_dim,
        }, path / "sdor.pt")

    @classmethod
    def load(cls, path, device="cpu"):
        path = Path(path)
        ckpt = th.load(path / "sdor.pt", map_location=device)
        agent = cls(
            obs_dim=ckpt["obs_dim"],
            n_actions=ckpt["n_actions"],
            n_agents=ckpt["n_agents"],
            hidden_dim=ckpt["hidden_dim"],
            device=device,
        )
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        agent.critic_target.load_state_dict(ckpt["critic_target"])
        agent.log_alpha.data.copy_(ckpt["log_alpha"])
        return agent
