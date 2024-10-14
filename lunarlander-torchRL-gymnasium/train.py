import torch
import gymnasium as gym
from torchrl.envs import GymEnv
from torchrl.modules import MLP, QValueActor
from torchrl.objectives import DQNLoss
from torchrl.data import TensorDictReplayBuffer

# Create the environment
env = GymEnv("LunarLander-v2")

# Create the Q-network
q_net = MLP(
    in_features=env.observation_spec.shape[-1],
    out_features=env.action_spec.space.n,
    num_cells=[64, 64],
)
actor = QValueActor(q_net, in_keys=["observation"])

# Create the loss module
loss_module = DQNLoss(actor)

# Create the optimizer
optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

# Create the replay buffer
replay_buffer = TensorDictReplayBuffer(storage=torch.empty(10000, *env.observation_spec.shape))

# Training loop
for _ in range(100000):
    td = env.reset()
    done = False
    while not done:
        action = actor(td).sample()
        next_td = env.step(action)
        replay_buffer.add(td)
        td = next_td
        done = td["done"].item()
        
        if len(replay_buffer) > 1000:
            batch = replay_buffer.sample(256)
            loss = loss_module(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Evaluate the agent
total_reward = 0
for _ in range(10):
    td = env.reset()
    done = False
    while not done:
        action = actor(td).sample()
        td = env.step(action)
        total_reward += td["reward"].item()
        done = td["done"].item()

print(f"Mean reward: {total_reward / 10:.2f}")

# Save the model
torch.save(actor.state_dict(), "dqn_lunar_lander.pth")