# RL Library Testing

TL;DR: TorchRL is the obvious choice, however it is not an apples to apples comparison. SB3 is designed to be beginner friendly, fast, and allow users to apply standard DRL algorithms to the problems they have. TorchRL is designed for larger scale teams doing research and who need custom algorithms / neural network architectures to achieve their goals. SB3 Provides the standard known algorithms, while in TorchRL we would have to build them ourselves with the building blocks provided to us by the library. These are not mutually exclusive libraries.

There is merit in using a hybrid approach; We can potentially use SB3 for quick prototyping or baseline implementations, and then use TorchRL for more custom or complex algorithms that require greater flexibility and creativity. However TorchRL should be our main focus for the long run.

My full thinking is described below:

TorchRL:
- steeper learning curve
- a lot more customizable
- easier implementation of new ideas
- we are going to have to understand a lot more about RL, DRL, and DL to be able to use torchrl effectively
	- pro and con bc it will take longer but also we will get a deeper understanding
- provides "low and high level abstractions for RL"
	- allowing us to start with high-level components and gradually dive into lower-level PyTorch implementations as they become more comfortable
	- gives users more control over the deep learning aspects of their RL algorithms
- part of the pytorch ecosystem so it will aid us in learning pytorch
	- this will be a high leverage movement as we need to learn pytorch anyways
	- "TorchRL introduces PyTorch-specific primitives like TensorDict, which "facilitates streamlined algorithm development". Working with these PyTorch-centric tools will deepen your team's understanding of PyTorch's ecosystem."
- less stars on github
- designed to work seamlessly with both vectorized and non-vectorized environments

SB3:
- very beginner friendly
- focused on providing implementations of standard DRL learning algorithms
	- better for users that want to quickly apply these given algorithms to their problem
	- these implemenations are described as rigid, and ready to use
- not made for creating custom deep learning algorithms
- will not help us learn pytorch
- designed to be used primarily with gymnasium
- 4x the stars on github
- harder to work with non-vectorized environments

Potential merit in using both:
- Component sharing: potentially using TorchRL's efficient replay buffers or loss functions within an SB3 implementation
- Transitioning: as our project grows more complex, we could start with SB3 and gradually transition to TorchRL for more advanced customization
- Benchmarking: using both libraries could allow for easier comparison and validation of implementations across different frameworks


Lunar Lander in SB3 using DQN algorithm:
```python
import gymnasium as gym
from stable_baselines3 import DQN

# Create the environment
env = gym.make("LunarLander-v2")

# Initialize the DQN agent
model = DQN("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=100000)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Save the model
model.save("dqn_lunar_lander")
```

Lunar Lander in TorchRL using DQN algorithm:
```python
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
```
