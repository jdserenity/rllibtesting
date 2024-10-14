import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

# Initialize Ray (if not already done)
import ray
ray.init()

# Configure the algorithm
config = (
    PPOConfig()
    .environment("LunarLander-v2")
    .rollouts(num_rollout_workers=2)
    .framework("torch")
    .training(model={"fcnet_hiddens": [64, 64]})
    .evaluation(evaluation_num_workers=1)
)

# Build the algorithm
algo = config.build()

# Train for 10 iterations
for i in range(10):
    result = algo.train()
    print(f"Iteration {i}:")
    print(pretty_print(result))

    if i % 5 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")

# Run the trained model
env = gym.make("LunarLander-v2", render_mode="human")
episode_reward = 0
terminated = truncated = False
obs, _ = env.reset()

while not terminated and not truncated:
    action = algo.compute_single_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward

print(f"Episode reward: {episode_reward}")

# Cleanup
algo.stop()
ray.shutdown()