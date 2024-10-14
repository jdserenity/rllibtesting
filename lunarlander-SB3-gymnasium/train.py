import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

# Create the Lunar Lander environment
env = gym.make("LunarLander-v2", render_mode="rgb_array")

# Instantiate the DQN agent
model = DQN("MlpPolicy", env, learning_rate=1e-4, verbose=1)

# Train the agent
model.learn(total_timesteps=100000)

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Save the model
model.save("dqn_lunar_lander")
