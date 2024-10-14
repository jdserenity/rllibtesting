import gymnasium as gym
from stable_baselines3 import DQN, PPO

# Load the saved model
# model = DQN.load("dqn_lunar_lander")
model = PPO.load("ppo_lunar_lander")

# Create the environment with rendering
env = gym.make("LunarLander-v2", render_mode="human")

# Run the trained agent
obs, _ = env.reset()
for _ in range(2000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
