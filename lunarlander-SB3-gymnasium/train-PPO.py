import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import torch

def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Create vectorized environment
    env = make_vec_env("LunarLander-v2", n_envs=16)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Create separate environment for evaluation
    eval_env = make_vec_env("LunarLander-v2", n_envs=5)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Set up callback
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                 log_path='./logs/', eval_freq=10000,
                                 deterministic=True, render=False, n_eval_episodes=10)

    # Define policy kwargs
    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
    )

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        target_kl=None,
        device=device,
        policy_kwargs=policy_kwargs,
        verbose=1
    )

    # Train the model
    model.learn(total_timesteps=1_000_000, callback=eval_callback)

    # Save the final model
    model.save("ppo_lunar_lander_1m")

    # Close environments
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
