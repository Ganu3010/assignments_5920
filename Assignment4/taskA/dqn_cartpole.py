import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import torch
env = gym.make("CartPole-v1", render_mode="rgb_array")

model = DQN(
    "MlpPolicy",
    env,
    learning_rate=0.0001,
    batch_size=100,
    learning_starts=10,
    target_update_interval=5000,
    exploration_final_eps=0.05,
    exploration_fraction=0.8, 
    policy_kwargs=dict(optimizer_class=torch.optim.Adam),
    verbose=1
)

print("Training started...")
model.learn(total_timesteps=5_000_000, log_interval=1000)
print("Training complete.")

model.save("dqn_cartpole")

eval_env = gym.make("CartPole-v1", render_mode="human")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)

print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

obs, info = eval_env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    eval_env.render()
    if terminated or truncated:
        obs, info = eval_env.reset()

eval_env.close()