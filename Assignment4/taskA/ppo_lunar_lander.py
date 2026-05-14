import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

env = gym.make("LunarLander-v3", render_mode="rgb_array")
env = Monitor(env)

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=3e-4,     # Standard RL learning rate
    n_steps=2048,           # Steps to run per update
    batch_size=64,          # Minibatch size for optimization
    n_epochs=10,            # Number of passes over the training data
    device="auto"           # Uses CUDA if available, else CPU
)

print("Starting training...")
model.learn(total_timesteps=500_000)

model.save("ppo_lunar_lander")
print("Model saved.")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

env = gym.make("LunarLander-v3", render_mode="human")
obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()

env.close()