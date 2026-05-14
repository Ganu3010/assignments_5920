import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback

# Hyperparameters (tracked by wandb)
config = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "total_timesteps": 500_000,
    "env_id": "LunarLander-v3",
    "gamma": 0.9
}

run = wandb.init(
    project="lunar-lander-ppo",
    config=config,
    sync_tensorboard=True,   # Auto-sync SB3's tensorboard metrics
    monitor_gym=True,        # Log env render videos
    save_code=True,
    name="ppo_lunar_lander_run2_bs_64_gamma_0.9"
)

env = gym.make(config["env_id"], render_mode="rgb_array")
env = Monitor(env)

model = PPO(
    policy=config["policy"],
    env=env,
    verbose=1,
    learning_rate=config["learning_rate"],
    n_steps=config["n_steps"],
    batch_size=config["batch_size"],
    n_epochs=config["n_epochs"],
    tensorboard_log=f"runs/{run.id}",  # SB3 logs here; wandb syncs it
    device="auto",
    gamma=config["gamma"]
)

print("Starting training...")
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=100,       # Log gradients every N steps
        model_save_path=f"models/{run.id}",
        model_save_freq=50_000,       # Checkpoint every 50k steps
        verbose=2,
    ),
)

model.save("ppo_lunar_lander")
print("Model saved.")

# Log the final model as a wandb artifact
artifact = wandb.Artifact("ppo_lunar_lander", type="model")
artifact.add_file("ppo_lunar_lander.zip")
wandb.log_artifact(artifact)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
wandb.log({"eval/mean_reward": mean_reward, "eval/std_reward": std_reward})

run.finish()
model = PPO.load("good_model.zip")
env= gym.make(config["env_id"], render_mode="human")
obs, info = env.reset()
for _ in range(2000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()

env.close()