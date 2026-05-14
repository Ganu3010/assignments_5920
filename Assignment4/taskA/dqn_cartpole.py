import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import torch
import wandb
from wandb.integration.sb3 import WandbCallback

# --- Hyperparameters (single source of truth — logged to wandb) ---
config = dict(
    learning_rate=3e-4,
    batch_size=128,
    buffer_size=50_000,
    learning_starts=1_000,
    target_update_interval=200,
    exploration_final_eps=0.02,
    exploration_fraction=0.15,
    train_freq=4,
    gradient_steps=1,
    gamma=0.99,
    net_arch=[128, 128],
    total_timesteps=100_000,
    n_eval_episodes=10,
)

run = wandb.init(
    project="dqn-cartpole",
    config=config,
    sync_tensorboard=True,   # Auto-sync SB3's tensorboard logs
    monitor_gym=True,        # Auto-log episode videos
    save_code=True,          # Snapshot this script in the run
    name="dqn_cartpole_run_lr3e-4_final"
)

# Pull config back so wandb sweeps can override values
cfg = wandb.config

# --- Environment ---
env = gym.make("CartPole-v1", render_mode="rgb_array")
env = Monitor(env)

# --- Model ---
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=cfg.learning_rate,
    batch_size=cfg.batch_size,
    buffer_size=cfg.buffer_size,
    learning_starts=cfg.learning_starts,
    target_update_interval=cfg.target_update_interval,
    exploration_final_eps=cfg.exploration_final_eps,
    exploration_fraction=cfg.exploration_fraction,
    train_freq=cfg.train_freq,
    gradient_steps=cfg.gradient_steps,
    gamma=cfg.gamma,
    policy_kwargs=dict(
        net_arch=list(cfg.net_arch),
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=dict(eps=1e-5),
    ),
    tensorboard_log=f"runs/{run.id}",  # SB3 writes TB logs; wandb syncs them
    verbose=1,
)

# --- Callbacks ---
wandb_callback = WandbCallback(
    gradient_save_freq=1_000,   # Log gradient histograms every 1k steps
    model_save_path=f"models/{run.id}",
    model_save_freq=25_000,     # Upload a model checkpoint every 25k steps
    verbose=1,
)

# Extra callback: log eval metrics to wandb at the end of training
class EvalLogCallback(BaseCallback):
    def __init__(self, eval_env, n_eval_episodes=10):
        super().__init__()
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes

    def _on_training_end(self):
        mean_reward, std_reward = evaluate_policy(
            self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes
        )
        wandb.log({"eval/mean_reward": mean_reward, "eval/std_reward": std_reward})
        print(f"\nFinal eval — Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    def _on_step(self):
        return True

eval_env_cb = gym.make("CartPole-v1", render_mode="rgb_array")
eval_env_cb = Monitor(eval_env_cb)

# --- Training ---
print("Training started...")
model.learn(
    total_timesteps=cfg.total_timesteps,
    log_interval=100,
    callback=[wandb_callback, EvalLogCallback(eval_env_cb, cfg.n_eval_episodes)],
)
print("Training complete.")

model.save("dqn_cartpole")
wandb.save("dqn_cartpole.zip")  # Upload final model artifact
model = DQN.load("good_model_dqn.zip")
# --- Visual evaluation ---
eval_env = gym.make("CartPole-v1", render_mode="human")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=config['n_eval_episodes'])
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

obs, info = eval_env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    eval_env.render()
    if terminated or truncated:
        obs, info = eval_env.reset()

eval_env.close()
# eval_env_cb.close()
# env.close()
# run.finish()