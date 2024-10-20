import os
import wandb
from mysim import create_rlgym_sim_env
from my_learner import Learner ### my learner which uses modified action space to exclude jumping
from my_metrics_logger import myMetricsLogger

def train_rlgym_ppo_multiple_instances():
    model_save_path = "Model_save_folder"
    save_freq = 50000
    num_instances = 30
    total_timesteps = 10_000_000_000
    os.makedirs(model_save_path, exist_ok=True)
    layer_sizes = (2048, 2048, 1024, 1024)
    env_create_function = create_rlgym_sim_env
    learner = Learner(
        wandb_run_name="awr",
        wandb_project_name="newst",
        wandb_group_name= "hehe",
        #render=True,
        #render_delay=0.05,
        env_create_function=env_create_function,
        n_proc=num_instances,
        timestep_limit=total_timesteps,
        log_to_wandb=True,
        checkpoints_save_folder=model_save_path,
        policy_layer_sizes=layer_sizes,
        critic_layer_sizes=layer_sizes,
        min_inference_size=20,
        exp_buffer_size=32768,
        ppo_batch_size=16384,
        ppo_epochs=10,
        ts_per_iteration=16384,
        ppo_minibatch_size=16384,
        metrics_logger=myMetricsLogger(),
        policy_lr=0.00001,
        critic_lr=0.00001,
        gae_gamma=0.99,
        gae_lambda=0.99,
        ppo_ent_coef=0.01,
        ppo_clip_range=0.0001,
        save_every_ts=save_freq
    )
    learner.load("Model_load_folder", load_wandb=True)
    learner.learn()
    wandb.finish()
if __name__ == "__main__":
    print("Starting training...")
    train_rlgym_ppo_multiple_instances()