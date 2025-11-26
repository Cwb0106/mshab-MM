from mshab.utils.dataset import ClosableDataLoader, ClosableDataset
from mshab.utils.array import to_tensor

from tqdm import tqdm
import gymnasium as gym
import torch
import torch.nn.functional as F
import random
import sys
from dacite import from_dict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
import h5py
import os
import json
from omegaconf import OmegaConf
import numpy as np
import ipdb
import os.path as osp

import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[1]
sys.path.append(str(project_root))

from mshab.envs.make import EnvConfig, make_env
from mshab.utils.config import parse_cfg
from mshab.utils.logger import Logger, LoggerConfig
from mshab.utils.time import NonOverlappingTimeProfiler
from mshab.agents.bc import Agent
from GAPartMobile.models.ACT.act_dataset import ACTCompatibleBCDataset
from GAPartMobile.dataset.dataset import GABCDataset
from GAPartMobile.models.GAMobile_v2.model_loader import HierarchicalCrossAttentionNetwork
import ipdb
from mani_skill.utils import common

PASSED_CONFIG_PATH = osp.join(project_root, "GAPartMobile/config/bc_pick.yml")

# Helper to slice observation dictionary at a specific time step (Shape: [B, ...])
def slice_obs_dict(obs_dict, time_step):
    sliced = {}
    for key, value in obs_dict.items():
        if isinstance(value, torch.Tensor):
            # Assumes time dimension K is always at index 1
            try:
                # Use Ellipsis (...) to handle tensors with different dimensions
                # e.g., [B, K, D] -> [B, D]
                sliced[key] = value[:, time_step, ...] 
            except IndexError as e:
                print(f"Error slicing key '{key}' with shape {value.shape} at time step {time_step}")
                raise e
        elif isinstance(value, dict):
            # Recursively slice for nested dictionaries
            sliced[key] = slice_obs_dict(value, time_step)
        else:
            sliced[key] = value
    return sliced

def save(save_path, model, optimizer):
    torch.save(
        dict(
            agent=model,
            optimizer=optimizer,
        ),
        save_path,
            )


def train(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =======================================================================
    # Create Eval Environment
    # =======================================================================
    print("making eval env")
    eval_envs = make_env(
        cfg.eval_env,
        video_path=cfg.logger.eval_video_path,
    )

    print("made")

    eval_obs, _ = eval_envs.reset(seed=cfg.seed + 1_000_000)
    eval_envs.action_space.seed(cfg.seed + 1_000_000)

    # =======================================================================
    # Load Dataset
    # =======================================================================
    logger = Logger(logger_cfg=cfg.logger)

    bc_dataset = GABCDataset(
        cfg.algo.data_dir_fp,
        cfg.algo.max_cache_size,
        cat_state=cfg.eval_env.cat_state,
        cat_pixels=cfg.eval_env.cat_pixels,
        trajs_per_obj=cfg.algo.trajs_per_obj,
    )

    logger.print(
        f"Made BC Dataset with {len(bc_dataset)} samples at {cfg.algo.trajs_per_obj} trajectories per object for {len(bc_dataset.obj_names_in_loaded_order)} objects",
        flush=True,
    )
    bc_dataloader = ClosableDataLoader(
        bc_dataset,
        batch_size=cfg.algo.batch_size,
        shuffle=True,
        num_workers=5,
    )
    
    # =======================================================================
    # Model and Optimizer
    # =======================================================================
    # --- Prepare Model Parameters ---
    EMBED_DIM = 1024
    NUM_POINTS = 1024
    BASE_ACTION_DIM = 2
    EE_ACTION_DIM = 11
    GOAL_DIM = 7
    STATE_DIM = 42
    # --- Get MEMORY_SLOTS from config ---
    MEMORY_SLOTS = 3

    agent = HierarchicalCrossAttentionNetwork(
            sample_obs=eval_obs,
            embed_dim=EMBED_DIM,
            goal_dim=GOAL_DIM,
            num_points=NUM_POINTS,
            state_dim=STATE_DIM,
            base_action_dim=BASE_ACTION_DIM,
            ee_action_dim=EE_ACTION_DIM,
            clip_model_name="openai/clip-vit-base-patch16",
            memory_slots=MEMORY_SLOTS 
        ).to(device)

    optimizer = torch.optim.Adam(
        agent.parameters(),
        lr=cfg.algo.lr,
    )

    epoch = 0
    logger_start_log_step = logger.last_log_step + 1 if logger.last_log_step > 0 else 0
    timer = NonOverlappingTimeProfiler()

    # Start Training
    for epoch in range(cfg.algo.epochs):

        if epoch + logger_start_log_step > cfg.algo.epochs:
            break
        agent.train()
        logger.print(
            f"Overall epoch: {epoch + logger_start_log_step}; Curr process epoch: {epoch}"
        )

        tot_loss, n_samples = 0, 0
        progress_bar = tqdm(
            bc_dataloader, 
            desc=f"Epoch {epoch + logger_start_log_step}/{cfg.algo.epochs}",
            leave=False
        )

        # --- "Stateful" Training Loop (BPTT) ---
        for batch in progress_bar:
            
            # 1. Unpack sequence data
            #    Assumes dataset returns (obs_seq, act_seq, episode_starts_seq)
            obs_seq, act_seq, episode_starts_seq = batch
            
            # 2. Move data to device (Shape: [B, K, ...])
            obs_seq = to_tensor(obs_seq, device=device, dtype="float")
            act_seq = to_tensor(act_seq, device=device, dtype="float")
            episode_starts_seq = to_tensor(episode_starts_seq, device=device, dtype=torch.bool) # [B, K]

            # 3. Get Batch Size and Sequence Length
            B = act_seq.size(0)
            K = act_seq.size(1) 
            n_samples += (B * K) 

            # 4. Initialize zero memory before the sequence starts
            prev_memory = torch.zeros(B, MEMORY_SLOTS, EMBED_DIM).to(device)

            total_loss_action = 0.0
            total_loss_base = 0.0
            
            # 5. Unroll over time dimension K
            for t in range(K):
                obs_t = slice_obs_dict(obs_seq, t)
                act_t_ground_truth = act_seq[:, t]
                episode_start_t = episode_starts_seq[:, t] # Shape: [B]

                # 6. Reset memory if this is the start of a new episode
                #    (Apply mask to [B, M, D] memory using broadcasting)
                reset_mask = (1.0 - episode_start_t.float()).unsqueeze(-1).unsqueeze(-1)
                prev_memory = prev_memory * reset_mask

                # 7. action
                action_out_t, new_memory, aux_data = agent(obs_t, prev_memory)
                predicted_feature, target_feature = aux_data

                # 8. loss
                loss_action_t = F.mse_loss(action_out_t, act_t_ground_truth)
                loss_base_t = F.mse_loss(predicted_feature, target_feature)
                
                total_loss_action += loss_action_t
                total_loss_base += loss_base_t

                # 9. update memory
                prev_memory = new_memory
                
            # 10. Compute total Loss and Backpropagate *after* K steps
            loss_action = total_loss_action / K
            loss_base = total_loss_base / K
            loss = loss_action + loss_base * 0.1 

            optimizer.zero_grad()
            loss.backward() # Gradients flow through K steps (BPTT)
            optimizer.step()

            tot_loss += loss.item() * K
        
        # 11. Update Loss Log
        loss_logs = dict(loss=tot_loss / n_samples)
            
        timer.end(key="train")

        # Log
        if epoch % cfg.algo.log_freq == 0:
            logger.store(tag="losses", **loss_logs)
            if epoch > 0:
                logger.store("time", **timer.get_time_logs(epoch))
            logger.log(logger_start_log_step + epoch)
            timer.end(key="log")
        
        if cfg.algo.eval_freq:
            if epoch % cfg.algo.eval_freq == 0:
                agent.eval()
                eval_obs, _ = eval_envs.reset()  # don't seed here
                
                # --- Evaluation Loop ---
                # 1. Get number of evaluation environments
                B_eval = cfg.algo.num_eval_envs
                
                # 2. Initialize zero memory at start of Episode
                eval_memory = torch.zeros(B_eval, MEMORY_SLOTS, EMBED_DIM).to(device)

                for _ in range(eval_envs.max_episode_steps):
                 
                    eval_obs = to_tensor(eval_obs, device=device, dtype="float")
                    with torch.no_grad():
                        # 3. Stateful Forward Pass
                        #    Model returns (action, new_memory) in eval mode
                        action, new_eval_memory = agent(eval_obs, eval_memory)
                        
                    eval_obs, _, _, _, _ = eval_envs.step(action)
                    
                    # 4. Pass new memory as input for next step
                    #    .detach() is crucial to prevent gradient accumulation during evaluation
                    eval_memory = new_eval_memory.detach()
                # --- End Evaluation Loop ---

                if len(eval_envs.return_queue) > 0:
                    logger.store(
                        "eval",
                        return_per_step=common.to_tensor(eval_envs.return_queue, device=device)
                        .float()
                        .mean()
                        / eval_envs.max_episode_steps,
                        success_once=common.to_tensor(eval_envs.success_once_queue, device=device)
                        .float()
                        .mean(),
                        success_at_end=common.to_tensor(eval_envs.success_at_end_queue, device=device)
                        .float()
                        .mean(),
                        len=common.to_tensor(eval_envs.length_queue, device=device).float().mean(),
                    )
                    eval_envs.reset_queues()
                logger.log(logger_start_log_step + epoch)
                timer.end(key="eval")


        if epoch % cfg.algo.save_freq == 0:
            if cfg.algo.save_backup_ckpts:
                save(save_path=os.path.join(logger.model_path, f"{epoch}_ckpt.pt"),
                     model=agent.state_dict(),
                     optimizer=optimizer.state_dict())


            save(save_path=os.path.join(logger.model_path, "latest.pt"),
                    model=agent.state_dict(),
                    optimizer=optimizer.state_dict())
            timer.end(key="checkpoint")

    save(save_path=os.path.join(logger.model_path, "final_ckpt.pt"),
            model=agent.state_dict(),
            optimizer=optimizer.state_dict())

    bc_dataloader.close()
    eval_envs.close()
    logger.close()

@dataclass
class BCConfig:
    name: str = "bc"

    # Training
    lr: float = 3e-4
    """learning rate"""
    batch_size: int = 512
    """batch size"""

    # Running
    epochs: int = 100
    """num epochs to run"""
    eval_freq: int = 2
    """evaluation frequency in terms of epochs"""
    log_freq: int = 1
    """log frequency in terms of epochs"""
    save_freq: int = 1
    """save frequency in terms of epochs"""
    save_backup_ckpts: bool = False
    """whether to save separate ckpts eace save_freq which are not overwritten"""

    # Dataset
    data_dir_fp: str = None
    """path to data dir containing data .h5 files"""
    max_cache_size: int = 0
    """max num data points to cache in cpu memory"""
    trajs_per_obj: Union[str, int] = "all"
    """num trajectories to use per object"""
    
    # --- New Config: Memory Slots ---
    memory_slots: int = 16
    """Number of memory slots for the MemoryVLA module"""

    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""

    # passed from env/eval_env cfg
    num_eval_envs: int = field(init=False)
    """the number of parallel environments"""

    def _additional_processing(self):
        assert self.name == "bc", "Wrong algo config"

        try:
            self.trajs_per_obj = int(self.trajs_per_obj)
        except:
            pass
        assert isinstance(self.trajs_per_obj, int) or self.trajs_per_obj == "all"


@dataclass
class TrainConfig:
    seed: int
    eval_env: EnvConfig
    algo: BCConfig
    logger: LoggerConfig

    wandb_id: Optional[str] = None
    resume_logdir: Optional[Union[Path, str]] = None
    model_ckpt: Optional[Union[Path, int, str]] = None

    def __post_init__(self):
        assert (
            self.resume_logdir is None or not self.logger.clear_out
        ), "Can't resume to a cleared out logdir!"

        if self.resume_logdir is not None:
            self.resume_logdir = Path(self.resume_logdir)
            old_config_path = self.resume_logdir / "config.yml"
            if old_config_path.absolute() == Path(PASSED_CONFIG_PATH).absolute():
                assert (
                    self.resume_logdir == self.logger.exp_path
                ), "if setting resume_logdir, must set logger workspace and exp_name accordingly"
            else:
                assert (
                    old_config_path.exists()
                ), f"Couldn't find old config at path {old_config_path}"
                old_config = get_mshab_train_cfg(
                    parse_cfg(default_cfg_path=old_config_path)
                )
                self.logger.workspace = old_config.logger.workspace
                self.logger.exp_path = old_config.logger.exp_path
                self.logger.log_path = old_config.logger.log_path
                self.logger.model_path = old_config.logger.model_path
                self.logger.train_video_path = old_config.logger.train_video_path
                self.logger.eval_video_path = old_config.logger.eval_video_path

            if self.model_ckpt is None:
                self.model_ckpt = self.logger.model_path / "latest.pt"

        if self.model_ckpt is not None:
            self.model_ckpt = Path(self.model_ckpt)
            assert self.model_ckpt.exists(), f"Could not find {self.model_ckpt}"

        self.algo.num_eval_envs = self.eval_env.num_envs
        self.algo._additional_processing()

        self.logger.exp_cfg = asdict(self)
        del self.logger.exp_cfg["logger"]["exp_cfg"]
        del self.logger.exp_cfg["resume_logdir"]
        del self.logger.exp_cfg["model_ckpt"]


def get_mshab_train_cfg(cfg: TrainConfig) -> TrainConfig:
    return from_dict(data_class=TrainConfig, data=OmegaConf.to_container(cfg))



if __name__ == "__main__":
    
    cfg = get_mshab_train_cfg(parse_cfg(default_cfg_path=PASSED_CONFIG_PATH))

    train(cfg)