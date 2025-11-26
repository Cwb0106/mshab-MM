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

sys.path.append("/raid/wenbo/project/mshab/")
# from GAPartMobile.dataset.dataset import BCDataset
from mshab.envs.make import EnvConfig, make_env
from mshab.utils.config import parse_cfg
from mshab.utils.logger import Logger, LoggerConfig
from mshab.utils.time import NonOverlappingTimeProfiler
from mshab.agents.bc import Agent
from GAPartMobile.models.ACT.act_dataset import ACTCompatibleBCDataset
from GAPartMobile.dataset.dataset_backup import GABCDataset
import ipdb
from mani_skill.utils import common


def save(save_path, model, optimizer):
    torch.save(
        dict(
            agent=model,
            optimizer=optimizer,
        ),
        save_path,
            )

class BCDataset(ClosableDataset):
    def __init__(
        self,
        data_dir_fp: str,
        max_cache_size: int,
        transform_fn=torch.from_numpy,
        trajs_per_obj: Union[str, int] = "all",
        cat_state=True,
        cat_pixels=False,
        num_queries=2,
    ):
        self.num_queries = num_queries
        data_dir_fp: Path = Path(data_dir_fp)
        self.data_files: List[h5py.File] = []
        self.json_files: List[Dict] = []
        self.obj_names_in_loaded_order: List[str] = []

        if data_dir_fp.is_file():
            data_file_names = [data_dir_fp.name]
            data_dir_fp = data_dir_fp.parent
        else:
            data_file_names = os.listdir(data_dir_fp)
        for data_fn in data_file_names:
            if data_fn.endswith(".h5"):
                json_fn = data_fn.replace(".h5", ".json")
                self.data_files.append(h5py.File(data_dir_fp / data_fn, "r"))
                with open(data_dir_fp / json_fn, "rb") as f:
                    self.json_files.append(json.load(f))
                self.obj_names_in_loaded_order.append(data_fn.replace(".h5", ""))

        self.dataset_idx_to_data_idx = dict()
        dataset_idx = 0
        # NOTE (arth): for the rearrange dataset, each json/h5 file contains trajectories for one object
        for file_idx, json_file in enumerate(self.json_files):

            # sample trajectories to use by trajs_per_obj
            if trajs_per_obj == "all":
                use_ep_jsons = json_file["episodes"]
            else:
                assert trajs_per_obj <= len(
                    json_file["episodes"]
                ), f"got {trajs_per_obj=} but only have {len(json_file['episodes'])} for data for obj={self.obj_names_in_loaded_order[file_idx]}"
                use_ep_jsons = random.sample(json_file["episodes"], k=trajs_per_obj)

            for ep_json in use_ep_jsons:
                ep_id = ep_json["episode_id"]
                # 因为有了填充逻辑，我们可以安全地遍历所有时间步
                for step in range(ep_json["elapsed_steps"]):
                    self.dataset_idx_to_data_idx[dataset_idx] = (file_idx, ep_id, step)
                    dataset_idx += 1

            # for ep_json in use_ep_jsons:
            #     ep_id = ep_json["episode_id"]
            #     for step in range(ep_json["elapsed_steps"]):
            #         self.dataset_idx_to_data_idx[dataset_idx] = (file_idx, ep_id, step)
            #         dataset_idx += 1
        self._data_len = dataset_idx

        self.max_cache_size = max_cache_size
        self.cache = dict()

        self.transform_fn = transform_fn
        self.cat_state = cat_state
        self.cat_pixels = cat_pixels

    def transform_idx(self, x, data_index):
        if isinstance(x, h5py.Group) or isinstance(x, dict):
            return dict((k, self.transform_idx(v, data_index)) for k, v in x.items())
        out = self.transform_fn(np.array(x[data_index]))
        if len(out.shape) == 0:
            out = out.unsqueeze(0)
        return out
    


    def get_single_item(self, index):
        if index in self.cache:
            return self.cache[index]

        file_num, ep_num, step_num = self.dataset_idx_to_data_idx[index]
        ep_data = self.data_files[file_num][f"traj_{ep_num}"]
     
        observation = ep_data["obs"]
        agent_obs = self.transform_idx(observation["agent"], step_num)
        extra_obs = self.transform_idx(observation["extra"], step_num)
        # unsqueeze to emulate a single frame stack
        fetch_head_depth = (
            self.transform_idx(
                observation["sensor_data"]["fetch_head"]["depth"], step_num
            )
            .squeeze(-1)
            .unsqueeze(0)
        )
        fetch_hand_depth = (
            self.transform_idx(
                observation["sensor_data"]["fetch_hand"]["depth"], step_num
            )
            .squeeze(-1)
            .unsqueeze(0)
        )

        # NOTE (arth): this works for seq task envs, but may not work for generic env obs
        state_obs = (
            dict(
                state=torch.cat(
                    [
                        *agent_obs.values(),
                        *extra_obs.values(),
                    ],
                    axis=0,
                )
            )
            if self.cat_state
            # else dict(agent_obs=agent_obs)
            else dict(agent_obs=agent_obs, extra_obs=extra_obs)
        )
        pixel_obs = (
            dict(all_depth=torch.stack([fetch_head_depth, fetch_hand_depth], axis=-3))
            if self.cat_pixels
            else dict(
                fetch_head_depth=fetch_head_depth,
                fetch_hand_depth=fetch_hand_depth,
            )
        )
    
        obs = dict(**state_obs, pixels=pixel_obs)
        act = self.transform_idx(ep_data["actions"], step_num)

        res = (obs, act)
        if len(self.cache) < self.max_cache_size:
            self.cache[index] = res
        return res

    def __getitem__(self, indexes):
        if isinstance(indexes, int):
            return self.get_single_item(indexes)
        return [self.get_single_item(i) for i in indexes]

    def __len__(self):
        return self._data_len

    def close(self):
        for f in self.data_files:
            f.close()


def train(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =======================================================================
    # 创建eval环境
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
    # 加载数据集
    # =======================================================================
    logger = Logger(logger_cfg=cfg.logger)


    # bc_dataset = BCDataset(
    #     cfg.algo.data_dir_fp,
    #     cfg.algo.max_cache_size,
    #     cat_state=cfg.eval_env.cat_state,
    #     cat_pixels=cfg.eval_env.cat_pixels,
    #     trajs_per_obj=cfg.algo.trajs_per_obj,
    # )

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
        num_workers=2,
    )
    

    # =======================================================================
    # 模型和优化器
    # =======================================================================
    agent = Agent(eval_obs, eval_envs.unwrapped.single_action_space.shape).to(device)
    # from GAPartMobile.models.GAMobile.model_loader import HierarchicalCrossAttentionNetwork
    # EMBED_DIM = 1024  # 网络内部统一的特征维度
    # NUM_POINTS = 1024
    # BASE_ACTION_DIM = 2
    # EE_ACTION_DIM = 11
    # GOAL_DIM = 7
    # STATE_DIM = 42
    # agent = HierarchicalCrossAttentionNetwork(
    #         sample_obs=eval_obs,
    #         embed_dim=EMBED_DIM,
    #         goal_dim=GOAL_DIM,
    #         num_points=NUM_POINTS,
    #         state_dim=STATE_DIM,
    #         base_action_dim=BASE_ACTION_DIM,
    #         ee_action_dim=EE_ACTION_DIM,
    #         clip_model_name="openai/clip-vit-base-patch16"
    #     ).to(device)


    optimizer = torch.optim.Adam(
        agent.parameters(),
        lr=cfg.algo.lr,
    )


    epoch = 0
    logger_start_log_step = logger.last_log_step + 1 if logger.last_log_step > 0 else 0
    timer = NonOverlappingTimeProfiler()

    # =======================================================================
    # 开始训练
    # =======================================================================
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
            leave=False # 完成后清除进度条，保持界面整洁
        )

        for obs, act in progress_bar:
            # Action: 0-7: arm, 8: gripper, 9-11: head, head, torso, 12-13: base
            obs, act = to_tensor(obs, device=device, dtype="float"), to_tensor(
                act, device=device, dtype="float"
            )
      
            n_samples += act.size(0)  # 这是batch？
            
            pi = agent(obs)

            loss = F.mse_loss(pi, act)

            # loss_action = F.mse_loss(pi[0], act)
            # loss_base = F.mse_loss(pi[1], pi[2])
            # loss = loss_action + loss_base * 0.1

            # loss = loss_dict['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tot_loss += loss.item()
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
        # if cfg.algo.eval_freq > -1:
        #      if epoch % cfg.algo.eval_freq > -1:
                agent.eval()
                eval_obs, _ = eval_envs.reset()  # don't seed here

                for _ in range(eval_envs.max_episode_steps):
                 
                    eval_obs = to_tensor(eval_obs, device=device, dtype="float")
                    with torch.no_grad():
                        action = agent(eval_obs)
                        # action = agent.sample(eval_obs)
                    eval_obs, _, _, _, _ = eval_envs.step(action)

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

        save(save_path=os.path.join(logger.model_path, f"{epoch}_ckpt.pt"),
                model=agent.state_dict(),
                optimizer=optimizer.state_dict())
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

    # args = OmegaConf.load("/raid/wenbo/project/mshab/GAPartMobile/config/train_bc.yaml")
    PASSED_CONFIG_PATH = "/raid/wenbo/project/mshab/GAPartMobile/config/bc_pick.yml"
    cfg = get_mshab_train_cfg(parse_cfg(default_cfg_path=PASSED_CONFIG_PATH))

    train(cfg)
   


