# Copyright 2020-2022 OpenDR European Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import gym
import numpy as np
import os

from pathlib import Path

from stable_baselines3.common.utils import configure_logger
from stable_baselines3.common.vec_env import VecEnv
from opendr.control.multi_object_search.algorithm.SB3.ppo import PPO_AUX
from opendr.control.multi_object_search.algorithm.SB3.encoder import EgocentricEncoders
from igibson.utils.utils import parse_config
from typing import Optional
from urllib.request import urlretrieve
from opendr.control.multi_object_search.algorithm.evaluation import evaluation_rollout
from opendr.control.multi_object_search.algorithm.SB3.save_model_callback import SaveModel

from opendr.engine.constants import OPENDR_SERVER_URL
from opendr.engine.learners import LearnerRL


from igibson.utils.assets_utils import download_assets,download_ig_dataset,download_demo_data
import shutil
import igibson

class ExplorationRLLearner(LearnerRL):
    def __init__(self, env: gym.Env, lr=1e-5, iters=6_000_000, batch_size=64, lr_schedule='',
                 lr_end: float = 1e-6, backbone='MultiInputPolicy', checkpoint_after_iter=20_000, checkpoint_load_iter=0,
                 restore_model_path: Optional[str] = None, temp_path='', device='cuda', seed: int = None,
                 gamma: float = 0.99,
                 buffer_size: int = 2048,
                 config_filename='',nr_evaluations: int = 50):
        """
        Specifies a soft-actor-critic (SAC) agent that can be trained for mobile manipulation.
        Internally uses Stable-Baselines3 (https://github.com/DLR-RM/stable-baselines3).
        """
        super(LearnerRL, self).__init__(lr=lr, iters=iters, batch_size=batch_size, optimizer='adam',
                                        lr_schedule=lr_schedule, backbone=backbone, network_head='',
                                        checkpoint_after_iter=checkpoint_after_iter,
                                        checkpoint_load_iter=checkpoint_load_iter, temp_path=temp_path,
                                        device=device, threshold=0.0, scale=1.0)
        self.seed = seed
        self.lr_end = lr_end
        self.nr_evaluations = nr_evaluations
        
        #self.evaluation_frequency = evaluation_frequency
        if env is not None:
            
            self.stable_bl_agent = self._construct_agent(env=env, config_filename=config_filename)

        
    def download(self, path=None,
                 mode="checkpoint",
                 url=OPENDR_SERVER_URL + "control/multi_object_search/",
                 robot_name: str = None):

        
        assert mode in ('checkpoint', 'ig_requirements')
        
        if path is None:
            path = self.temp_path
        
        if mode == 'checkpoint':
            assert robot_name is not None, robot_name
            print(f"-----> Start Download SB3 Checkpoint for {robot_name}")
            filename = f"checkpoints/{robot_name.lower()}.zip"
            file_destination = Path(path) / filename
            
            if not file_destination.exists():
                file_destination.parent.mkdir(parents=True, exist_ok=True)
                url = os.path.join(url, filename)
                urlretrieve(url=url, filename=file_destination)
            return file_destination
        else:
            print("-----> Start downloading iGibson assets")
            download_assets()
            download_demo_data()
            
            file_destinations = []
            for d in ["fetch.urdf","locobot.urdf"]:
                    
                if d == "fetch.urdf":
                    filename = f"{d}"
                    file_destination = Path(path) / filename
                    file_destinations.append(file_destination)
                    if not file_destination.exists():
                        file_destination.parent.mkdir(parents=True, exist_ok=True)
                        url_download = os.path.join(url, filename)
                        
                        urlretrieve(url=url_download, filename=file_destination)
                    
                    fetch_src = Path(path) / f"{d}"
                    dst = Path(igibson.assets_path) / "models/fetch/"
                    shutil.copy(fetch_src, dst)
                    
                else:
                    filename = f"{d}"
                    file_destination = Path(path) / filename
                    file_destinations.append(file_destination)
                    if not file_destination.exists():
                        file_destination.parent.mkdir(parents=True, exist_ok=True)
                        url_download = os.path.join(url, filename)
                        
                        urlretrieve(url=url_download, filename=file_destination)
                    
                    locobot_src = Path(path) / f"{d}"
                    dst = Path(igibson.assets_path) / "models/locobot/"
                    shutil.copy(locobot_src, dst)
                    
                    

            return file_destinations
        
        

    def _construct_agent(self, env, config_filename):
        
        self.config = parse_config(config_filename)
        aux_bin_number = self.config.get("num_bins", 12)
        task_obs = env.observation_space['task_obs'].shape[0] - aux_bin_number
        policy_kwargs = dict(features_extractor_class=EgocentricEncoders)
        return PPO_AUX(policy=self.backbone,
            env=env,
            ent_coef=self.config.get("ent_coef", 0.005), 
            batch_size=self.config.get("batch_size", 64), 
            clip_range=self.config.get("clip_range", 0.1),
            gamma=self.config.get("gamma", 0.99),
            n_steps=self.config.get("rollout_buffer_size", 2048), 
            n_epochs=self.config.get("n_epochs", 4),
            learning_rate=self.config.get("learning_rate", 0.0001),
            verbose=0,
            tensorboard_log=self.temp_path, 
            policy_kwargs=policy_kwargs,
            aux_pred_dim=aux_bin_number, 
            proprio_dim=task_obs)

    def fit(self, env=None, logging_path='', silent=False, verbose=True):
        """
        Train the agent on the environment.

        :param env: gym.Env, optional, if specified use this env to train
        :param val_env:  gym.Env, optional, if specified periodically evaluate on this env
        :param logging_path: str, path for logging and checkpointing
        :param silent: bool, disable verbosity
        :param verbose: bool, enable verbosity
        :return:
        """
        if logging_path == '':
            logging_path = self.temp_path

        if env is not None:
            assert env.action_space == self.stable_bl_agent.env.action_space
            assert env.observation_space == self.stable_bl_agent.env.observation_space
            self.stable_bl_agent.env = env

        #rospy.loginfo("Start learning loop")
        save_model_callback = SaveModel(check_freq=self.config.get("rollout_buffer_size", 2048), log_dir=logging_path)

        self.stable_bl_agent.learn(total_timesteps=self.iters,callback=save_model_callback)
                                   

        self.stable_bl_agent.save(os.path.join(logging_path, 'last_model'))

        

    def eval(self, env, name_prefix='', name_scene='',nr_evaluations: int = 75,deterministic_policy: bool = False):
        """
        Evaluate the agent on the specified environment.

        :param env: gym.Env, env to evaluate on
        :param name_prefix: str, name prefix for all logged variables
        :param nr_evaluations: int, number of episodes to evaluate over
        :return:
        """
        if nr_evaluations is None:
            nr_evaluations = self.nr_evaluations
        if isinstance(env, VecEnv):
            assert env.num_envs == 1, "You must pass only one environment when using this function"
            env = env.envs[0]

        if self.stable_bl_agent.logger is None:
            self.stable_bl_agent.set_logger(configure_logger(self.stable_bl_agent.verbose,
                                                             self.stable_bl_agent.tensorboard_log,
                                                             tb_log_name="PPO",
                                                             reset_num_timesteps=False))

        prefix = ''
        episode_rewards, episode_lengths, metrics, name_prefix = evaluation_rollout(
            self.stable_bl_agent,
            env,
            nr_evaluations, name_prefix=prefix,
            name_scene=name_scene,
            deterministic_policy=deterministic_policy,
            verbose=2)
        #env.clear()
        return {"episode_rewards": episode_rewards,
                "episode_lengths": episode_lengths,
                "metrics": metrics,
                "name_prefix": name_prefix}

    def save(self, path):
        """
        Saves the model in the path provided.

        :param path: Path to save directory
        :type path: str
        :return: Whether save succeeded or not
        :rtype: bool
        """
        self.stable_bl_agent.save(path)

    def load(self, path):
        """
        Loads a model from the path provided.

        :param path: Path to saved model
        :type path: str
        :return: Whether load succeeded or not
        :rtype: bool
        """
        

        if path == 'pretrained':
            path = str(self.download(self.temp_path, mode="checkpoint",robot_name=self.config.get("robot", "Fetch")))

        #self.stable_bl_agent.load(path)#,custom_objects={'data':None,'system_info.txt','_stable_baselines3_version','pytorch_variables.pth','policy.pth'})
        self.stable_bl_agent.set_parameters(path,exact_match=False)
        
        

    def infer(self, batch, deterministic: bool = False):
        return self.stable_bl_agent.predict(batch, deterministic=deterministic)

    def reset(self):
        raise NotImplementedError()

    def optimize(self, target_device):
        raise NotImplementedError()
