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

import numpy as np
import os
import shutil
import torch
import unittest
import random
from opendr.control.multi_object_search import MultiObjectEnv
from opendr.control.multi_object_search import ExplorationRLLearner 
from pathlib import Path

device = os.getenv('TEST_DEVICE') if os.getenv('TEST_DEVICE') else 'cpu'
TEST_ITERS = 1
TEMP_SAVE_DIR = Path(__file__).parent / "multi_object_search_tmp"


EVAL_CONFIG_FILE = str(Path(__file__).parent / 'test_config.yaml')


def get_first_weight(learner):
    return list(learner.stable_bl_agent.get_parameters()['policy'].values())[0].clone()

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class MultiObjectSearchTest(unittest.TestCase):
    learner = None

    @classmethod
    def setUpClass(cls):
        print("\n\n**********************************\nTEST Multi-Object-Search\n"
              "**********************************")
        set_seed(0)
        cls.env = MultiObjectEnv(config_file=EVAL_CONFIG_FILE, scene_id="Rs_int")
        cls.learner = ExplorationRLLearner(cls.env, device=device, iters=TEST_ITERS, temp_path=str(TEMP_SAVE_DIR),config_filename=EVAL_CONFIG_FILE)

    @classmethod
    def tearDownClass(cls):
        del cls.learner


    def test_load_prerequisites(cls):
        
        prereqs_folders = cls.learner.download(mode='ig_requirements')
        for file_dest in prereqs_folders:
            cls.assertTrue(Path(file_dest).exists, f"file could not be downloaded {file_dest}")
        
        # Remove temporary files
        try:
            pass
            #shutil.rmtree(TEMP_SAVE_DIR)
        except OSError as e:
            print(f"Exception when trying to remove temp directory: {e.strerror}")

    def test_ckpt_download(cls):
        
        ckpt_folder = cls.learner.download(str(TEMP_SAVE_DIR), robot_name=cls.env.config.get("robot","Locobot"))
        cls.assertTrue(Path(ckpt_folder).exists, f"Checkpoint file could not be downloaded {ckpt_folder}")

        # Remove temporary files
        try:
            pass
            #shutil.rmtree(TEMP_SAVE_DIR)
        except OSError as e:
            print(f"Exception when trying to remove temp directory: {e.strerror}")

    def test_fit(cls):
      
        weights_before_fit = get_first_weight(cls.learner)
        cls.learner.fit()
        cls.assertFalse(torch.equal(weights_before_fit, get_first_weight(cls.learner)),
                         msg="Fit method did not alter model weights")

    def test_eval(cls):
        
        nr_evaluations = 2
        metrics = cls.learner.eval(cls.env, nr_evaluations=nr_evaluations,name_scene="Rs_int")
        cls.assertTrue(len(metrics['episode_rewards']) == nr_evaluations, "Episode rewards have incorrect length.")
        cls.assertTrue((np.array(metrics['episode_rewards']) <= 0.0).all(), "Test reward not below 0.")
        cls.assertTrue((np.array(metrics['episode_lengths']) >= 0.0).all(), "Test episode lengths is negative")

    def test_eval_pretrained(cls):
        
        nr_evaluations = 7
        cls.learner.load('pretrained')
        metrics = cls.learner.eval(cls.env, nr_evaluations=nr_evaluations,name_scene="Rs_int")
        success = metrics['metrics']['success']
        print("SUCCESS-rate",success)
        cls.assertTrue(success > 0.60, f"Success rate of pretrained model is only {success}")

    

    def test_infer(cls):
        obs = cls.env.observation_space.sample()
        actions = cls.learner.infer(obs)[0]
        actions = np.array(actions)
        cls.assertTrue(actions.shape == cls.env.action_space.shape)
        cls.assertTrue((actions >= -1).all(), "Actions below -1")
        cls.assertTrue((actions <= 1).all(), "Actions above 1")

    def test_save_load(cls):
        weights_before_saving = get_first_weight(cls.learner)
        cls.learner.save(os.path.join(TEMP_SAVE_DIR, 'initial_weights'))
        
        cls.learner.load('pretrained')
        cls.assertFalse(torch.equal(weights_before_saving, get_first_weight(cls.learner)),
                         msg="Load() did not alter model weights")

        cls.learner.load(os.path.join(TEMP_SAVE_DIR, 'initial_weights'))
        cls.assertTrue(torch.equal(weights_before_saving, get_first_weight(cls.learner)),
                        msg="Load did not restore initial weights correctly")

        # Remove temporary files
        try:
            shutil.rmtree(TEMP_SAVE_DIR)
        except OSError as e:
            print(f"Exception when trying to remove temp directory: {e.strerror}")


if __name__ == "__main__":
    unittest.main()
