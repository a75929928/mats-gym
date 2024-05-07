import logging
import carla
import gymnasium
import numpy as np
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration, ActorConfigurationData
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarios.cut_in import CutIn
from srunner.scenarios.freeride import FreeRide
from srunner.tools.scenario_parser import ScenarioConfigurationParser

from mats_gym.envs import renderers
import mats_gym

import os

import ray
import supersuit as ss
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from torch import nn

from pettingzoo.butterfly import pistonball_v6


class CNNModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, [8, 8], stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, [4, 4], stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, [3, 3], stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(3136, 512)),
            nn.ReLU(),
        )
        self.policy_fn = nn.Linear(512, num_outputs)
        self.value_fn = nn.Linear(512, 1)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.model(input_dict["obs"].permute(0, 3, 1, 2))
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()

"""
This example shows how to use the BaseScenarioEnv class directly by passing a scenario factory function.
"""

NUM_EPISODES = 3
SENSOR_SPECS = [
    {"id": "rgb-center", "type": "sensor.camera.rgb", "x": 0.7, "y": 0.0, "z": 1.60},
    {
        "id": "lidar",
        "type": "sensor.lidar.ray_cast",
        "range": 100,
        "channels": 32,
        "x": 0.7,
        "y": -0.4,
        "z": 1.60,
        "yaw": -45.0,
    },
    {"id": "gps", "type": "sensor.other.gnss", "x": 0.7, "y": -0.4, "z": 1.60},
]

def scenario_fn(client: carla.Client, config: ScenarioConfiguration):
    """
    This function is called by the environment to create the scenario.
    :param client: The carla client.
    :param config: The scenario configuration.
    :return: A base scenario instance.
    """
    world = client.get_world()
    ego_vehicles = []
    for vehicle in config.ego_vehicles:
        actor = CarlaDataProvider.request_new_actor(
            model=vehicle.model,
            spawn_point=vehicle.transform,
            rolename=vehicle.rolename,
            color=vehicle.color,
            actor_category=vehicle.category,
        )
        ego_vehicles.append(actor)
    scenario = FreeRide(
        world=world,
        ego_vehicles=ego_vehicles,
        config=config,
        debug_mode=False,
        timeout=10,
    )
    return scenario

def env_creator(args):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(filename)s - [%(levelname)s] - %(message)s",
    )

    # The base environment can be used directly by providing a scenario factory function. This function takes two
    # arguments (client and config) and returns a scenario instance. The scenario instance must be a subclass of BaseScenario.
    # Furthermore, the function is responsible creating the ego vehicles. For concrete examples, see
    # the adapter wrappers in adex_gym.envs.adapters (for Scenic and ScenarioRunner scenarios).

    # By default, the observation space for each agent is a dictionary with one key "state" which holds a
    # vector containing the position and velocity of the agent. If sensor specs are provided, the observation space
    # will contain an entry for each sensor with its corresponding id. Sensor specs are of the same format as in
    # the autonomous driving challenge (https://leaderboard.carla.org/get_started/#33-override-the-sensors-method).

    # If you wish to run the base environment, you need to set up the scenario configuration manually.
    config = ScenarioConfiguration()
    config.name = "cut_in"
    config.town = "Town04"
    ego_vehicle = ActorConfigurationData(
        rolename="ego",
        model="vehicle.lincoln.mkz2017",
        transform=carla.Transform(
            carla.Location(x=284.4, y=16.4, z=2.5),
            carla.Rotation(yaw=180)
        )
    )
    config.ego_vehicles = [ego_vehicle]
    config.trigger_points = [ego_vehicle.transform]
    config.other_actors = [ActorConfigurationData(
        model="vehicle.tesla.model3",
        rolename="scenario",
        transform=carla.Transform(
            carla.Location(x=324.2, y=20.7, z=2.5 - 105), # Original scenario let the car fall from the sky (for some reason)
            carla.Rotation(yaw=180)
        )
    )]
    config.weather = carla.WeatherParameters(sun_altitude_angle=75)

    # This config is then passed to the base environment.
    env = mats_gym.raw_env(
        config=config,  # The scenario configuration.
        scenario_fn=scenario_fn,  # A function that takes a carla client and a scenario config to instantiate a scenario.
        render_mode="human",  # The render mode. Can be "human", "rgb_array", "rgb_array_list".
        render_config=renderers.camera_pov(
            agent="scenario",  # The agent to follow with the camera.
        ),  # See adex_gym.envs.renderers for more render configs.
        sensor_specs={"ego": SENSOR_SPECS},  # sensor specs for each agent
    )

    # obs, info = env.reset(options={"client": carla.Client("localhost", 2000)})
    
    env.observation_spaces = {agent: env.observation_space(agent) for agent in env.agents}
    env.action_spaces = {agent: env.action_space(agent) for agent in env.agents}
    
    return env

    # SuperSuit revise observation
    # env = ss.color_reduction_v0(env, mode="B")
    # env = ss.dtype_v0(env, "float32")
    # env = ss.resize_v1(env, x_size=84, y_size=84)
    # env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    # env = ss.frame_stack_v1(env, 3)


if __name__ == "__main__":
    ray.init()

    env_name = "base_env"
    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)

    config = (
        PPOConfig()
        .environment(env=env_name, clip_actions=True)
        .rollouts(num_rollout_workers=1, rollout_fragment_length=128)
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.99,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "1")))
    )
    
    env = env_creator(config)

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 5000000 if not os.environ.get("CI") else 50000},
        checkpoint_freq=10,
        local_dir="~/ray_results/" + env_name,
        config=config.to_dict(),
    )