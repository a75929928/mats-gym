from __future__ import annotations

import logging
import os

import carla
import numpy as np
from srunner.autoagents.autonomous_agent import AutonomousAgent
from srunner.scenarioconfigs.route_scenario_configuration import (
    RouteScenarioConfiguration,
)
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration
from srunner.scenariomanager.timer import GameTime
from srunner.scenarios.route_scenario import RouteScenario
from srunner.tools.route_parser import RouteParser

import mats_gym
from mats_gym.envs import renderers
from mats_gym.scenarios.actor_configuration import ActorConfiguration
from examples.example_agents import AutopilotAgent

import sys
sys.path.append("D:\\Code\\.SOTA\\carla_garage\\team_code")
from sensor_agent import SensorAgent
# Set python path
CARLA_ROOT = os.getenv('1', 'D:\\Code\\carla\\CARLA_0.9.15')
WORK_DIR = os.getenv('2', 'D:\Code\carla_garage')
CARLA_SERVER = f'{CARLA_ROOT}\\CarlaUE4.exe'
PYTHONPATH = os.environ.get('PYTHONPATH', '')

PYTHONPATH += f';{CARLA_ROOT}\\PythonAPI'
PYTHONPATH += f';{CARLA_ROOT}\\PythonAPI\\carla'
PYTHONPATH += f';{CARLA_ROOT}\\PythonAPI\\carla\\dist\\carla-0.9.15-py3.7-win-amd64.egg'

# Use leaderboard 2.0 and corresponding scenario_runner
# PYTHONPATH += f';{WORK_DIR}\\scenario_runner'
# PYTHONPATH += f';{WORK_DIR}\\leaderboard'

os.environ['PYTHONPATH'] = PYTHONPATH

"""
This example shows how run a leaderboard route scenario with a custom agent.
"""


def get_policy_for_agent(agent: AutonomousAgent):
    def policy(obs):
        control = agent.run_step(input_data=obs, timestamp=GameTime.get_time())
        action = np.array([control.throttle, control.steer, control.brake])
        return action

    return policy


def main():
    # Set environment variable for the scenario runner root. It can be found in the virtual environment.

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(filename)s - [%(levelname)s] - %(message)s",
    )

    actor_config = ActorConfiguration(
        rolename="hero",
        model="vehicle.lincoln.mkz2017",
        transform=None
    )
    agent = AutopilotAgent(role_name=actor_config.rolename, carla_host="localhost", carla_port=2000)
    env = mats_gym.route_scenario_env(
        route_file="D:\\Code\\carla\\mats-gym\\tests\\leaderboard_test\\dev_test.xml",
        actor_configuration=actor_config,
        render_mode="human",
        render_config=renderers.camera_pov(agent="hero")
    )

    client = carla.Client("localhost", 2000)
    client.set_timeout(120.0)

    t = 0
    for _ in range(100):
        obs, info = env.reset(options={"client": client})
        agent.setup(path_to_conf_file="", route=env.current_actor_config.route)
        policy = get_policy_for_agent(agent)
        done = False
        while not done:
            # Use agent to get control for the current step
            actions = {agent: policy(o) for agent, o in obs.items()}
            obs, reward, done, truncated, info = env.step(actions)
            done = done["hero"]
            env.render()
            t += 1
            print("EVENTS: ", info["hero"]["events"])

    env.close()

    
if __name__ == "__main__":
    main()
