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

"""
This example shows how run a leaderboard route scenario with a custom agent.
"""
import importlib
import os
import sys
from datetime import datetime
import pathlib

import os
import yaml
import inspect


# 定义替换函数
def replace_config_values(config_dict, key_value_dict):
    for key, value in config_dict.items():
        if isinstance(value, str):
            # Replace the %key% to its act
            for k, v in key_value_dict.items():
                value = value.replace(f'%{k}%', v)
            config_dict[key] = value
        elif isinstance(value, dict):
            # Consider dictionary, use recursion
            replace_config_values(value, key_value_dict)
def config_loader(policy):
    yaml_path = os.path.join(os.path.dirname(__file__), 'configs', policy + '.yml')
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    # Change %key% in config string to the actual path
    key_value_dict = {k: v for k, v in config.items() if not k.startswith('%')}
    replace_config_values(config, key_value_dict)

    for key, value in config.items():
        if isinstance(value, str):
            config[key] = os.path.expandvars(value)

    return config
def agent_loader(args):
    """
    Used in Leaderboard 
    Load agent with importlib
    """
    # Load agent
    module_name = os.path.basename(args.agent).split('.')[0]
    sys.path.insert(0, os.path.dirname(args.agent)) # insert parent dir to path
    module_agent = importlib.import_module(module_name)
    agent_class_name = getattr(module_agent, 'get_entry_point')()
    
    now = datetime.now()
    route_string = pathlib.Path(os.environ.get('ROUTES', '')).stem + '_'
    # route_string += f'route{config.index}'
    route_date_string = route_string + '_' + '_'.join(
        map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
    
    agent_instance = getattr(module_agent, agent_class_name)(args.agent_config)
    # agent_instance = getattr(module_agent, agent_class_name)(args.agent_config, route_date_string)
    return agent_instance

def get_policy_for_agent(agent: AutonomousAgent):
    def policy(obs):
        control = agent.run_step(input_data=obs, timestamp=GameTime.get_time())
        action = np.array([control.throttle, control.steer, control.brake])
        return action
    return policy

def main(args):
    # Set environment variable for the scenario runner root. It can be found in the virtual environment.

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(filename)s - [%(levelname)s] - %(message)s",
    )

    actor_config = ActorConfiguration(
        rolename="hero",
        model="vehicle.lincoln.mkz_2017",
        transform=None
    )
    # agent = agent_loader(args)
    agent = AutopilotAgent(role_name=actor_config.rolename, carla_host="localhost", carla_port=2000)
    env = mats_gym.route_scenario_env(
        route_file=args.routes,
        agent_instance=agent, # added 
        actor_configuration=actor_config,
        render_mode="human",
        render_config=renderers.camera_top(agent="hero"),
        # render_config=renderers.camera_pov(agent="hero"),
        sensor_specs={"hero": agent.sensors()},  # sensor specs for each agent
    )

    client = carla.Client("localhost", 2000)
    client.set_timeout(120.0)

    t = 0
    for _ in range(100):
        obs, info = env.reset(options={"client": client})

        setup_signature = inspect.signature(agent.setup)
        # find whether route_index is in setup signature
        route_index_param = 'route_index' in setup_signature.parameters 
        if route_index_param:
            agent.setup(path_to_conf_file=args.agent_config, route_index=None)
        else:
            agent.setup(path_to_conf_file=args.agent_config)

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


import argparse
from argparse import RawTextHelpFormatter
if __name__ == "__main__":
    
    policy = 'garage' # options: 'garage', 'lmdrive', 'ppo'
    config = config_loader(policy)

    description = "Multi-Agent env with carla garage\n"
    # general parameters
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default='2000', help='TCP port to listen to (default: 2000)')
    parser.add_argument('--trafficManagerPort', default='8000',
                        help='Port to use for the TrafficManager (default: 8000)')
    parser.add_argument('--trafficManagerSeed', default='0',
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--debug', type=int, help='Run with debug output', default=0)
    parser.add_argument('--record', type=str, default='',
                        help='Use CARLA recording feature to create a recording of the scenario')
    parser.add_argument('--timeout', 
                        # default=config["TIMEOUT"],
                        default="60.0",
                        help='Set the CARLA client timeout value in seconds')

    # simulation setup
    parser.add_argument('--routes',
                        help='Name of the route to be executed. Point to the route_xml_file to be executed.',
                        default=config["ROUTES"])
                        # required=True)
    parser.add_argument('--scenarios',
                        help='Name of the scenario annotation file to be mixed with the route.',
                        default=config["SCENARIOS"])
                        # required=True)
    parser.add_argument('--repetitions',
                        type=int,
                        default=config["REPETITIONS"],
                        # default=1,
                        help='Number of repetitions per route.')

    # agent-related options
    parser.add_argument("-a", "--agent", type=str, help="Path to Agent's py file to evaluate", 
                        default=config["TEAM_AGENT"])
                        # required=True)
    parser.add_argument("--agent-config", type=str, help="Path to Agent's configuration file", 
                        default=config["AGENT_CONFIG"])
                        # default="")

    parser.add_argument("--track", type=str, 
                        default=config["CHALLENGE_TRACK_CODENAME"], 
                        # default='SENSORS', 
                        help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=bool, 
                        default=(config["RESUME"]==1), 
                        # default=False, 
                        help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str,
                        default=config["CHECKPOINT_ENDPOINT"],
                        # default='./simulation_results.json',
                        help="Path to checkpoint used for saving statistics and resuming")

    arguments = parser.parse_args()

    main(arguments)