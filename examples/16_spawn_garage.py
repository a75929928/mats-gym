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
This example shows how run a leaderboard route scenario with certain agent.
"""
import importlib
import os
import sys
from datetime import datetime
import pathlib

import os
import yaml
import inspect

# Choose which policy to use
# Options: 'garage','expert, TODO 'lmdrive', 'ppo', 'transfuser', 
POLICY = 'garage' 
NUM_EGO_VEHICLES = 5

def replace_config_values(config_dict, key_value_dict):
    for key, value in config_dict.items():
        if isinstance(value, str):
            # Replace the %key% to its actual value
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

def load_agents(args, num_agents):
    """
    Load multiple agent instances using importlib.

    :param args: The arguments containing the path to the agent module and its configuration.
    :param num_agents: The number of agent instances to load.
    :return: A list of agent instances.
    """
    # List to store the loaded agent instances
    agents_instances = {}

    # Load each agent instance
    for i in range(num_agents):
        # Insert parent dir to path
        sys.path.insert(0, os.path.dirname(args.agent))

        # Import the agent module
        module_name = os.path.basename(args.agent).split('.')[0]
        module_agent = importlib.import_module(module_name)

        # Get the agent class name from the entry point function
        agent_class_name = getattr(module_agent, 'get_entry_point')()

        # Create an instance of the agent class with the provided configuration
        # Assuming args.agent_config can be copied or is suitable for multiple instances
        agent_instance = getattr(module_agent, agent_class_name)(args.agent_config)

        # Remove the parent directory from sys.path to avoid pollution
        sys.path.pop(0)

        # Add the created agent instance to the list
        agents_instances.update({f"hero_{i}": agent_instance})

    return agents_instances

def get_policy_for_agent(agent: AutonomousAgent):
    def policy(obs):
        control = agent.run_step(input_data=obs, timestamp=GameTime.get_time())
        action = np.array([control.throttle, control.steer, control.brake])
        return action
    return policy

def main(args):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(filename)s - [%(levelname)s] - %(message)s",
    )

    actor_config = ActorConfiguration(
        rolename="hero_0",
        model="vehicle.lincoln.mkz_2017",
        transform=None
    )
    
    # Create individual agent instance to decide seperately
    # agent_instances = {f"hero_{i}": agent_loader(args) for i in range(NUM_EGO_VEHICLES)}
    agent_instances = load_agents(args, NUM_EGO_VEHICLES)
    
    env = mats_gym.parallel_env(
        route_file=args.routes,
        agent_instances=agent_instances, # added 
        actor_configuration=actor_config,
        # Rendering
        no_rendering_mode=True,
        # render_mode="human",
        # render_config=renderers.camera_top(agent="hero_0"), # whether to render with pygame
        # debug_mode=True, # whether to draw waypoints

        num_agents = NUM_EGO_VEHICLES,
        sensor_specs={agent_id: agent_ins.sensors() for agent_id, agent_ins in agent_instances.items()},  # sensor specs for each agent
        timestep=0.05, # fixed_delta_seconds: interval of function step means in carla
    )

    client = carla.Client("localhost", 2000)
    client.set_timeout(60.0)

    t = 0
    for _ in range(100):
        obs, info = env.reset(options={"client": client})
        
        for agent_ins in agent_instances.values():
            # Find whether route_index is in setup signature
            setup_signature = inspect.signature(agent_ins.setup)
            route_index_param = 'route_index' in setup_signature.parameters 
            if route_index_param:
                agent_ins.setup(path_to_conf_file=args.agent_config, route_index=None)
            else:
                agent_ins.setup(path_to_conf_file=args.agent_config)

        policy = {agent_id : get_policy_for_agent(agent_instance) for agent_id, agent_instance in agent_instances.items()}
        done = False
        while not done:
            # Get action with certain policy
            actions = {agent: policy[agent](o) for agent, o in obs.items()}
            obs, reward, done, truncated, info = env.step(actions)
            done = all(done.values())
            env.render()
            t += 1
            print("EVENTS: ", info["hero_0"]["events"])

    env.close()


import argparse
from argparse import RawTextHelpFormatter
if __name__ == "__main__":
    config = config_loader(POLICY)

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