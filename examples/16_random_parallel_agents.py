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

import time

os.environ['CURL_CA_BUNDLE'] = ''

# Choose which policy to use
# Options: 'garage','expert, TODO 'lmdrive', 'ppo', 'transfuser', 
POLICY = 'garage' 
NUM_EGO_VEHICLES = 1

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

def yaml_loader(policy):
    yaml_path = os.path.join(os.path.dirname(__file__), 'configs', policy + '.yml')
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    key_value_dict = {k: v for k, v in config.items() if not k.startswith('%')}
    replace_config_values(config, key_value_dict)
    for key, value in config.items():
        if isinstance(value, str):
            config[key] = os.path.expandvars(value)
    return config

def load_policy(args, num_agents):
    """
    Load multiple agent instances using importlib
    """
    # TODO add several policy parallelly
    # Load each agent instance
    agents_instances = {}
    sys.path.insert(0, os.path.dirname(args.agent))
    for i in range(num_agents):
        module_name = os.path.basename(args.agent).split('.')[0]
        module_agent = importlib.import_module(module_name)

        agent_class_name = getattr(module_agent, 'get_entry_point')()
        agent_instance = getattr(module_agent, agent_class_name)(args.agent_config)
        agents_instances.update({f"hero_{i}": agent_instance})
    sys.path.pop(0)
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
    
    policy_instances = load_policy(args, NUM_EGO_VEHICLES)
    env = mats_gym.parallel_env(
        # route_file=args.routes,
        agent_instances=policy_instances, # added 
        actor_configuration=actor_config,
        # Rendering
        no_rendering_mode=True,
        # render_mode="human",
        # render_config=renderers.camera_pov(agent="hero_0"), # whether to render with pygame
        # debug_mode=True, # whether to draw waypoints

        num_agents = NUM_EGO_VEHICLES,
        sensor_specs={agent_id: agent_ins.sensors() for agent_id, agent_ins in policy_instances.items()},  # sensor specs for each agent
        timestep=0.05, # fixed_delta_seconds: interval of function step means in carla
    )

    client = carla.Client("localhost", 2000)
    client.set_timeout(60.0)

    for _ in range(100):
        obs, info = env.reset(options={"client": client})
        
        for agent_ins in policy_instances.values():
            # Find whether route_index is in setup signature
            setup_signature = inspect.signature(agent_ins.setup)
            route_index_param = 'route_index' in setup_signature.parameters 
            if route_index_param:
                agent_ins.setup(path_to_conf_file=args.agent_config, route_index=None)
            else:
                agent_ins.setup(path_to_conf_file=args.agent_config)

        policy = {agent_id : get_policy_for_agent(agent_instance) for agent_id, agent_instance in policy_instances.items()}
        terminated = False
        while not terminated:
            actions = {agent: policy[agent](o) for agent, o in obs.items()}
            obs, reward, terminated, truncated, info = env.step(actions)
            env.render()

            print("EVENTS: ")
            flag = False # detect whether agent collided
            # flag = True # for debug
            for agent in obs:
                events = info[agent]['events']
                if len(events) > 0: print(agent,  events)
                for event in events:
                    if 'COLLISION' in event: flag = True
            terminated = all(terminated.values()) or flag
            
    env.close()


import argparse
from argparse import RawTextHelpFormatter
import subprocess
if __name__ == "__main__":
    config = yaml_loader(POLICY)

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
                        default="60.0",
                        help='Set the CARLA client timeout value in seconds')

    # simulation setup
    parser.add_argument('--routes',
                        help='Name of the route to be executed. Point to the route_xml_file to be executed.',
                        default=config["ROUTES"])
    parser.add_argument('--scenarios',
                        help='Name of the scenario annotation file to be mixed with the route.',
                        default=config["SCENARIOS"])
    parser.add_argument('--repetitions',
                        type=int,
                        default=1,
                        help='Number of repetitions per route.')

    # agent-related options
    parser.add_argument("-a", "--agent", type=str, help="Path to Agent's py file to evaluate", 
                        default=config["TEAM_AGENT"])
    parser.add_argument("--agent-config", type=str, help="Path to Agent's configuration file", 
                        default=config["AGENT_CONFIG"])

    parser.add_argument("--track", type=str, 
                        default=config["CHALLENGE_TRACK_CODENAME"], 
                        help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=bool, 
                        default=(config["RESUME"]==1), 
                        help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str,
                        default=config["CHECKPOINT_ENDPOINT"],
                        help="Path to checkpoint used for saving statistics and resuming")

    # add paralisim via launch_carla
    parser.add_argument('--cuda_visible_devices', type=str, default='0')
    parser.add_argument('--num-workers', type=int, default=1)

    arguments = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, "scripts", "launch_carla.sh")
    p=subprocess.Popen([script_path, str(arguments.cuda_visible_devices), str(arguments.num_workers), str(int(arguments.port)+11*int(arguments.cuda_visible_devices))])
    
    main(arguments)