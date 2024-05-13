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
import inspect
import importlib
import os
import sys
from datetime import datetime
import pathlib

CARLA_ROOT = os.getenv('1', 'D:\Code\carla\CARLA_0.9.15')
WORK_DIR = os.getenv('2', 'D:\Code\.LLM\LMDrive')
MATS_ROOT = os.getenv('3', 'D:\Code\carla\mats-gym')
CARLA_SERVER = f'{CARLA_ROOT}\\CarlaUE4.exe'

PYTHONPATH = os.environ.get('PYTHONPATH', '')
PYTHONPATH += f';{CARLA_ROOT}\\PythonAPI'
PYTHONPATH += f';{CARLA_ROOT}\\PythonAPI\\carla'
os.environ['PYTHONPATH'] = PYTHONPATH

SCENARIOS = f'{WORK_DIR}\\leaderboard\\data\\official\\all_towns_traffic_scenarios_public.json'
ROUTES_SUBSET=0
ROUTES = f'{MATS_ROOT}/scenarios/routes/training.xml'
# ROUTES = f'{WORK_DIR}\\langauto\\benchmark_long.xml'
os.environ["ROUTES"] = ROUTES # add for lmdrive
REPETITIONS = '1'
CHALLENGE_TRACK_CODENAME = 'SENSORS'
CHECKPOINT_ENDPOINT = f'{WORK_DIR}\\results\\transfuser_plus_plus_longest6.json'
TEAM_AGENT = f'{WORK_DIR}\\leaderboard\\team_code\\lmdriver_agent.py'
AGENT_CONFIG = f'{WORK_DIR}\\leaderboard\\team_code\\lmdriver_config.py'
DEBUG_CHALLENGE = '0'
RESUME = '1'
DATAGEN = '0'
SAVE_PATH = f'{WORK_DIR}\\results'

UNCERTAINTY_THRESHOLD = '0.33'
BENCHMARK = 'longest6'

def agent_loader(args):
    """
    Used in Leaderboard 
    To load agent with importlib
    """
    # Load agent
    module_name = os.path.basename(args.agent).split('.')[0]
    parent_dir = os.path.dirname(args.agent)
    sys.path.insert(0, parent_dir) # insert parent dir to path
    sys.path.insert(0, os.path.dirname(parent_dir)) # FOR LMDRIVE: import leaderboard folder
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
    agent = agent_loader(args)
    # agent = AutopilotAgent(role_name=actor_config.rolename, carla_host="localhost", carla_port=2000)
    env = mats_gym.route_scenario_env(
        route_file=args.routes,
        agent_instance=agent, # added 
        actor_configuration=actor_config,
        render_mode="human",
        render_config=renderers.camera_pov(agent="hero"),
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
    parser.add_argument('--timeout', default="60.0",
                        help='Set the CARLA client timeout value in seconds')

    # simulation setup
    parser.add_argument('--routes',
                        help='Name of the route to be executed. Point to the route_xml_file to be executed.',
                        default=ROUTES)
                        # required=True)
    parser.add_argument('--scenarios',
                        help='Name of the scenario annotation file to be mixed with the route.',
                        default=SCENARIOS)
                        # required=True)
    parser.add_argument('--repetitions',
                        type=int,
                        default=1,
                        help='Number of repetitions per route.')

    # agent-related options
    parser.add_argument("-a", "--agent", type=str, help="Path to Agent's py file to evaluate", 
                        default=TEAM_AGENT)
                        # required=True)
    parser.add_argument("--agent-config", type=str, help="Path to Agent's configuration file", 
                        default=AGENT_CONFIG)
                        # default="")

    parser.add_argument("--track", type=str, default='SENSORS', help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=bool, default=False, help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str,
                        default='./simulation_results.json',
                        help="Path to checkpoint used for saving statistics and resuming")

    arguments = parser.parse_args()

    main(arguments)