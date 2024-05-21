import os
from xml.etree import ElementTree

import carla
import gymnasium
import numpy as np
import srunner
from pettingzoo.utils.env import AgentID, ObsType
from srunner.scenarioconfigs.route_scenario_configuration import RouteScenarioConfiguration
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration, \
    ActorConfigurationData
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
# from srunner.scenarios.route_scenario import RouteScenario
from mats_gym.scenarios.route_parallel import RouteParallel # add parallel_scenario

import mats_gym
from mats_gym.envs.scenario_env_wrapper import BaseScenarioEnvWrapper
from mats_gym.envs.adapters import ParallelEnv

from mats_gym.scenarios.actor_configuration import ActorConfiguration

from typing import Dict, Union, Tuple

'''
    This Env intends to implement parallel agent with differentr scenarios
'''
class CommunicationEnv(ParallelEnv):

    def __init__(
            self,
            actor_configuration: ActorConfiguration,
            agent_instances: None, # added to import policy
            num_agents: int = 1, # control num of ego agents
            reset_progress_threshold: float = None,
            debug_mode: bool = False,
            no_rendering_mode: bool = False,
            **kwargs,
    ):
        super().__init__(actor_configuration, agent_instances, num_agents, reset_progress_threshold,
                         debug_mode, no_rendering_mode, **kwargs)
    def step(self, action: dict) -> tuple[dict[AgentID, ObsType], dict[AgentID, float], dict, dict]:
        obs, reward, term, trun, info = self.env.step(action)
        for agent in self.env.agents:
            progress = self._get_progress(info=info, agent=agent)
            self._progress[agent] = progress
        obs = self._add_progress(obs)
        return obs, reward, term, trun, info

    def _add_progress(self, obs: dict) -> dict:
        for agent in self.agents:
            obs[agent]["progress"] = np.array(self._progress.get(agent, 0), dtype=np.float32)
        return obs

    def reset(self, seed: Union[int, None] = None, options: Union[Dict, None] = None) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        options = options or {}
        if options.get("reload", True):
            if "route" not in options:
                config = self._configs
                self._current_route = 0
            else:
                self._current_route = int(options["route"])
                config = next(
                    filter(
                        lambda c: c.name.replace("RouteScenario_", "") == str(self._current_route),
                        self._configs
                    ))
            options["scenario_config"] = config
            obs, self._info = self.env.reset(seed=seed, options=options)
        else:
            # Maybe add bias manually according to reset?
            progress = options.get("progress", self._progress)
            route = self.current_scenario.route
            start_idx = int(len(route) * progress)
            start, _ = route[start_idx]
            map = CarlaDataProvider.get_map()
            wp = map.get_waypoint(start.location, project_to_road=True)
            for role_name in self._ego_role_name:
                self.actors[role_name].set_transform(wp.transform)
                self._progress = {agent: options.get("progress", progress) for agent in self.agents}
                obs = {role_name: self.env.observe(role_name)}

            CarlaDataProvider.get_world().tick()

        obs = self._add_progress(obs)

        return obs, self._info

    def _get_progress(self, info: dict, agent: str):
        events = info.get(agent, {}).get("events", [])
        completion = list(filter(lambda e: e["event"] == "ROUTE_COMPLETION", events))
        if completion:
            return float(completion[0]["route_completed"]) / 100
        else:
            return self._progress[agent]
