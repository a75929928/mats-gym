#!/usr/bin/env python

# Copyright (c) 2019-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides Challenge routes as standalone scenarios
"""

from __future__ import print_function

import glob
import os
import sys
import importlib
import inspect
import traceback
import py_trees

from numpy import random
import carla

from agents.navigation.local_planner import RoadOption

from srunner.scenarioconfigs.scenario_configuration import ActorConfigurationData
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ScenarioTriggerer, Idle
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import WaitForBlackboardVariable
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (CollisionTest,
                                                                     InRouteTest,
                                                                     RouteCompletionTest,
                                                                     OutsideRouteLanesTest,
                                                                     RunningRedLightTest,
                                                                     RunningStopTest,
                                                                     ActorBlockedTest,
                                                                     MinimumSpeedRouteTest)

from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenarios.background_activity import BackgroundBehavior
from srunner.scenariomanager.weather_sim import RouteWeatherBehavior
from srunner.scenariomanager.lights_sim import RouteLightsBehavior
from srunner.scenariomanager.timer import RouteTimeoutBehavior

from srunner.tools.route_parser import RouteParser, DIST_THRESHOLD
from srunner.tools.route_manipulation import interpolate_trajectory

import re
SECONDS_GIVEN_PER_METERS = 0.4


class RouteParallel(BasicScenario):

    """
    Implementation of a RouteScenario, i.e. a scenario that consists of driving along a pre-defined route,
    along which several smaller scenarios are triggered
    """

    def __init__(self, world, config, debug_mode=False, criteria_enable=True, timeout=300):
        """
        Setup all relevant parameters and create scenarios along route
        """

        self.config = config
        self.route = {}
        ego_vehicles = []
        for agent_id, agent_instance in config.agent.items():
            self.route.update({agent_id: self._get_route(config, agent_id)})
            ego_vehicles.append(self._spawn_ego_vehicle(agent_id))

        super(RouteParallel, self).__init__(
            config.name, ego_vehicles, config, world, debug_mode > 1, False, criteria_enable
        )

    def _get_route(self, config, agent_id):
        """
        Gets the route from the configuration, interpolating it to the desired density,
        saving it to the CarlaDataProvider and sending it to the agent

        Parameters:
        - world: CARLA world
        - config: Scenario configuration (RouteConfiguration)
        - debug_mode: boolean to decide whether or not the route poitns are printed
        """
        # prepare route's trajectory (interpolate and add the GPS route)
        match = re.search(r"hero_(\d+)", agent_id)
        if match:
            num_agent_id = str(match.group(1))
        gps_route, route = interpolate_trajectory(config.keypoints[num_agent_id])
        if config.agent[agent_id] is not None:
            config.agent[agent_id].set_global_plan(gps_route, route)

        return route

    def _spawn_ego_vehicle(self, agent_id):
        """Spawn the ego vehicle at the first waypoint of the route"""
        elevate_transform = self.route[agent_id][0][0]
        elevate_transform.location.z += 0.5

        ego_vehicle = CarlaDataProvider.request_new_actor('vehicle.lincoln.mkz_2017',
                                                          elevate_transform,
                                                          rolename=agent_id)

        return ego_vehicle

    def _initialize_environment(self, world):
        """
        Set the weather
        """
        # Set the appropriate weather conditions
        world.set_weather(self.config.weather[0][1])

    def _create_behavior(self):
        """
        """
        behavior = py_trees.composites.Parallel(name="Route Behavior",
                                                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)

        # behavior.add_child(Idle()) # Just keep going
        
        # Add the Background Activity
        for ego_vehicle  in self.ego_vehicles:
            _role_name = ego_vehicle.attributes["role_name"]
            behavior.add_child(BackgroundBehavior(ego_vehicle, self.route[_role_name], name="BackgroundActivity"))
        return behavior
    
    # TODO add various Tests
    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        for ego_vehicle in self.ego_vehicles:
            collision_criterion = CollisionTest(ego_vehicle)
            criteria.append(collision_criterion)

        return criteria
    
    '''
    TODO
    Add debug functions
    '''
    # def draw_waypoints()...