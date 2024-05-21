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
import random
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

# add navigation to produce waypoints between two points
from mats_gym.navigation.global_route_planner import GlobalRoutePlanner

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
        self.spawn_points = world.get_map().get_spawn_points()
        self.grp = GlobalRoutePlanner(carla_map=world.get_map(), sampling_resolution=25.0)

        ego_vehicles = []
        self.route = {}
        for agent_id in config.agents:
            ego_vehicle = None
            while ego_vehicle is None:
                self.route.update({agent_id: self._get_route(world, config, agent_id)})
                ego_vehicle = self._spawn_ego_vehicle(agent_id)
            ego_vehicles.append(ego_vehicle)

        self.timeout = 60
        if debug_mode:
            for agent_id in config.agents:
                self._draw_waypoints(world, self.route[agent_id], vertical_shift=0.1, size=0.1, persistency=self.timeout, downsample=5)

        super(RouteParallel, self).__init__(
            config.name, ego_vehicles, config, world, debug_mode > 1, False, criteria_enable
        )

    def _get_route(self, world, config, agent_id):
        """
        Gets the route from the configuration, interpolating it to the desired density,
        saving it to the CarlaDataProvider and sending it to the agent

        Parameters:
        - world: CARLA world
        - config: Scenario configuration (RouteConfiguration)
        - debug_mode: boolean to decide whether or not the route poitns are printed
        """

        sp = self.spawn_points
        num_list = list(range(0, len(sp))) # get the list of indices
        start_index, end_index = random.sample(num_list, 2)
        waypoints = self.grp.trace_route(sp[start_index].location, sp[end_index].location)
        locations = [waypoint[0].transform.location for waypoint in waypoints]
        gps_route, route = interpolate_trajectory(locations) # need carla.Location
        config.agents[agent_id].set_global_plan(gps_route, route)

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
        world.set_weather(self.config.weather)

    def _create_behavior(self):
        """
        Define the behavior tree for ego_vehicles
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

    def _draw_waypoints(self, world, waypoints, vertical_shift, size, persistency=-1, downsample=1):
        """
        Draw a list of waypoints at a certain height given in vertical_shift.
        """
        for i, w in enumerate(waypoints):
            if i % downsample != 0:
                continue

            wp = w[0].location + carla.Location(z=vertical_shift)

            if w[1] == RoadOption.LEFT:  # Yellow
                color = carla.Color(128, 128, 0)
            elif w[1] == RoadOption.RIGHT:  # Cyan
                color = carla.Color(0, 128, 128)
            elif w[1] == RoadOption.CHANGELANELEFT:  # Orange
                color = carla.Color(128, 32, 0)
            elif w[1] == RoadOption.CHANGELANERIGHT:  # Dark Cyan
                color = carla.Color(0, 32, 128)
            elif w[1] == RoadOption.STRAIGHT:  # Gray
                color = carla.Color(64, 64, 64)
            else:  # LANEFOLLOW
                color = carla.Color(0, 128, 0)  # Green

            world.debug.draw_point(wp, size=0.1, color=color, life_time=persistency)

        world.debug.draw_point(waypoints[0][0].location + carla.Location(z=vertical_shift), size=2*size,
                               color=carla.Color(0, 0, 128), life_time=persistency)
        world.debug.draw_point(waypoints[-1][0].location + carla.Location(z=vertical_shift), size=2*size,
                               color=carla.Color(128, 128, 128), life_time=persistency)
    '''
    TODO
    Add debug functions
    '''
    # def draw_waypoints()...