import carla
from agents.navigation.basic_agent import BasicAgent
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from leaderboard.envs.sensor_interface import SensorInterface
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.tools.route_manipulation import interpolate_trajectory

def get_entry_point():
    return "AutopilotAgent"

import re
class AutopilotAgent(AutonomousAgent):
    def __init__(
        self, role_name: str, carla_host="localhost", carla_port=2000, debug=False, opt_dict={}
    ):
        self.track = Track.SENSORS
        #  current global plans to reach a destination
        self._global_plan = None
        self._global_plan_world_coord = None
        # this data structure will contain all sensor data
        self.sensor_interface = SensorInterface()
        self.wallclock_t0 = None
        self._carla_host = carla_host
        self._carla_port = carla_port
        self._client = carla.Client(self._carla_host, self._carla_port)
        self.role_name = role_name
        self._agent: BasicAgent = None
        self._plan = None
        self._debug = debug
        self._opt_dict = opt_dict

    def setup(self, path_to_conf_file, route=None, trajectory=None):
        actors = CarlaDataProvider.get_world().get_actors()
        
        # search agent named like "hero_xx"
        pattern = r'^hero_\d{1,3}(?<=\d)$'
        vehicle = [
            actor
            for actor in actors
            if isinstance(actor.attributes.get("role_name"), str) and 
            re.match(pattern, actor.attributes.get("role_name")) is not None
        ][0]

        if vehicle:
            self._agent = BasicAgent(vehicle, target_speed=50, opt_dict=self._opt_dict)
        world = CarlaDataProvider.get_world()
        if self._agent and route:
            plan = []
            map = CarlaDataProvider.get_map()
            for tf, option in route:
                wp = (map.get_waypoint(tf.location), option)
                plan.append(wp)
                if self._debug:
                    world.debug.draw_point(
                        tf.location, size=0.1, color=carla.Color(0, 255, 0), life_time=120.0
                    )
            self._agent.set_global_plan(plan)

        elif self._agent and trajectory:
            plan = []
            waypoints = []
            for item in trajectory:
                if isinstance(item, tuple):
                    wp, _ = item
                else:
                    wp = item

                if self._debug:
                    world.debug.draw_point(
                        wp, size=0.2, color=carla.Color(255, 0, 0), life_time=120.0
                    )

                waypoints.append(wp)

            _, route = interpolate_trajectory(waypoints)
            map = CarlaDataProvider.get_map()
            for tf, option in route:
                wp = (map.get_waypoint(tf.location), option)
                plan.append(wp)
                if self._debug:
                    world.debug.draw_point(
                        tf.location,
                        size=0.1,
                        color=carla.Color(0, 255, 0),
                        life_time=120.0,
                    )

            self._agent.set_global_plan(plan)

    def sensors(self):
        sensors = []
        '''
        Duplicate the setting OPV2V used
        Calculated with the cords between sensor and vehicle position
        'width'/'height' could be determined by png
        however 'fov' is unknown, so set to default 100 degree
        '''
        sensors = [
            {
                "type": "sensor.camera.rgb",
                'x': 2.5, 
                'y': 0, 
                'z': 1.0, 
                'roll': 0.0, 
                'pitch': 0.0, 
                'yaw': 0.0,
                "width": 800,
                "height": 600,
                "fov": 100,
                "id": "rgb_front",
            },
            {
                "type": "sensor.camera.rgb",
                'x': 0.0, 
                'y': 0.3, 
                'z': 1.8, 
                'roll': 0, 
                'pitch': 0.0, 
                'yaw': 100,
                "width": 800,
                "height": 600,
                "fov": 100,
                "id": "rgb_right",
            },
            {
                "type": "sensor.camera.rgb",
                'x': 0, 
                'y': -0.3, 
                'z': 1.8, 
                'roll': 0, 
                'pitch': 0.0, 
                'yaw': -100,
                "width": 800,
                "height": 600,
                "fov": 100,
                "id": "rgb_left",
            },
            {
                "type": "sensor.camera.rgb",
                'x': -2.0, 
                'y': 0.0, 
                'z': 1.5, 
                'roll': 0, 
                'pitch': 0.0, 
                'yaw': 180,
                "width": 800,
                "height": 600,
                "fov": 100,
                "id": "rgb_back",
            },
            # same params with rgb_front
            {
                "type": "sensor.camera.depth",
                'x': 2.5, 
                'y': 0, 
                'z': 1.0, 
                'roll': 0.0, 
                'pitch': 0.0, 
                'yaw': 0.0,
                "width": 800,
                "height": 600,
                "fov": 100,
                "id": "depth_front",
            },
            {
                "type": "sensor.lidar.ray_cast",
                'x': 0.5, 
                'y': 0, 
                'z': 1.9, 
                'roll': 0.0, 
                'pitch': 0.0, 
                'yaw': 0.0,
                'channels': 32,
                'range': 120,
                'points_per_second': 1000000,
                'rotation_frequency': 20, # the simulation is 20 fps
                'upper_fov': 2,
                'lower_fov': -25,
                'dropoff_general_rate': 0.3,
                'dropoff_intensity_limit': 0.7,
                'dropoff_zero_intensity': 0.4,
                'noise_stddev': 0.02,
                "id": "lidar",
            },
            # {
            #     "type": "sensor.lidar.ray_cast_semantic",
            #     'x': 0.5, 
            #     'y': 0, 
            #     'z': 1.9, 
            #     'roll': 0.0, 
            #     'pitch': 0.0, 
            #     'yaw': 0.0,
            #     'channels': 32,
            #     'range': 120,
            #     'points_per_second': 100000,
            #     'rotation_frequency': 20, # the simulation is 20 fps
            #     'upper_fov': 2,
            #     'lower_fov': -25,
            #     'dropoff_general_rate': 0.3,
            #     'dropoff_intensity_limit': 0.7,
            #     'dropoff_zero_intensity': 0.4,
            #     'noise_stddev': 0.02,
            #     "id": "semantic_lidar",
            # },
            {
                "type": "sensor.other.gnss", 
                "x": 0.7, 
                "y": -0.4, 
                "z": 1.60, 
                "id": "gps"
            },
        ]

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """
        control = self._agent.run_step()
        return control

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)
        if self._agent:
            self._agent.set_global_plan(global_plan_world_coord)
