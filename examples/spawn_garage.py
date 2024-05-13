import os

CARLA_ROOT = os.getenv('1', 'D:\Code\carla\CARLA_0.9.15')
WORK_DIR = os.getenv('2', 'D:\Code\.SOTA\carla_garage')
MATS_ROOT = os.getenv('3', 'D:\Code\carla\mats-gym')

# Use leaderboard 2.0 and corresponding scenario_runner
# PYTHONPATH += f';{WORK_DIR}\\scenario_runner'
# PYTHONPATH += f';{WORK_DIR}\\leaderboard'
CARLA_SERVER = f'{CARLA_ROOT}\\CarlaUE4.exe'
# SRUNNER_FOLDER = f'D:\Code\carla\scenario_runner'
LEADERBOARD_FOLDER = f'D:\Code\carla\leaderboard' 

PYTHONPATH = os.environ.get('PYTHONPATH', '')
PYTHONPATH += f';{CARLA_ROOT}\\PythonAPI'
PYTHONPATH += f';{CARLA_ROOT}\\PythonAPI\\carla'
# PYTHONPATH += f';{CARLA_ROOT}\\PythonAPI\\carla\\dist\\carla-0.9.15-py3.7-win-amd64.egg'
# PYTHONPATH += f';{SRUNNER_FOLDER}'
PYTHONPATH += f';{LEADERBOARD_FOLDER}'
os.environ['PYTHONPATH'] = PYTHONPATH

# SCENARIOS = f'{WORK_DIR}\\leaderboard\\data\\scenarios\\eval_scenarios.json'
ROUTES_SUBSET=0
# ROUTES = f'D:\Code\leaderboard/data/routes_devtest.xml'
# ROUTES = f'{LEADERBOARD_FOLDER}\\data\\routes_devtest.xml'
ROUTES = f'{WORK_DIR}\\leaderboard\\data\\longest6.xml'
REPETITIONS = '1'
CHALLENGE_TRACK_CODENAME = 'SENSORS'
CHECKPOINT_ENDPOINT = f'{WORK_DIR}\\results\\transfuser_plus_plus_longest6.json'
TEAM_AGENT = f'{WORK_DIR}\\team_code\\sensor_agent.py'
AGENT_CONFIG = f'{WORK_DIR}\\pretrained_models\\longest6\\tfpp_all_0'
DEBUG_CHALLENGE = '0'
RESUME = '1'
DATAGEN = '0'
SAVE_PATH = f'{WORK_DIR}\\results'

UNCERTAINTY_THRESHOLD = '0.33'
BENCHMARK = 'longest6'

# STOP_CONTROL=1
# BENCHMARK = 'LAV'

os.system(f'python {MATS_ROOT}/examples/16_spawn_garage.py \
          --routes-subset={ROUTES_SUBSET} --routes={ROUTES} --repetitions={REPETITIONS} \
          --track={CHALLENGE_TRACK_CODENAME} --checkpoint={CHECKPOINT_ENDPOINT} \
          --agent={TEAM_AGENT} --agent-config={AGENT_CONFIG} --debug=0 --resume={RESUME} --timeout=600')