import yaml

# 假设你的YAML文件内容保存在config.yml中
yaml_file = """
CARLA_ROOT: 'D:/Code/carla/CARLA_0.9.15'
WORK_DIR: 'D:/Code/.SOTA/carla_garage'
MATS_ROOT: 'D:/Code/carla/mats-gym'
CARLA_SERVER: '%CARLA_ROOT%/CarlaUE4.exe'

PYTHONPATH: ''
PYTHONPATH: '%PYTHONPATH%;%CARLA_ROOT%/PythonAPI;%CARLA_ROOT%/PythonAPI/carla'

SCENARIOS: '%WORK_DIR%/leaderboard/data/scenarios/eval_scenarios.json'
ROUTES: '%MATS_ROOT%/scenarios/routes/training.xml'
REPETITIONS: '1'
CHALLENGE_TRACK_CODENAME: 'SENSORS'
CHECKPOINT_ENDPOINT: '%WORK_DIR%/results/transfuser_plus_plus_longest6.json'
TEAM_AGENT: '%WORK_DIR%/team_code/sensor_agent.py'
AGENT_CONFIG: '%WORK_DIR%/pretrained_models/longest6/tfpp_all_0'
DEBUG_CHALLENGE: '0'
RESUME: '1'
DATAGEN: '0'
SAVE_PATH: '%WORK_DIR%/results'
"""

# 使用yaml.safe_load加载YAML内容
config = yaml.safe_load(yaml_file)

# 定义替换函数
def replace_config_values(config_dict, key_value_dict):
    for key, value in config_dict.items():
        if isinstance(value, str):
            # 替换所有的%key%为对应的值
            for k, v in key_value_dict.items():
                value = value.replace(f'%{k}%', v)
            config_dict[key] = value
        elif isinstance(value, dict):
            # 如果是字典，递归调用替换函数
            replace_config_values(value, key_value_dict)

# 从config中提取键值对到一个新的字典中，用于替换
key_value_dict = {k: v for k, v in config.items() if not k.startswith('%')}
replace_config_values(config, key_value_dict)

# 打印替换后的CARLA_SERVER和ROUTES
print(config['CARLA_SERVER'])  # 输出: 'D:/Code/carla/CARLA_0.9.15/CarlaUE4.exe'
print(config['ROUTES'])        # 输出: 'D:/Code/carla/mats-gym/scenarios/routes/training.xml'