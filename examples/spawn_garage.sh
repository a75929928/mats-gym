export CARLA_ROOT=/home/hjh/carla/CARLA_0915
export WORK_DIR=/home/hjh/.SOTA/carla_garage
export MATS_ROOT=/home/hjh/carla/mats-gym

# export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
# export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
# export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

# export SCENARIOS=${WORK_DIR}/leaderboard/data/scenarios/eval_scenarios.json
# export ROUTES=${WORK_DIR}/leaderboard/data/longest6.xml
export REPETITIONS=1
export CHALLENGE_TRACK_CODENAME=SENSORS
export CHECKPOINT_ENDPOINT=${WORK_DIR}/results/transfuser_plus_plus_longest6.json
export TEAM_AGENT=${WORK_DIR}/team_code/sensor_agent.py
export TEAM_CONFIG=${WORK_DIR}/pretrained_models/longest6/plant_all_1
export DEBUG_CHALLENGE=0
export RESUME=1
export DATAGEN=0
export SAVE_PATH=${WORK_DIR}/results
export UNCERTAINTY_THRESHOLD=0.33
export BENCHMARK=longest6

python3 ${MATS_ROOT}/examples/16_spawn_garage.py \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=0 \
--resume=${RESUME} \
--timeout=600