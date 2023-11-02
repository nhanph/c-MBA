#!/bin/bash

cd ../../../

ENV_ARGS="env_args.scenario=""Ant-v2"" env_args.agent_conf=""4x2"""

CORE_CONFIG="test_nepisode=50 \
		checkpoint_path=""results/marl_models/maddpg_ant"" \
		seed=42 evaluate=True\
		noise_constraint_type=""linf_norm""\
		adv_eps=[0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25]\
		"

agent1=0
agent2=1

echo "Attacking agent ${agent1} and ${agent2}"
# Uniform noise attack
CONFIG="adv_noise_type=uniform"

NAME=maddpg_adv_noise_linf_norm_uniform_ant_4x2_agent${agent1}${agent2}
AGENT_CONFIG="attack_agent=[${agent1},${agent2}] num_atk_agent=2"

python src/main.py --config=maddpg_adv_noise --env-config=mujoco_multi with ${ENV_ARGS} name=${NAME} ${CORE_CONFIG} ${CONFIG} ${AGENT_CONFIG}

## Gaussian noise attack
CONFIG="adv_noise_type=gaussian"

NAME=maddpg_adv_noise_linf_norm_gaussian_ant_4x2_agent${agent1}${agent2}
AGENT_CONFIG="attack_agent=[${agent1},${agent2}] num_atk_agent=2"

python src/main.py --config=maddpg_adv_noise --env-config=mujoco_multi with ${ENV_ARGS} name=${NAME} ${CORE_CONFIG} ${CONFIG} ${AGENT_CONFIG}


agent1=0
agent2=2

echo "Attacking agent ${agent1} and ${agent2}"
# Uniform noise attack
CONFIG="adv_noise_type=uniform"

NAME=maddpg_adv_noise_linf_norm_uniform_ant_4x2_agent${agent1}${agent2}
AGENT_CONFIG="attack_agent=[${agent1},${agent2}] num_atk_agent=2"

python src/main.py --config=maddpg_adv_noise --env-config=mujoco_multi with ${ENV_ARGS} name=${NAME} ${CORE_CONFIG} ${CONFIG} ${AGENT_CONFIG}

## Gaussian noise attack
CONFIG="adv_noise_type=gaussian"

NAME=maddpg_adv_noise_linf_norm_gaussian_ant_4x2_agent${agent1}${agent2}
AGENT_CONFIG="attack_agent=[${agent1},${agent2}] num_atk_agent=2"

python src/main.py --config=maddpg_adv_noise --env-config=mujoco_multi with ${ENV_ARGS} name=${NAME} ${CORE_CONFIG} ${CONFIG} ${AGENT_CONFIG}

agent1=0
agent2=3

echo "Attacking agent ${agent1} and ${agent2}"
# Uniform noise attack
CONFIG="adv_noise_type=uniform"

NAME=maddpg_adv_noise_linf_norm_uniform_ant_4x2_agent${agent1}${agent2}
AGENT_CONFIG="attack_agent=[${agent1},${agent2}] num_atk_agent=2"

python src/main.py --config=maddpg_adv_noise --env-config=mujoco_multi with ${ENV_ARGS} name=${NAME} ${CORE_CONFIG} ${CONFIG} ${AGENT_CONFIG}

## Gaussian noise attack
CONFIG="adv_noise_type=gaussian"

NAME=maddpg_adv_noise_linf_norm_gaussian_ant_4x2_agent${agent1}${agent2}
AGENT_CONFIG="attack_agent=[${agent1},${agent2}] num_atk_agent=2"

python src/main.py --config=maddpg_adv_noise --env-config=mujoco_multi with ${ENV_ARGS} name=${NAME} ${CORE_CONFIG} ${CONFIG} ${AGENT_CONFIG}


agent1=1
agent2=2

echo "Attacking agent ${agent1} and ${agent2}"
# Uniform noise attack
CONFIG="adv_noise_type=uniform"

NAME=maddpg_adv_noise_linf_norm_uniform_ant_4x2_agent${agent1}${agent2}
AGENT_CONFIG="attack_agent=[${agent1},${agent2}] num_atk_agent=2"

python src/main.py --config=maddpg_adv_noise --env-config=mujoco_multi with ${ENV_ARGS} name=${NAME} ${CORE_CONFIG} ${CONFIG} ${AGENT_CONFIG}

## Gaussian noise attack
CONFIG="adv_noise_type=gaussian"

NAME=maddpg_adv_noise_linf_norm_gaussian_ant_4x2_agent${agent1}${agent2}
AGENT_CONFIG="attack_agent=[${agent1},${agent2}] num_atk_agent=2"

python src/main.py --config=maddpg_adv_noise --env-config=mujoco_multi with ${ENV_ARGS} name=${NAME} ${CORE_CONFIG} ${CONFIG} ${AGENT_CONFIG}


agent1=1
agent2=3

echo "Attacking agent ${agent1} and ${agent2}"
# Uniform noise attack
CONFIG="adv_noise_type=uniform"

NAME=maddpg_adv_noise_linf_norm_uniform_ant_4x2_agent${agent1}${agent2}
AGENT_CONFIG="attack_agent=[${agent1},${agent2}] num_atk_agent=2"

python src/main.py --config=maddpg_adv_noise --env-config=mujoco_multi with ${ENV_ARGS} name=${NAME} ${CORE_CONFIG} ${CONFIG} ${AGENT_CONFIG}

## Gaussian noise attack
CONFIG="adv_noise_type=gaussian"

NAME=maddpg_adv_noise_linf_norm_gaussian_ant_4x2_agent${agent1}${agent2}
AGENT_CONFIG="attack_agent=[${agent1},${agent2}] num_atk_agent=2"

python src/main.py --config=maddpg_adv_noise --env-config=mujoco_multi with ${ENV_ARGS} name=${NAME} ${CORE_CONFIG} ${CONFIG} ${AGENT_CONFIG}


agent1=2
agent2=3

echo "Attacking agent ${agent1} and ${agent2}"
# Uniform noise attack
CONFIG="adv_noise_type=uniform"

NAME=maddpg_adv_noise_linf_norm_uniform_ant_4x2_agent${agent1}${agent2}
AGENT_CONFIG="attack_agent=[${agent1},${agent2}] num_atk_agent=2"

python src/main.py --config=maddpg_adv_noise --env-config=mujoco_multi with ${ENV_ARGS} name=${NAME} ${CORE_CONFIG} ${CONFIG} ${AGENT_CONFIG}

## Gaussian noise attack
CONFIG="adv_noise_type=gaussian"

NAME=maddpg_adv_noise_linf_norm_gaussian_ant_4x2_agent${agent1}${agent2}
AGENT_CONFIG="attack_agent=[${agent1},${agent2}] num_atk_agent=2"

python src/main.py --config=maddpg_adv_noise --env-config=mujoco_multi with ${ENV_ARGS} name=${NAME} ${CORE_CONFIG} ${CONFIG} ${AGENT_CONFIG}

