#!/bin/bash

cd ../../../

ENV_ARGS=""

CORE_CONFIG="test_nepisode=50 \
		checkpoint_path=""results/marl_models/maddpg_particle_simple"" \
		seed=42 evaluate=True\
		noise_constraint_type=""linf_norm""\
		adv_eps=[0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.0]\
		"

# 0 1
# Uniform noise attack
CONFIG="adv_noise_type=uniform"

NAME=maddpg_adv_noise_linf_norm_uniform_particle_agent01
AGENT_CONFIG="attack_agent=[0,1] num_atk_agent=2"

python src/main.py --config=maddpg_adv_noise --env-config=particle with ${ENV_ARGS} name=${NAME} ${CORE_CONFIG} ${CONFIG} ${AGENT_CONFIG}

## Gaussian noise attack
CONFIG="adv_noise_type=gaussian"

NAME=maddpg_adv_noise_linf_norm_gaussian_particle_agent01
AGENT_CONFIG="attack_agent=[0,1] num_atk_agent=2"

python src/main.py --config=maddpg_adv_noise --env-config=particle with ${ENV_ARGS} name=${NAME} ${CORE_CONFIG} ${CONFIG} ${AGENT_CONFIG}

# 0 2
# Uniform noise attack
CONFIG="adv_noise_type=uniform"

NAME=maddpg_adv_noise_linf_norm_uniform_particle_agent02
AGENT_CONFIG="attack_agent=[0,2] num_atk_agent=2"

python src/main.py --config=maddpg_adv_noise --env-config=particle with ${ENV_ARGS} name=${NAME} ${CORE_CONFIG} ${CONFIG} ${AGENT_CONFIG}

## Gaussian noise attack
CONFIG="adv_noise_type=gaussian"

NAME=maddpg_adv_noise_linf_norm_gaussian_particle_agent02
AGENT_CONFIG="attack_agent=[0,2] num_atk_agent=2"

python src/main.py --config=maddpg_adv_noise --env-config=particle with ${ENV_ARGS} name=${NAME} ${CORE_CONFIG} ${CONFIG} ${AGENT_CONFIG}

# 1 2
# Uniform noise attack
CONFIG="adv_noise_type=uniform"

NAME=maddpg_adv_noise_linf_norm_uniform_particle_agent12
AGENT_CONFIG="attack_agent=[1,2] num_atk_agent=2"

python src/main.py --config=maddpg_adv_noise --env-config=particle with ${ENV_ARGS} name=${NAME} ${CORE_CONFIG} ${CONFIG} ${AGENT_CONFIG}

## Gaussian noise attack
CONFIG="adv_noise_type=gaussian"

NAME=maddpg_adv_noise_linf_norm_gaussian_particle_agent12
AGENT_CONFIG="attack_agent=[1,2] num_atk_agent=2"

python src/main.py --config=maddpg_adv_noise --env-config=particle with ${ENV_ARGS} name=${NAME} ${CORE_CONFIG} ${CONFIG} ${AGENT_CONFIG}
