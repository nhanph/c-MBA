#!/bin/bash


cd ../../../

ENV_ARGS="env_args.scenario=""Ant-v2"" env_args.agent_conf=""4x2"""

CORE_CONFIG="test_nepisode=50 \
		checkpoint_path=""results/marl_models/maddpg_ant"" \
		seed=42 evaluate=True\
		noise_constraint_type=""l1_norm""\
		adv_eps=[0,0.5,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.,5.5,6.,6.5,7.,7.5,8.,8.5,9.,9.5,10.,10.]
		"

for agent in 0 1 2 3
  do
	# Uniform noise attack
	CONFIG="adv_noise_type=uniform"

	NAME=maddpg_adv_noise_l1_norm_uniform_ant_4x2_agent${agent}
	AGENT_CONFIG="attack_agent=${agent}"

	python src/main.py --config=maddpg_adv_noise --env-config=mujoco_multi with ${ENV_ARGS} name=${NAME} ${CORE_CONFIG} ${CONFIG} ${AGENT_CONFIG}

	## Gaussian noise attack
	CONFIG="adv_noise_type=gaussian"

	NAME=maddpg_adv_noise_l1_norm_gaussian_ant_4x2_agent${agent}
	AGENT_CONFIG="attack_agent=${agent}"

	python src/main.py --config=maddpg_adv_noise --env-config=mujoco_multi with ${ENV_ARGS} name=${NAME} ${CORE_CONFIG} ${CONFIG} ${AGENT_CONFIG}

done