#!/bin/bash


cd ../../../

ENV_ARGS="env_args.scenario=""Ant-v2"" env_args.agent_conf=""4x2"""


CONFIG="test_nepisode=50 \
		checkpoint_path=""results/marl_models/maddpg_ant"" \
		single_adv_model_path=""results/adv_models/adv_ddpg_ant""\
		seed=42 evaluate=True
		noise_constraint_type=""linf_norm""\
		adv_eps=[0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25]\
		fgsm_iter=30\
		"

for agent in 0 1 2 3
  do
	NAME=fgsm_atk_linf_norm_ant_4x2_new_agent${agent}
	AGENT_CONFIG="attack_agent=${agent}"

	python src/main.py --config=maddpg_adv_fgsm --env-config=mujoco_multi with ${ENV_ARGS} name=${NAME} ${CONFIG} ${AGENT_CONFIG}

done
