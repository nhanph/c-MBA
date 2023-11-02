#!/bin/bash


cd ../../../

ENV_ARGS="env_args.scenario=""Walker2d-v2"" env_args.agent_conf=""2x3"""


CONFIG="test_nepisode=50 \
		checkpoint_path=""results/marl_models/maddpg_walker"" \
		single_adv_model_path=""results/adv_models/adv_ddpg_walker""\
		seed=42 evaluate=True
		noise_constraint_type=""linf_norm""\
		adv_eps=[0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.0]\
		fgsm_iter=30\
		"

for agent in 0 1
  do
	NAME=fgsm_atk_linf_norm_walker2x3_new_agent${agent}
	AGENT_CONFIG="attack_agent=${agent}"

	python src/main.py --config=maddpg_adv_fgsm --env-config=mujoco_multi with ${ENV_ARGS} name=${NAME} ${CONFIG} ${AGENT_CONFIG}

done