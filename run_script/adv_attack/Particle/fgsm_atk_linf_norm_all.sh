#!/bin/bash


cd ../../../

ENV_ARGS=""


CONFIG="test_nepisode=50 \
		checkpoint_path=""results/marl_models/maddpg_particle_simple"" \
		single_adv_model_path=""results/adv_models/adv_ddpg_particle""\
		seed=42 evaluate=True
		noise_constraint_type=""linf_norm""\
		adv_eps=[0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.0]\
		fgsm_iter=30\
		"

NAME=fgsm_atk_linf_norm_particle_new_agent_all
AGENT_CONFIG="attack_agent=[0,1,2] num_atk_agent=3"

python src/main.py --config=maddpg_adv_fgsm --env-config=particle with ${ENV_ARGS} name=${NAME} ${CONFIG} ${AGENT_CONFIG}
