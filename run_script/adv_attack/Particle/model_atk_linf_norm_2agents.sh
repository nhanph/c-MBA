#!/bin/bash


cd ../../../

ENV_ARGS=""


CONFIG="pgd_lr=0.1 pgd_step=30 plan_step=1 test_nepisode=50 \
		dynamic_model_path="""results/dynamic_models/particle/model.pt""" \
		checkpoint_path=""results/marl_models/maddpg_particle_simple"" \
		init_type=epsilon seed=42 evaluate=True pgd_solver=adam\
		noise_constraint_type=""linf_norm"" data_driven_failure_state=True\
		adv_eps=[0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.0]\
		"

# 0,1
NAME=maddpg_model_atk_linf_norm_particle_agent01
AGENT_CONFIG="attack_agent=[0,1] num_atk_agent=2"

python src/main.py --config=maddpg_adv_model --env-config=particle with ${ENV_ARGS} name=${NAME} ${CONFIG} ${AGENT_CONFIG}

# 0,2
NAME=maddpg_model_atk_linf_norm_particle_agent02
AGENT_CONFIG="attack_agent=[0,2] num_atk_agent=2"

python src/main.py --config=maddpg_adv_model --env-config=particle with ${ENV_ARGS} name=${NAME} ${CONFIG} ${AGENT_CONFIG}

# 1,2
NAME=maddpg_model_atk_linf_norm_particle_agent12
AGENT_CONFIG="attack_agent=[1,2] num_atk_agent=2"

python src/main.py --config=maddpg_adv_model --env-config=particle with ${ENV_ARGS} name=${NAME} ${CONFIG} ${AGENT_CONFIG}
