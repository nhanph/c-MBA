#!/bin/bash


cd ../../../

ENV_ARGS="env_args.scenario=""Walker2d-v2"" env_args.agent_conf=""2x3"""


CONFIG="pgd_lr=0.5 pgd_step=30 plan_step=1 test_nepisode=50 \
		dynamic_model_path="""results/dynamic_models/Walker2d-v2/model.pt""" \
		checkpoint_path=""results/marl_models/maddpg_walker"" \
		init_type=epsilon seed=42 evaluate=True pgd_solver=adam\
		noise_constraint_type=""linf_norm""\
		adv_eps=[0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.0]\
		"


NAME=maddpg_model_atk_linf_norm_walker2x3_opt_new
AGENT_CONFIG="optimal_adv_agent=True num_atk_agent=1 brute_force_atk=False attack_again=True"

python src/main.py --config=maddpg_adv_model --env-config=mujoco_multi with ${ENV_ARGS} name=${NAME} ${CONFIG} ${AGENT_CONFIG}
