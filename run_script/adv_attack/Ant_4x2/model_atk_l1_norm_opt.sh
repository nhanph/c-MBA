#!/bin/bash


cd ../../../

ENV_ARGS="env_args.scenario=""Ant-v2"" env_args.agent_conf=""4x2"""


CONFIG="pgd_lr=0.5 pgd_step=30 plan_step=1 test_nepisode=50 \
		dynamic_model_path="""results/dynamic_models/Ant_4x2/model.pt""" \
		checkpoint_path=""results/marl_models/maddpg_ant"" \
		init_type=epsilon seed=42 evaluate=True pgd_solver=adam\
		noise_constraint_type=""l1_norm""\
		adv_eps=[0,0.5,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.,5.5,6.,6.5,7.,7.5,8.,8.5,9.,9.5,10.,10.]\
		"

NAME=maddpg_model_atk_l1_norm_ant_4x2_opt
AGENT_CONFIG="optimal_adv_agent=True num_atk_agent=1 brute_force_atk=False"

python src/main.py --config=maddpg_adv_model --env-config=mujoco_multi with ${ENV_ARGS} name=${NAME} ${CONFIG} ${AGENT_CONFIG}