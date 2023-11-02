#!/bin/bash


cd ../../../

ENV_ARGS="env_args.scenario=""Ant-v2"" env_args.agent_conf=""4x2"""


CONFIG="pgd_lr=0.5 pgd_step=30 plan_step=1 test_nepisode=16 \
		dynamic_model_path="""results/dynamic_models/Ant_4x2/model.pt""" \
		checkpoint_path=""results/marl_models/maddpg_ant"" \
		init_type=epsilon seed=42 evaluate=True pgd_solver=adam\
		noise_constraint_type=""linf_norm""\
		adv_eps=[0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25]\
		"

for agent in 0 1 2 3
  do
	NAME=maddpg_model_atk_fix_linf_norm_ant_4x2_agent${agent}
	AGENT_CONFIG="attack_agent=${agent}"

	python src/main.py --config=maddpg_adv_model --env-config=mujoco_multi with ${ENV_ARGS} name=${NAME} ${CONFIG} ${AGENT_CONFIG}

done
