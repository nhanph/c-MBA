from gym import spaces
import torch as th
import torch.distributions as tdist
import numpy as np

from .basic_controller import BasicMAC
from modules.agents import REGISTRY as agent_REGISTRY


# This multi-agent controller shares parameters between agents
class SingleBasicMAC(BasicMAC):

    def __init__(self, scheme, groups, args):

        super(SingleBasicMAC, self).__init__(scheme, groups, args)

        if hasattr(args,'attack_agent'):
            if args.attack_agent is not None:
                self.attack_agent = args.attack_agent
            else:
                raise ValueError('Need to specify attack_agent')
        else:
            raise ValueError('Need to specify attack_agent')

        # indice of non-adversary agents
        self.non_adv_agent_idx = np.delete(np.arange(args.n_agents),self.attack_agent)


    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        
        chosen_actions = self.forward(ep_batch[bs], t_ep, test_mode=test_mode, select_actions=True)["actions"]
        
        # print('ddd',ep_batch[bs].batch_size)
        chosen_actions = chosen_actions.view(ep_batch[bs].batch_size, self.n_agents, self.args.n_actions).detach()

        # Now do appropriate exploration
        exploration_mode = getattr(self.args, "exploration_mode", "gaussian")
        if not test_mode:
            if exploration_mode == "ornstein_uhlenbeck":
                x = getattr(self, "ou_noise_state", chosen_actions.clone().zero_())
                mu = 0
                theta = getattr(self.args, "ou_theta", 0.15)
                sigma = getattr(self.args, "ou_sigma", 0.2)
                noise_scale = getattr(self.args, "ou_noise_scale", 0.3) if t_env < self.args.env_args["episode_limit"]*self.args.ou_stop_episode else 0.0

                dx = theta * (mu - x) + sigma * x.clone().normal_()
                self.ou_noise_state = x + dx
                ou_noise = self.ou_noise_state * noise_scale
                chosen_actions[:,self.attack_agent:self.attack_agent+1] = chosen_actions[:,self.attack_agent:self.attack_agent+1] + ou_noise
            elif exploration_mode == "gaussian":
                start_steps = getattr(self.args, "start_steps", 0)
                act_noise = getattr(self.args, "act_noise", 0.1)
                # print(chosen_actions.shape, self.args.n_agents, self.attack_agent, self.args.action_spaces)
                if t_env >= start_steps:
                    x = chosen_actions[:,self.attack_agent:self.attack_agent+1].clone().zero_()
                    chosen_actions[:,self.attack_agent:self.attack_agent+1] += act_noise * x.clone().normal_()
                else:
                    if self.args.env_args["scenario"] in ["Humanoid-v2", "HumanoidStandup-v2"]:
                        chosen_actions[:,self.attack_agent:self.attack_agent+1] = th.from_numpy(np.array(self.args.action_spaces[0].sample())).unsqueeze(0).float().to(device=ep_batch.device)
                    else:
                        chosen_actions[:,self.attack_agent:self.attack_agent+1] = th.from_numpy(np.array(self.args.action_spaces[self.attack_agent].sample())).unsqueeze(0).float().to(device=ep_batch.device)

        # now clamp actions to permissible action range (necessary after exploration)
        if all([isinstance(act_space, spaces.Box) for act_space in self.args.action_spaces]):
            for _aid in range(self.n_agents):
                for _actid in range(self.args.action_spaces[_aid].shape[0]):
                    chosen_actions[:, _aid, _actid].clamp_(np.asscalar(self.args.action_spaces[_aid].low[_actid]),
                                                           np.asscalar(self.args.action_spaces[_aid].high[_actid]))
        elif all([isinstance(act_space, spaces.Tuple) for act_space in self.args.action_spaces]):
            for _aid in range(self.n_agents):
                for _actid in range(self.args.action_spaces[_aid].spaces[0].shape[0]):
                    chosen_actions[:, _aid, _actid].clamp_(self.args.action_spaces[_aid].spaces[0].low[_actid],
                                                           self.args.action_spaces[_aid].spaces[0].high[_actid])
                for _actid in range(self.args.action_spaces[_aid].spaces[1].shape[0]):
                    tmp_idx = _actid + self.args.action_spaces[_aid].spaces[0].shape[0]
                    chosen_actions[:, _aid, tmp_idx].clamp_(self.args.action_spaces[_aid].spaces[1].low[_actid],
                                                            self.args.action_spaces[_aid].spaces[1].high[_actid])
        return chosen_actions

    def get_weight_decay_weights(self):
        return self.agent.get_weight_decay_weights()

    def forward(self, ep_batch, t, actions=None, select_actions=False, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)

        all_agent_outs = th.zeros((self.n_agents, self.args.n_actions), device=agent_inputs.device)

        # ret = self.agent(agent_inputs)['actions']

        # print('bbb', atk_inputs.shape, non_atk_inputs.shape)
        ret1 = self.agent(agent_inputs)['actions']
        ret2 = self.trained_agent(agent_inputs)['actions']

        # print('aaa', agent_inputs.shape, ret1.shape)
        # all_agent_outs = th.zeros((self.n_agents, self.args.n_actions), device=agent_inputs.device)
        all_agent_outs = th.zeros(ret1.shape, device=agent_inputs.device)
        all_agent_outs[self.attack_agent] = ret1[self.attack_agent]
        all_agent_outs[self.non_adv_agent_idx] = ret2[self.non_adv_agent_idx]

        if select_actions:
            return {'actions': all_agent_outs}
        # print(ret)
        # agent_outs = ret["actions"]

        # if self.agent_output_type == "pi_logits":
        #     agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
        #     if not test_mode:
        #         agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
        #                       + th.ones_like(agent_outs) * self.action_selector.epsilon/agent_outs.size(-1))
        # return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), actions
        return all_agent_outs.view(ep_batch.batch_size, self.n_agents, -1), actions

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])

        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions"][:, t]))
            else:
                inputs.append(batch["actions"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)

        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

    def cuda(self, device="cuda"):
        self.agent.cuda(device=device)
        self.trained_agent.cuda(device=device)

    def parameters(self):
        return self.agent.parameters()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        self.trained_agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def load_trained_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.trained_agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
