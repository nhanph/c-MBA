from gym import spaces
import torch as th
import torch.distributions as tdist
import numpy as np

from .basic_controller import BasicMAC
from utils.rl_utils import project_l1_ball, project_l2_ball


# This multi-agent controller shares parameters between agents
class CQMixAdvNoiseMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(CQMixAdvNoiseMAC, self).__init__(scheme, groups, args)

        if hasattr(args,'attack_agent'):
            if args.attack_agent is not None:
                if type(args.attack_agent) != list:
                    self.attack_agent = [args.attack_agent]
                else:
                    self.attack_agent = args.attack_agent
            else:
                raise ValueError('Need to specify attack_agent')
        else:
            raise ValueError('Need to specify attack_agent')

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, epsilon=0.):
        # if self.args.agent in ["naf", "mlp"]:
        chosen_actions, info, final_noise = self.forward(ep_batch[bs], t_ep, test_mode=test_mode, select_actions=True, epsilon=epsilon)
        chosen_actions = chosen_actions.view(ep_batch[bs].batch_size, self.n_agents, self.args.n_actions).detach()
        # elif self.args.agent in ["cem"]:
        #     chosen_actions = self.cem_sampling(ep_batch, t_ep, bs)
        # else:
        #     raise Exception("No known agent type selected! ({})".format(self.args.agent))

        
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
        return chosen_actions, info, final_noise

    def get_weight_decay_weights(self):
        return self.agent.get_weight_decay_weights()

    def forward(self, ep_batch, t, actions=None, select_actions=False, test_mode=False, epsilon=0.):
        agent_inputs = self._build_inputs(ep_batch, t)

        final_noise = th.zeros_like(agent_inputs).to(agent_inputs.device)

        # print(final_noise.shape)

        if epsilon > 0:
            # select attack agent
            if type(self.attack_agent) == list:
                if len(self.attack_agent) <= self.args.num_atk_agent:
                    agents = self.attack_agent
                else:
                    # random from a set
                    agents = np.random.choice(self.attack_agent, self.args.num_atk_agent, replace=False)
            else:
                raise ValueError('Unsupported type of attack_agent',type(self.attack_agent))

            # generate noise
            if self.args.adv_noise_type=='uniform':
                noise = epsilon*(2*th.rand(agent_inputs.shape, device=agent_inputs.device)-1)
            else:    
                noise = epsilon*th.randn(agent_inputs.shape, device=agent_inputs.device)
            
            if self.args.noise_constraint_type == 'l1_norm':
                for agent in agents:
                    noise[agent] = project_l1_ball(noise[agent], rad=epsilon)
            elif self.args.noise_constraint_type == 'l2_norm':
                for agent in agents:
                    noise[agent] = project_l2_ball(noise[agent], rad=epsilon)
            else:
                noise = noise.clamp(-epsilon,epsilon)

            agent_inputs[agents,:] += noise[agents,:]

            final_noise[agents] = noise[agents].detach()

            # noise_info = {
            #     "l1_norm": th.linalg.norm(noise[agents].view(self.args.num_atk_agent,-1), ord=1).item(),
            #     "l2_norm": th.linalg.norm(noise[agents].view(self.args.num_atk_agent,-1), ord=2).item(),
            #     "linf_norm": th.linalg.norm(noise[agents].view(self.args.num_atk_agent,-1), ord=float('inf')).item()
            # }
            noise_info = {
                "l1_norm": th.max(th.sum(th.abs(final_noise.view(self.n_agents,-1)), 1)).item(),
                "l2_norm": th.max(th.sqrt(th.sum(th.square(final_noise.view(self.n_agents,-1)), 1))).item(),
                "linf_norm": th.max(th.abs(final_noise.view(self.n_agents,-1))).item()
            }
        else:
            noise_info = {
                "l1_norm": 0.,
                "l2_norm": 0.,
                "linf_norm": 0.,
            }

        ret = self.agent(agent_inputs, actions=actions)

        return ret["actions"], noise_info, final_noise.to('cpu').clone().detach().numpy().tolist()
        

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

    def cem_sampling(self, ep_batch, t, bs):
        # Number of samples from the param distribution
        N = 64
        # Number of best samples we will consider
        Ne = 6

        ftype = th.FloatTensor if not next(self.agent.parameters()).is_cuda else th.cuda.FloatTensor
        mu = ftype(ep_batch[bs].batch_size, self.n_agents, self.args.n_actions).zero_()
        std = ftype(ep_batch[bs].batch_size, self.n_agents, self.args.n_actions).zero_() + 1.0
        its = 0
        maxits = 2
        agent_inputs = self._build_inputs(ep_batch[bs], t)

        while its < maxits:
            dist = tdist.Normal(mu.view(-1, self.args.n_actions), std.view(-1, self.args.n_actions))
            actions = dist.sample((N,)).detach()
            actions_prime = th.tanh(actions)
            ret = self.agent(agent_inputs.unsqueeze(0).expand(N, *agent_inputs.shape).contiguous().view(-1, agent_inputs.shape[-1]),
                             actions=actions_prime.view(-1, actions_prime.shape[-1]))
            out = ret["Q"].view(N, -1, 1)
            topk, topk_idxs = th.topk(out, Ne, dim=0)
            mu = th.mean(actions.gather(0, topk_idxs.repeat(1, 1, self.args.n_actions).long()), dim=0)
            std = th.std(actions.gather(0, topk_idxs.repeat(1, 1, self.args.n_actions).long()), dim=0)
            its += 1
        topk, topk_idxs = th.topk(out, 1, dim=0)
        action_prime = th.mean(actions_prime.gather(0, topk_idxs.repeat(1, 1, self.args.n_actions).long()), dim=0)
        chosen_actions = action_prime.clone().view(ep_batch[bs].batch_size, self.n_agents, self.args.n_actions).detach()
        return chosen_actions