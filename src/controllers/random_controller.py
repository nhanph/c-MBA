from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from .basic_controller import BasicMAC
from gym import spaces
import numpy as np

# This multi-agent controller shares parameters between agents
class RANDOM_MAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(RANDOM_MAC, self).__init__(scheme, groups, args)

        self.action_device = None
        
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        if self.args.agent in ["naf", "mlp"]:
            if self.action_device is None:
                chosen_actions = self.forward(ep_batch[bs], t_ep, test_mode=test_mode, select_actions=True)["actions"]
                self.action_device = chosen_actions.device
                chosen_actions = chosen_actions.view(ep_batch[bs].batch_size, self.n_agents, self.args.n_actions).detach()
        elif self.args.agent in ["cem"]:
            if self.action_device is None:
                chosen_actions = self.cem_sampling(ep_batch, t_ep, bs)
                self.action_device = chosen_actions.device
        else:
            raise Exception("No known agent type selected! ({})".format(self.args.agent))

        if self.action_device:
            chosen_actions = th.zeros(ep_batch[bs].batch_size, self.n_agents, self.args.n_actions, device=self.action_device)

        if all([isinstance(act_space, spaces.Box) for act_space in self.args.action_spaces]):
            for _aid in range(self.n_agents):
                for _actid in range(self.args.action_spaces[_aid].shape[0]):
        #             chosen_actions[:, _aid, _actid].clamp_(np.asscalar(self.args.action_spaces[_aid].low[_actid]),
        #                                                    np.asscalar(self.args.action_spaces[_aid].high[_actid]))
                    low = np.asscalar(self.args.action_spaces[_aid].low[_actid])
                    high = np.asscalar(self.args.action_spaces[_aid].high[_actid])
                    chosen_actions[:, _aid, _actid] = (high-low)*th.rand(chosen_actions[:, _aid, _actid].shape, device=self.action_device) + low
        elif all([isinstance(act_space, spaces.Tuple) for act_space in self.args.action_spaces]):
            for _aid in range(self.n_agents):
                for _actid in range(self.args.action_spaces[_aid].spaces[0].shape[0]):
                    # chosen_actions[:, _aid, _actid].clamp_(self.args.action_spaces[_aid].spaces[0].low[_actid],
                    #                                        self.args.action_spaces[_aid].spaces[0].high[_actid])
                    low = self.args.action_spaces[_aid].spaces[0].low[_actid]
                    high = self.args.action_spaces[_aid].spaces[0].high[_actid]
                    chosen_actions[:, _aid, _actid] = (high-low)*th.rand(chosen_actions[:, _aid, _actid].shape, device=self.action_device) + low
                for _actid in range(self.args.action_spaces[_aid].spaces[1].shape[0]):
                    tmp_idx = _actid + self.args.action_spaces[_aid].spaces[0].shape[0]
                    # chosen_actions[:, _aid, tmp_idx].clamp_(self.args.action_spaces[_aid].spaces[1].low[_actid],
                    #                                         self.args.action_spaces[_aid].spaces[1].high[_actid])
                    low = self.args.action_spaces[_aid].spaces[1].low[_actid]
                    high = self.args.action_spaces[_aid].spaces[1].high[_actid]
                    chosen_actions[:, _aid, tmp_idx] = (high-low)*th.rand(chosen_actions[:, _aid, _actid].shape, device=self.action_device) + low
        return chosen_actions

    def forward(self, ep_batch, t, actions=None, select_actions=False, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        ret = self.agent(agent_inputs, actions=actions)
        if select_actions:
            return ret
        # print(ret)
        agent_outs = ret["actions"]

        if self.agent_output_type == "pi_logits":
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                              + th.ones_like(agent_outs) * self.action_selector.epsilon/agent_outs.size(-1))
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), actions

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