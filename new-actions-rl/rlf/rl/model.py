import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from method.utils import Conv2d3x3
from rlf.rl.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_space, args,
                 base=None,
                 dist_mem=None,
                 z_dim=None,
                 add_input_dim=0):
        super().__init__()
        self.obs_shape = obs_shape
        self.add_input_dim = add_input_dim

        if base is None:
            if args.soft_mudule:
                base = SoftModuleCNN
            if len(obs_shape) == 3:
                base = CNNBase_NEW
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError(
                    'Observation space is %s' % str(obs_shape))

        self.action_space = action_space
        self.args = args
        self.env_name = args.env_name

        use_action_output_size = 0

        if args.soft_mudule:
            self.base = base(obs_shape[0], add_input_dim, num_tasks=len(args.env_names),
                             action_output_size=use_action_output_size,
                             recurrent=args.recurrent_policy, hidden_size=args.state_encoder_hidden_size,
                             use_batch_norm=args.use_batch_norm, args=args)
        else:
            self.base = base(obs_shape[0], add_input_dim,
                             action_output_size=use_action_output_size,
                             recurrent=args.recurrent_policy, hidden_size=args.state_encoder_hidden_size,
                             use_batch_norm=args.use_batch_norm, args=args)

    def clone_fresh(self):
        p = Policy(self.obs_shape, self.action_space, self.args,
                   type(self.base) if self.base is not None else None,
                   self.add_input_dim)

        if list(self.parameters())[0].is_cuda:
            p = p.cuda()

        return p

    def get_policies(self):
        return [self]

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def get_pi(self, inputs, rnn_hxs, masks, add_input=None, task_encoding=None):
        if self.args.soft_mudule:
            value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, task_encoding, add_input)
        else:
            value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, add_input)
        self.prev_actor_features = actor_features

        dist = self.dist(actor_features, add_input)
        return dist, value

    def act(self, inputs, rnn_hxs, masks, task_encoding=None, deterministic=False, add_input=None):
        dist, value = self.get_pi(
            inputs, rnn_hxs, masks, add_input, task_encoding)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        self.prev_action = action

        action_log_probs = dist.log_probs(action.double() if self.args.use_dist_double else action)
        dist_entropy = dist.entropy()

        if self.args.use_dist_double:
            if self.action_space.__class__.__name__ != "Discrete":
                action = action.float()
            action_log_probs = action_log_probs.float()
            dist_entropy = dist_entropy.float()
        if len(dist_entropy.shape) == 1:
            dist_entropy = dist_entropy.unsqueeze(-1)
        extra = {
            'dist_entropy': dist_entropy
        }

        return value, action, action_log_probs, rnn_hxs, extra

    def get_value(self, inputs, rnn_hxs, masks, task_encoding, action, add_input):
        if self.args.soft_mudule:
            value, actor_features, _ = self.base(inputs, rnn_hxs, masks, task_encoding, add_input)
        else:
            value, actor_features, _ = self.base(inputs, rnn_hxs, masks, add_input)

        self.prev_actor_features = actor_features
        self.prev_action = action[:, :1].long()

        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, task_encoding, action, add_input):
        if self.args.soft_mudule:
            value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, task_encoding, add_input)
        else:
            value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, add_input)

        dist = self.dist(actor_features, add_input)

        self.prev_actor_features = actor_features
        self.prev_action = action

        action_log_probs = dist.log_probs(action.double() if self.args.use_dist_double else action)
        dist_entropy = dist.entropy()

        if self.args.use_dist_double:
            action_log_probs = action_log_probs.float()
            dist_entropy = dist_entropy.float()

        return value, action_log_probs, dist_entropy, rnn_hxs, dist


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0)
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs



class CNNBase_NEW(NNBase):
    def __init__(self, num_inputs, add_input_dim,
                 action_output_size=32,
                 recurrent=False, hidden_size=64,
                 use_batch_norm=False, args=None):
        super().__init__(recurrent, hidden_size, hidden_size)
        self.args = args

        if self.args.env_name.startswith('MiniGrid'):
            self.conv_layers = nn.ModuleList([
                Conv2d3x3(in_channels=num_inputs,
                          out_channels=8, downsample=True),
                # shape is now (-1, 8, 5, 5)
                Conv2d3x3(in_channels=8, out_channels=8, downsample=True),
                # shape is now (-1, 8, 3, 3)
                Conv2d3x3(in_channels=8, out_channels=16, downsample=False),
                # shape is now (-1, 16, 3, 3)
            ])

            self.flat_size = 16 * 3 * 3

        else:
            self.conv_layers = nn.ModuleList([
                Conv2d3x3(in_channels=num_inputs,
                          out_channels=16, downsample=True),
                # shape is now (-1, 16, 42, 42)
                Conv2d3x3(in_channels=16, out_channels=16, downsample=True),
                # shape is now (-1, 16, 21, 21)
                Conv2d3x3(in_channels=16, out_channels=32, downsample=True),
                # shape is now (-1, 16, 11, 11)
                Conv2d3x3(in_channels=32, out_channels=32, downsample=True),
                # shape is now (-1, 32, 6, 6)
                Conv2d3x3(in_channels=32, out_channels=32, downsample=True),
                # shape is now (-1, 32, 3, 3)
            ])

            self.flat_size = 32 * 3 * 3

        def init_(m): return init(m, nn.init.orthogonal_, lambda x: nn.init.
                                  constant_(x, 0), nn.init.calculate_gain('relu'))

        self.linear_layer = nn.Sequential(
            init_(nn.Linear(self.flat_size, hidden_size)), nn.ReLU())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        self.nonlinearity = nn.ReLU()
        self.raw_state_emb = None
        self.hidden_size = hidden_size

        self.train()

    def forward(self, inputs, rnn_hxs, masks, action_pooled=None,
                add_input=None):
        if inputs.dtype == torch.uint8:
            inputs = inputs.float()
        x = inputs
        for conv in self.conv_layers:
            x = conv(x)
            x = self.nonlinearity(x)
        x = x.view(-1, self.flat_size)
        x = self.linear_layer(x)

        self.raw_state_emb = x.clone()

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, add_input_dim,
                 action_output_size=32,
                 recurrent=False, hidden_size=64,
                 use_batch_norm=False, args=None):
        super().__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        def init_(m): return init(m, nn.init.orthogonal_, lambda x: nn.init.
                                  constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        # Action embedder
        self.actor_action = nn.Sequential(
            init_(nn.Linear(hidden_size + action_output_size, hidden_size)), nn.Tanh())
        # Action embedder - Critic
        self.critic_action = nn.Sequential(
            init_(nn.Linear(hidden_size + action_output_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(
            nn.Linear(hidden_size + action_output_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks, add_input):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs

class SoftModuleCNN(NNBase):
    def __init__(self, num_inputs, add_input_dim,
                 num_tasks,
                 action_output_size=32,
                 recurrent=False, hidden_size=64,
                 use_batch_norm=False, args=None):
        super().__init__(recurrent, hidden_size, hidden_size)
        self.args = args
        self.activation_func = F.relu

        self.conv_layers = nn.ModuleList([
            Conv2d3x3(in_channels=num_inputs, out_channels=16, downsample=True),
            # shape is now (-1, 16, 42, 42)
            Conv2d3x3(in_channels=16, out_channels=16, downsample=True),
            # shape is now (-1, 16, 21, 21)
            Conv2d3x3(in_channels=16, out_channels=32, downsample=True),
            # shape is now (-1, 16, 11, 11)
            Conv2d3x3(in_channels=32, out_channels=32, downsample=True),
            # shape is now (-1, 32, 6, 6)
            Conv2d3x3(in_channels=32, out_channels=32, downsample=True),
            # shape is now (-1, 32, 3, 3)
        ])

        self.flat_size = 32 * 3 * 3

        def init_(m): return init(m, nn.init.orthogonal_, lambda x: nn.init.
                                  constant_(x, 0), nn.init.calculate_gain('relu'))

        self.linear_layer = init_(nn.Linear(self.flat_size, hidden_size))

        # Policy network modules

        self.layer_modules = []
        self.num_layers = 2
        self.num_modules = 2

        module_input_size = hidden_size

        for i in range(self.num_layers):
            layer_module = []
            for j in range(self.num_modules):
                module = nn.Sequential(
                    nn.BatchNorm1d(module_input_shape),
                    init_(nn.Linear(module_input_shape, hidden_size)),
                    nn.ReLU()
                )

                layer_module.append(module)
                self.__setattr__("module_{}_{}".format(i,j), module)

            module_input_shape = hidden_size
            self.layer_modules.append(layer_module)

        # self.linear_layer = init_(nn.Linear(self.flat_size, hidden_size))
        self.last_linear_layer = init_(nn.Linear(hidden_size, hidden_size))

        # Routing network
        # num_tasks -> hidden_size
        self.routing_linear_layer = nn.Linear(num_tasks, hidden_size)
        
        gating_input_shape = hidden_size

        # routing_layers = []
        # for i in range(self.num_modules - 1):
        #     routing_layers.append(nn.Linear(gating_input_shape, (self.num_modules * self.num_modules)))
        # self.routing_layers = nn.ModuleList(*routing_layers)

        self.gating_fcs = []
        num_gating_layers = 2
        
        for i in range(num_gating_layers):
            gating_fc = nn.Linear(gating_input_shape, hidden_size)
            self.gating_fcs.append(gating_fc)
            self.__setattr__("gating_fc_{}".format(i), gating_fc)
            gating_input_shape = hidden_size

        self.gating_weight_fcs = []
        self.gating_weight_cond_fcs = []

        self.gating_weight_fc_0 = nn.Linear(gating_input_shape,
                    num_modules * num_modules )
        # last_init_func( self.gating_weight_fc_0)
        # self.gating_weight_fcs.append(self.gating_weight_fc_0)

        for layer_idx in range(num_layers-2):
            gating_weight_cond_fc = nn.Linear((layer_idx+1) * num_modules * num_modules,
                                              gating_input_shape)
            # module_hidden_init_func(gating_weight_cond_fc)
            self.__setattr__("gating_weight_cond_fc_{}".format(layer_idx+1),
                             gating_weight_cond_fc)
            self.gating_weight_cond_fcs.append(gating_weight_cond_fc)

            gating_weight_fc = nn.Linear(gating_input_shape,
                                         num_modules * num_modules)
            # last_init_func(gating_weight_fc)
            self.__setattr__("gating_weight_fc_{}".format(layer_idx+1),
                             gating_weight_fc)
            self.gating_weight_fcs.append(gating_weight_fc)

        self.gating_weight_cond_last = nn.Linear((num_layers-1) * \
                                                 num_modules * num_modules,
                                                 gating_input_shape)
        # module_hidden_init_func(self.gating_weight_cond_last)

        self.gating_weight_last = nn.Linear(gating_input_shape, num_modules)
        # last_init_func( self.gating_weight_last )

        self.pre_softmax = pre_softmax
        self.cond_ob = cond_ob

        # Critic network
        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        self.nonlinearity = nn.ReLU()
        self.raw_state_emb = None
        self.hidden_size = hidden_size

        self.train()

    def forward(self, inputs, rnn_hxs, masks, task_encoding, action_pooled=None,
                add_input=None):
        if inputs.dtype == torch.uint8:
            inputs = inputs.float()
        x = inputs
        for conv in self.conv_layers:
            x = conv(x)
            x = self.nonlinearity(x)
        x = x.view(-1, self.flat_size)
        out = self.linear_layer(x)

        self.raw_state_emb = out.clone()

        # Soft modularization start

        embedding = self.routing_linear_layer(task_encoding)
        embedding = embedding * out

        out = self.activation_func(out)

        if len(self.gating_fcs) > 0:
            embedding = self.activation_func(embedding)
            for fc in self.gating_fcs[:-1]:
                embedding = fc(embedding)
                embedding = self.activation_func(embedding)
            embedding = self.gating_fcs[-1](embedding)

        base_shape = embedding.shape[:-1]

        weights = []
        flatten_weights = []

        raw_weight = self.gating_weight_fc_0(self.activation_func(embedding))

        weight_shape = base_shape + torch.Size([self.num_modules,
                                                self.num_modules])
        flatten_shape = base_shape + torch.Size([self.num_modules * \
                                                self.num_modules])

        raw_weight = raw_weight.view(weight_shape)

        softmax_weight = F.softmax(raw_weight, dim=-1)
        weights.append(softmax_weight)
        if self.pre_softmax:
            flatten_weights.append(raw_weight.view(flatten_shape))
        else:
            flatten_weights.append(softmax_weight.view(flatten_shape))

        for gating_weight_fc, gating_weight_cond_fc in zip(self.gating_weight_fcs, self.gating_weight_cond_fcs):
            cond = torch.cat(flatten_weights, dim=-1)
            if self.pre_softmax:
                cond = self.activation_func(cond)
            cond = gating_weight_cond_fc(cond)
            cond = cond * embedding
            cond = self.activation_func(cond)

            raw_weight = gating_weight_fc(cond)
            raw_weight = raw_weight.view(weight_shape)
            softmax_weight = F.softmax(raw_weight, dim=-1)
            weights.append(softmax_weight)
            if self.pre_softmax:
                flatten_weights.append(raw_weight.view(flatten_shape))
            else:
                flatten_weights.append(softmax_weight.view(flatten_shape))

        cond = torch.cat(flatten_weights, dim=-1)
        if self.pre_softmax:
            cond = self.activation_func(cond)
        cond = self.gating_weight_cond_last(cond)
        cond = cond * embedding
        cond = self.activation_func(cond)

        raw_last_weight = self.gating_weight_last(cond)
        last_weight = F.softmax(raw_last_weight, dim = -1)

        module_outputs = [(layer_module(out)).unsqueeze(-2) \
                for layer_module in self.layer_modules[0]]

        module_outputs = torch.cat(module_outputs, dim = -2 )

        for i in range(self.num_layers - 1):
            new_module_outputs = []
            for j, layer_module in enumerate(self.layer_modules[i + 1]):
                module_input = (module_outputs * \
                    weights[i][..., j, :].unsqueeze(-1)).sum(dim=-2)

                module_input = self.activation_func(module_input)
                new_module_outputs.append((
                        layer_module(module_input)
                ).unsqueeze(-2))

            module_outputs = torch.cat(new_module_outputs, dim = -2)

        out = (module_outputs * last_weight.unsqueeze(-1)).sum(-2)
        out = self.activation_func(out)
        out = self.last_linear_layer(out)

        # if self.is_recurrent:
        #     out, rnn_hxs = self._forward_gru(out, rnn_hxs, masks)

        return self.critic_linear(out), out, rnn_hxs
