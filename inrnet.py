import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F

# import torch_geometric

import math
import numpy as np

from functools import partial

class IdentityMapping(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim

    @property
    def flops(self):
        return 0

    @property
    def out_dim(self):
        return self.in_dim

    def forward(self, X):
        return X

class PositionalEncoding(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, num_frequencies=-1, sidelength=None, use_nyquist=True):
        super().__init__()

        self.in_features = in_features

        self.num_frequencies = num_frequencies
        if self.num_frequencies < 0:
            if self.in_features == 3:
                self.num_frequencies = 10
            elif self.in_features == 2:
                assert sidelength is not None
                if isinstance(sidelength, int):
                    sidelength = (sidelength, sidelength)
                self.num_frequencies = 4
                if use_nyquist:
                    self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
            elif self.in_features == 1:
                assert sidelength is not None
                self.num_frequencies = 4
                if use_nyquist:
                    self.num_frequencies = self.get_num_frequencies_nyquist(sidelength)

    @property
    def out_dim(self):
        return self.in_features + 2 * self.in_features * self.num_frequencies

    @property
    def flops(self):
        return self.in_features + (2 * self.in_features * self.num_frequencies) * 2

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)

class RBFLayer(nn.Module):
    '''Transforms incoming data using a given radial basis function.
        - Input: (1, N, in_features) where N is an arbitrary batch size
        - Output: (1, N, out_features) where N is an arbitrary batch size'''

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigmas = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

        self.freq = nn.Parameter(np.pi * torch.ones((1, self.out_features)))

    @property
    def out_dim(self):
        return self.out_features

    @property
    def flops(self):
        raise NotImplementedError

    def reset_parameters(self):
        nn.init.uniform_(self.centres, -1, 1)
        nn.init.constant_(self.sigmas, 10)

    def forward(self, input):
        input = input[0, ...]
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1) * self.sigmas.unsqueeze(0)
        return self.gaussian(distances).unsqueeze(0)

    def gaussian(self, alpha):
        phi = torch.exp(-1 * alpha.pow(2))
        return phi

class FourierFeatMapping(nn.Module):
    def __init__(self, in_dim, map_scale=16, map_size=1024, tunable=False):
        super().__init__()

        B = torch.normal(0., map_scale, size=(map_size//2, in_dim))

        if tunable:
            self.B = nn.Parameter(B, requires_grad=True)
        else:
            self.register_buffer('B', B)

    @property
    def out_dim(self):
        return 2 * self.B.shape[0]

    @property
    def flops(self):
        return self.B.shape[0] * self.B.shape[1]

    def forward(self, x):
        x_proj = torch.matmul(x, self.B.T)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class RandomFourierMapping(nn.Module):
    '''
    Generate Random Fourier Features (RFF) corresponding to the kernel:
        k(x, y) = k_a(x_a, y_a)*k_b(x_b, y_b)
    where 
        k_a(x_a, y_a) = exp(-\norm(x_a-y_a)/gamma_1),
        k_b(x_b, y_b) = <x_b, y_b>^gamma_2.
    '''
    def __init__(self, in_dim, kernel='exp', map_size=1024, tunable=False, **kwargs):
        super().__init__()

        if kernel == 'exp1':
            length_scale = kwargs.get('length_scale', 64)
            W = exp_sample(length_scale, map_size)
        elif kernel == 'exp2':
            length_scale = kwargs.get('length_scale', 64)
            W = exp2_sample(length_scale, map_size)
        elif kernel == 'matern':
            length_scale = kwargs.get('length_scale', 64)
            matern_order = kwargs.get('matern_order', 0.5)
            W = matern_sample(length_scale, matern_order, map_size)
        elif kernel == 'gamma_exp':
            length_scale = kwargs.get('length_scale', 64)
            gamma_order = kwargs.get('gamma_order', 1)
            W = gamma_exp2_sample(length_scale, gamma_order, map_size)
        elif kernel == 'rq':
            length_scale = kwargs.get('length_scale', 64)
            rq_order = kwargs.get('rq_order', 4)
            W = rq_sample(length_scale, rq_order, map_size)
        elif args.kernel == 'poly':
            poly_order = kwargs.get('poly_order', 4)
            W = poly_sample(poly_order, map_size)
        else:
            raise NotImplementedError()
        b = np.random.uniform(0, np.pi * 2, map_size)

        if tunable:
            self.W = nn.Parameter(W, requires_grad=True)
            self.b = nn.Parameter(b, requires_grad=True)
        else:
            self.register_buffer('W', W)
            self.register_buffer('b', b)

    @property
    def out_dim(self):
        return self.W.shape[0]

    def forward(self, x):
        Z = torch.cos(x @ self.W.T + self.b)
        return Z

### Taken from official SIREN repo
class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)

class LipLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super(LipLinear, self).__init__(in_features, out_features)
        # super(LipLinear, self).__init__(in_features, out_features, bias, device, dtype)
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self) -> None:
        super().reset_parameters()
        # self.c = nn.Parameter(torch.ones(1) * 1.4)
        self.c = nn.Parameter(torch.ones(1), requires_grad=False)
        # self.c = nn.Parameter(torch.linalg.norm(self.weight, ord=float('inf')))
        # self.c = nn.Parameter(torch.max(self.weight.abs().sum(axis=1)))
    def forward(self, input):
        absrowsum = self.weight.abs().sum(axis=1)
        ww = F.softplus(self.c) / absrowsum
        scale = torch.minimum(torch.ones_like(ww, device=ww.device, requires_grad=True), ww)
        # print(scale)
        return F.linear(input, self.weight * scale[:, None], self.bias)

        

class FCBlock(nn.Module):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.num_hidden_layers = num_hidden_layers

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {
            'sine':(Sine(), sine_init, first_layer_sine_init),
            # 'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
            'relu':(nn.ReLU(inplace=True), init_weights_relu, None),
            'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
            'tanh':(nn.Tanh(), init_weights_xavier, None),
            'selu':(nn.SELU(inplace=True), init_weights_selu, None),
            'softplus':(nn.Softplus(), init_weights_normal, None),
            'elu':(nn.ELU(inplace=True), init_weights_elu, None)
        }

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        myLinear = nn.Linear
        # myLinear = LipLinear

        self.net = []
        self.net.append(nn.Sequential(
            myLinear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(
                myLinear(hidden_features, hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(nn.Sequential(myLinear(hidden_features, out_features)))
        else:
            self.net.append(nn.Sequential(
                myLinear(hidden_features, out_features), nl
            ))

        self.net = nn.Sequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)
        
        # print(self.named_parameters())
        # for name, param in self.named_parameters():
        #     print(param)

    @property
    def flops(self):
        # (in_dim + 1) * out_dim: plus one for bias
        return (self.in_features+1) * self.hidden_features + \
            self.num_hidden_layers * (self.hidden_features+1) * self.hidden_features + \
            (self.hidden_features+1) * self.out_features

    def forward(self, coords):
        output = self.net(coords)
        return output

import diff_operators
class INRNet(nn.Module):
    '''A canonical representation network.'''

    def __init__(self, pos_emb='ffm', out_features=1, in_features=2, num_hidden_layers=3, hidden_features=256, nonlinearity='relu', **kwargs):
        super().__init__()
        self.pos_embed = pos_emb

        if self.pos_embed == 'Id':
            self.map = IdentityMapping(in_features)
        elif self.pos_embed == 'rbf':
            self.map = RBFLayer(in_features=in_features,out_features=args.rbf_centers)
        elif self.pos_embed == 'pe':
            self.map = PositionalEncoding(in_features=in_features,
                num_frequencies=args.num_freqs,
                sidelength=kwargs.get('sidelength', None),
                use_nyquist=args.use_nyquist
            )
        elif self.pos_embed == 'ffm':
            self.map = FourierFeatMapping(in_features,
                map_scale=128, # args.ffm_map_scale,
                map_size=4096 # args.ffm_map_size,
            )
        elif self.pos_embed == 'gffm':
            self.map = RandomFourierMapping(in_features,
                length_scale = args.length_scale,
                matern_order = args.matern_order,
                gamma_order = args.gamma_order,
                rq_order = args.rq_order,
                poly_order = args.poly_order
            )
        else:
            raise ValueError(f'Unknown type of positional embedding: {self.pos_embed}')
        in_features = self.map.out_dim

        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
            hidden_features=hidden_features, outermost_linear=True, nonlinearity=nonlinearity)
        print(self)

    @property
    def flops(self):
        return self.map.flops + self.net.flops

    def forward(self, x):
        # various input processing methods for different applications
        # offset = ((torch.rand(x['coords'].shape, device=x['coords'].device) - 0.5) * 2) / 256
        # x['coords'] += offset
        xx = x['coords'].clone().requires_grad_(True)
        coords = self.map(xx)
        c = 1
        # li = []
        # for name, param in self.net.named_parameters():
        #     # print(name)
        #     if '.c' in name:
        #         c = c * F.softplus(param)
        #         li.append(param.item())
        # print(li, c)
        # output = self.net(xx)
        # output = F.sigmoid(ouytpu)
        output = self.net(coords)
        # output = F.sigmoid(output)
        # print(output.dtype)
        # grad = diff_operators.gradient(output, xx)
        # grad = diff_operators.all_2(output, xx)
        # print(grad.shape)
        return {'model_out': output, 'model_in': xx, 'c': c}
        # return {'model_out': output, 'model_in': xx, 'c': c, 'grad': grad}

########################
# Initialization methods

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # grab from upstream pytorch branch and paste here for now
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def init_weights_trunc_normal(m):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            fan_in = m.weight.size(1)
            fan_out = m.weight.size(0)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            mean = 0.
            # initialize with the same behavior as tf.truncated_normal
            # "The generated values follow a normal distribution with specified mean and
            # standard deviation, except that values whose magnitude is more than 2
            # standard deviations from the mean are dropped and re-picked."
            _no_grad_trunc_normal_(m.weight, mean, std, -2 * std, 2 * std)


def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_relu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.uniform_(-stdv, stdv)

def init_weights_selu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)

class INRDict(nn.Module):
    def __init__(self, dict_size, args, in_dim=3, out_dim=1):
        super().__init__()

        self.in_dim, self.out_dim = in_dim, out_dim
        self.code_dim = args.code_dim

        self.siren = INRNet(args, out_features=self.code_dim, in_features=in_dim)

        ## TO-DO restrict different norm of the code
        # self.codebook = nn.Embedding(dict_size, out_dim*args.code_dim, max_norm=1., norm_type=1.0)
        self.codebook = nn.Embedding(dict_size, out_dim*args.code_dim)

    def code_parameters(self):
        for name, param in self.named_parameters():
            if name.startswith('codebook.'):
                yield param

    def freeze_dict(self):
        for name, param in self.named_parameters():
            if name.startswith('siren.'):
                param.requires_grad_(False)

    def load_dict_from_checkpoint(self, ckpt):
        param_dict = self.state_dict()
        for name, param in ckpt.items():
            if name.startswith('siren.'):
                param_dict[name] = param
        self.load_state_dict(param_dict)

    def get_dict(self, coords):
        return self.siren(coords) # [N_coords, N_basis]

    def forward(self, coords, model_ids):
        N = coords.shape[0]
        basis = self.siren(coords) # [N_coords, N_basis]
        code = self.codebook(model_ids).reshape(-1, self.code_dim, self.out_dim) # [N_imgs, N_basis, out_dim]

        y = torch.matmul(basis[None, ...], code) # [N_imgs, N_coords, out_dim]
        y = torch.sigmoid(y)

        return y, code

class INRMoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, args, gate_module=None, noise_module=None, in_dim=3, out_dim=1, bias=True):
        super().__init__()
        self.noisy_gating = (noise_module is not None)
        self.num_experts = args.num_experts
        self.output_size = out_dim
        self.input_size = in_dim
        self.hidden_size = args.hidden_dim
        self.k = args.num_topk
        self.bias = bias

        assert(self.k <= self.num_experts)
        assert(gate_module is not None)

        # instantiate experts
        self.experts = INRNet(args, out_features=self.num_experts*out_dim, in_features=in_dim)
        self.combiner = MoECombiner()

        # instantiate gate network
        if bias:
            self.gate_generator = gate_module(output_size=self.num_experts+out_dim)
        else:
            self.gate_generator = gate_module(output_size=self.num_experts)

        # self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True).to(self.device)
        if self.noisy_gating and noise_module is not None:
            self.noise_generator = noise_module(output_size=self.num_experts)
            self.softplus = nn.Softplus()

        self.softmax = nn.Softmax(1)

    @property
    def flops(self):
        flops = self.gate_generator.flops

        unshared_layer_flops = (self.num_experts * self.output_size + 1) * self.hidden_size
        saved_params = int(float(unshared_layer_flops) * (self.num_experts - self.k) / self.num_experts)
        flops += self.experts.flops - saved_params

        flops += self.k * self.output_size

        return flops

    def code_parameters(self):
        for name, param in self.named_parameters():
            if name.startswith('gate_generator.'):
                yield param
            elif name.startswith('noise_generator.'):
                yield param

    def freeze_dict(self):
        for name, param in self.named_parameters():
            if name.startswith('experts.'):
                param.requires_grad_(False)

    def load_dict_from_checkpoint(self, ckpt):
        param_dict = self.state_dict()
        for name, param in ckpt.items():
            if name.startswith('experts.'):
                param_dict[name] = param
        self.load_state_dict(param_dict)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)


    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=top_values_flat.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = torch.distributions.normal.Normal(
            loc=torch.tensor([0.0], device=clean_values.device),
            scale=torch.tensor([1.0], device=clean_values.device)
        )
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob


    def noisy_top_k_gating(self, model_input, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        raw_logits = self.gate_generator(model_input) # [bs, num_exps+dim)]
        bias = 0.
        if self.bias:
            clean_logits = raw_logits[:, :-self.output_size] # [bs, num_exps]
            bias = raw_logits[:, -self.output_size:] # [bs, dim]
        else:
            clean_logits = raw_logits # [bs, num_exps]

        if self.noisy_gating and self.training:
            # raw_noise_stddev = x @ self.w_noise
            raw_noise_stddev = self.noise_generator(model_input)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = torch.abs(logits).topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        # top_k_gates = self.softmax(top_k_logits)
        top_k_gates = torch.gather(logits, 1, top_k_indices)

        zeros = torch.zeros_like(logits, device=logits.device, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates) # clear entries out of activated gates

        if self.noisy_gating and self.k < self.num_experts and self.training:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, bias, load

    def get_dict(self, coords):
        expert_outputs = self.experts(coords) # [N_coords, num_exps x dim_out]
        N_coords = expert_outputs.shape[0]
        expert_outputs = expert_outputs.reshape(-1, self.num_experts, self.output_size) # [N_coords, num_exps, dim_out]
        expert_outputs = expert_outputs.permute(1, 0, 2) # [num_exps, N_coords, dim_out]
        return expert_outputs

    def forward(self, model_input, topk_sparse=True):
        """Args:
        x: tensor shape [batch_size, input_size]
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """

        expert_outputs = self.experts(model_input['coords']) # [N_coords, num_exps x dim_out]
        N_coords = expert_outputs.shape[0]
        expert_outputs = expert_outputs.reshape(-1, self.num_experts, self.output_size) # [N_coords, num_exps, dim_out]
        expert_outputs = expert_outputs.permute(1, 0, 2) # [num_exps, N_coords, dim_out]
        expert_outputs = expert_outputs.reshape(self.num_experts, -1) # [num_exps, N_coords x dim_out]

        if topk_sparse:
            gates, bias, load = self.noisy_top_k_gating(model_input)
            # calculate importance loss
            importance = gates.sum(0)
            gates = gates.reshape(-1, self.num_experts) # [N_imgs, num_exps]
            N_imgs = gates.shape[0]

            y = self.combiner(expert_outputs, gates) # [N_imgs, N_coords x dim_out]
            y = y.reshape(N_imgs, N_coords, self.output_size) # [N_imgs, N_coords, dim_out]
            y = y + bias[:, None, :] # [N_imgs, N_coords, dim_out]

            return {'preds': y, 'gates': gates, 'load': load, 'importance': importance}
        else:
            raw_logits = self.gate_generator(model_input) # [bs, num_exps+dim]
            bias = 0.
            if self.bias:
                gates = raw_logits[:, :-self.output_size] # [N_imgs, num_exps]
                bias = raw_logits[:, -self.output_size:] # [N_imgs, dim]
            else:
                gates = raw_logits # [N_imgs, num_exps]
            importance = gates.sum(0)
            N_imgs = gates.shape[0]

            y = torch.matmul(gates, expert_outputs) # [N_imgs, N_coords x dim_out]
            y = y.reshape(N_imgs, N_coords, self.output_size) # [N_imgs, N_coords, dim_out]
            y = y + bias[:, None, :] # [N_imgs, N_coords, dim_out]

            return {'preds': y, 'gates': gates, 'importance': importance}

class MoECombiner(nn.Module):
# class MoECombiner(torch_geometric.nn.conv.MessagePassing):
    
    def __init__(self):
        super().__init__(aggr='add')

    def message(self, x_j, x_i, edge_weights):
        return x_j * edge_weights

    def forward(self, expert_outputs, gates):
        expert_indices = torch.nonzero(gates)
        edge_index = torch.stack([expert_indices[:, 1], expert_indices[:, 0]], 0) # [2, N_edges]
        edge_weights = gates[expert_indices[:, 0], expert_indices[:, 1], None] # [N_edges]
        num_experts, num_images = expert_outputs.shape[0], gates.shape[0]

        out = self.propagate(edge_index, x=(expert_outputs, None), edge_weights=edge_weights, size=(num_experts, num_images))
        return out

def cv_squared_loss(x, eps=1e-10):
    """The squared coefficient of variation of a sample.
    Useful as a loss to encourage a positive distribution to be more uniform.
    Epsilons added for numerical stability.
    Returns 0 for an empty Tensor.
    Args:
    x: a `Tensor`.
    Returns:
    a `Scalar`.
    """
    # if only num_experts = 1
    if x.shape[0] == 1:
        return torch.Tensor([0], device=x.device)
    return x.float().var() / (x.float().mean()**2 + eps)

class SimpleConvImgEncoder(nn.Module):

    def __init__(self, input_size, hidden_dim, num_layers, output_size):
        super().__init__()
        convs = [nn.Conv2d(input_size, hidden_dim, 5, 1, 2), nn.ReLU()]
        for i in range(num_layers-1):
            convs.append(nn.Conv2d(hidden_dim, hidden_dim, 5, 1, 2))
            convs.append(nn.ReLU())
        self.convs = nn.Sequential(*convs)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, model_input):
        x = model_input['imgs'].permute(0, 3, 1, 2)
        x = self.convs(x)
        B, C = x.shape[0], x.shape[1]
        x = torch.sum(x.reshape(B, C, -1), -1)
        x = self.fc(x)
        return x

class ResConvImgEncoder(nn.Module):

    def __init__(self, input_size, output_size, image_resolution):
        super().__init__()
        self.convs = ConvImgEncoder(input_size, image_resolution)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(256, output_size)

    def forward(self, model_input):
        x = model_input['imgs'].permute(0, 3, 1, 2) # [B, C, H, W]
        x = self.convs(x) # [B, 256]
        x = self.relu(x) # [B, 256]
        x = self.fc(x) # [B, out_dim]
        return x

class LinearImgEncoder(nn.Module):
    
    def __init__(self, input_size, output_size):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(input_size, output_size), requires_grad=True)

    @property
    def flops(self):
        return self.w.shape[0] * self.w.shape[1]

    def forward(self, model_input):
        x = model_input['imgs'].permute(0, 3, 1, 2)
        x = x.reshape(x.shape[0], -1)
        return x @ self.w

# class CodebookImgEncoder(nn.Module):
    
#     def __init__(self, num_images, output_size):
#         super().__init__()
#         self.codebook = nn.Embedding(num_images, output_size)

#     def forward(self, model_input):
#         img_ids = model_input['img_ids']
#         return self.codebook(img_ids) # [N_imgs, out_dim]

class CodebookImgEncoder(nn.Module):
    
    def __init__(self, num_images, output_size, max_norm=None, norm_type=2.):
        super().__init__()
        self.codebook = nn.Embedding(num_images, output_size, max_norm=max_norm, norm_type=norm_type)

    @property
    def flops(self):
        return 0

    def forward(self, model_input):
        sample_ids = model_input['img_ids']
        return self.codebook(sample_ids) # [N_samples, out_dim]

########################
# Encoder modules

class SetEncoder(nn.Module):
    def __init__(self, in_features, out_features,
                 num_hidden_layers, hidden_features, nonlinearity='relu'):
        super().__init__()

        assert nonlinearity in ['relu', 'sine'], 'Unknown nonlinearity type'

        if nonlinearity == 'relu':
            nl = nn.ReLU(inplace=True)
            weight_init = init_weights_normal
        elif nonlinearity == 'sine':
            nl = Sine()
            weight_init = sine_init

        self.net = [nn.Linear(in_features, hidden_features), nl]
        self.net.extend([nn.Sequential(nn.Linear(hidden_features, hidden_features), nl)
                         for _ in range(num_hidden_layers)])
        self.net.extend([nn.Linear(hidden_features, out_features), nl])
        self.net = nn.Sequential(*self.net)

        self.net.apply(weight_init)

    def forward(self, context_x, context_y, ctxt_mask=None, **kwargs):
        input = torch.cat((context_x, context_y), dim=-1)
        embeddings = self.net(input)

        if ctxt_mask is not None:
            embeddings = embeddings * ctxt_mask
            embedding = embeddings.mean(dim=-2) * (embeddings.shape[-2] / torch.sum(ctxt_mask, dim=-2))
            return embedding
        return embeddings.mean(dim=-2)


class ConvImgEncoder(nn.Module):
    def __init__(self, channel, image_resolution):
        super().__init__()

        # conv_theta is input convolution
        self.conv_theta = nn.Conv2d(channel, 128, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

        self.cnn = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            Conv2dResBlock(256, 256),
            Conv2dResBlock(256, 256),
            Conv2dResBlock(256, 256),
            Conv2dResBlock(256, 256),
            nn.Conv2d(256, 256, 1, 1, 0)
        )

        self.relu_2 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(image_resolution*image_resolution, 1)

        self.image_resolution = image_resolution

    def forward(self, I):
        o = self.relu(self.conv_theta(I))
        o = self.cnn(o)

        o = self.fc(self.relu_2(o).view(o.shape[0], 256, -1)).squeeze(-1)
        return o


class PartialConvImgEncoder(nn.Module):
    '''Adapted from https://github.com/NVIDIA/partialconv/blob/master/models/partialconv2d.py
    '''
    def __init__(self, channel, image_resolution):
        super().__init__()

        self.conv1 = PartialConv2d(channel, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = BasicBlock(256, 256)
        self.layer2 = BasicBlock(256, 256)
        self.layer3 = BasicBlock(256, 256)
        self.layer4 = BasicBlock(256, 256)

        self.image_resolution = image_resolution
        self.channel = channel

        self.relu_2 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(1024, 1)

        for m in self.modules():
            if isinstance(m, PartialConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, I):
        M_c = I.clone().detach()
        M_c = M_c > 0.
        M_c = M_c[:,0,...]
        M_c = M_c.unsqueeze(1)
        M_c = M_c.float()

        x = self.conv1(I, M_c)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        o = self.fc(x.view(x.shape[0], 256, -1)).squeeze(-1)

        return o


class Conv2dResBlock(nn.Module):
    '''Aadapted from https://github.com/makora9143/pytorch-convcnp/blob/master/convcnp/modules/resblock.py'''
    def __init__(self, in_channel, out_channel=128):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 5, 1, 2),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            nn.ReLU()
        )

        self.final_relu = nn.ReLU()

    def forward(self, x):
        shortcut = x
        output = self.convs(x)
        output = self.final_relu(output + shortcut)
        return output

class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2], input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return PartialConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
