"""The S4ND layer which act like a Conv2D with global receptive field.

This code comes from the author of S4ND paper (Eric) and 
has been modified by Pin-Jui Ku
"""

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
from einops import rearrange, reduce
#from omegaconf import DictConfig
import opt_einsum as oe

from model.DSSM_modules.s4 import SSKernel
from util.utils import is_list

optimized = True

if optimized:
    contract = oe.contract
else:
    contract = torch.einsum

from model.DSSM_modules.components import LinearActivation, Activation, DropoutNd


def multiple_axis_slice(x, L):
    """
    x: (..., L1, L2, .., Lk)
    L: list of length k [l1, l2, .., lk]
    returns: x[..., :l1, :l2, .., :lk]
    """
    # TODO I don't see a way to do this programmatically in Pytorch without sacrificing speed so...
    assert len(L) > 0
    if len(L) == 1:
        return x[..., :L[0]]
    elif len(L) == 2:
        return x[..., :L[0], :L[1]]
    elif len(L) == 3:
        return x[..., :L[0], :L[1], :L[2]]
    elif len(L) == 4:
        return x[..., :L[0], :L[1], :L[2], :L[3]]
    else: raise NotImplementedError("lol")


class S4ND(nn.Module):
    #requires_length = True

    def __init__(
        self,
        d_model,
        d_state=64,
        l_max=None, # Maximum length of sequence (list or tuple). None for unbounded
        dim=2, # Dimension of data, e.g. 2 for images and 3 for video
        out_channels=None, # Do depthwise-separable or not
        channels=1, # maps 1-dim to C-dim
        bidirectional=True,
        # Arguments for FF
        activation='gelu', # activation in between SS and FF
        ln=False, # Extra normalization
        postact=None, # activation after FF
        initializer=None, # initializer on FF
        weight_norm=False, # weight normalization on FF
        hyper_act=None, # Use a "hypernetwork" multiplication
        dropout=0.0, tie_dropout=False,
        transposed=True, # axis ordering (B, L, D) or (B, D, L)
        verbose=False,
        trank=1, # tensor rank of C projection tensor
        linear=True,
        return_state=True,
        contract_version=0,
        # SSM Kernel arguments
        **kernel_args,
    ):
        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum sequence length, also denoted by L
          if this is not known at model creation, or inconvenient to pass in,
          set l_max=None and length_correction=True
        dropout: standard dropout argument
        transposed: choose backbone axis ordering of (B, L, D) or (B, D, L) [B=batch size, L=sequence length, D=feature dimension]

        Other options are all experimental and should not need to be configured
        """

        super().__init__()

        self.h = d_model
        self.n = d_state
        self.bidirectional = bidirectional
        self.ln = ln
        self.channels = channels
        self.transposed = transposed
        self.linear = linear
        self.return_state = return_state
        self.contract_version = contract_version
        self.out_channels = out_channels
        self.verbose = verbose
        self.kernel_args = kernel_args

        self.D = nn.Parameter(torch.randn(self.channels, self.h)) # TODO if self.out_channels

        self.trank = trank

        if self.out_channels is not None:
            channels *= self.out_channels

            # # Swap channels and out_channels
            # # self.out_channels = channels
            # self.h = self.out_channels
            # # channels *= out_channels
            # self.in_channels = d_model
            # channels *= d_model
            assert self.linear # TODO change name of linear_output

        channels *= self.trank

        if self.bidirectional:
            channels *= 2

        # Check dimensions and kernel sizes
        if dim is None:
            assert is_list(l_max)

        # assert l_max is not None # TODO implement auto-sizing functionality for the kernel
        if l_max is None:
            self.l_max = [None] * dim
        elif isinstance(l_max, int):
            self.l_max = [l_max] * dim
        else:
            assert l_max is None or is_list(l_max)
            self.l_max = l_max

        # SSM Kernel
        self._kernel_channels = channels
        self.kernel = nn.ModuleList([
            SSKernel(self.h, N=self.n, L=L, channels=channels, verbose=verbose, **kernel_args)
            for L in self.l_max
        ])

        if not self.linear:

            self.activation = Activation(activation)
            dropout_fn = partial(DropoutNd, transposed=self.transposed) if tie_dropout else nn.Dropout
            self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()


            # position-wise output transform to mix features
            self.output_linear = LinearActivation(
                self.h*self.channels,
                self.h,
                transposed=self.transposed,
                initializer=initializer,
                activation=postact,
                activate=True,
                weight_norm=weight_norm,
            )

        ## To handle some operations with unspecified number of dims, we're going to define the einsum/einops contractions programmatically

        # Outer product function for the convolution kernel taking arbitary number of dims
        contract_str = ', '.join([f'... {chr(i+97)}' for i in range(len(self.l_max))]) \
            + ' -> ... ' \
            + ' '.join([f'{chr(i+97)}' for i in range(len(self.l_max))])
        # self.nd_outer = oe.contract_expression(
        #     contract_str,
        #     *[(channels*self.trank, self.h, 2*l) for l in l_max]
        # )

        # Slice first half of each length dimension after the iFFT
        # e.g. in 2D the forward pass wants to call:
        #   y = rearrange(y, '... (f0 l1) (f1 l1) -> (f0 f1) ... (l0 l1)', f0=2, f1=2)
        #   y = y[0]
        # self.nd_slice = '... ' + ' '.join([f"(f{i} l{i})" for i in range(len(l_max))]) + ' -> (' + ' '.join([f"f{i}" for i in range(len(l_max))]) + ') ... (' + ' '.join([f"l{i}" for i in range(len(l_max))]) + ')'

        # unflattened L dim by removing last '()'
        # self.nd_slice = '... ' + ' '.join([f"(f{i} l{i})" for i in range(len(l_max))]) + ' -> (' + ' '.join([f"f{i}" for i in range(len(l_max))]) + ') ... ' + ' '.join([f"l{i}" for i in range(len(l_max))])
        # self.nd_slice_args = { f"f{i}": 2 for i in range(len(l_max)) }

    def _reinit(self, dt_min=None, dt_max=None, normalize=False, **kwargs):
        """ Sets time kernel to custom value """
        assert len(self.l_max) == 3
        L = self.l_max[-3]
        # measure = measure or 'fourier'
        dt_min = dt_min or 2./L
        dt_max = dt_max or 2./L
        print(f"S4ND reinit args: {dt_min=} {dt_max=}", kwargs)
        # print("S4ND reinit kernel_args", self.kernel_args)
        kernel_args = {
            **self.kernel_args, **{
                'H': self.h,
                'N': self.n,
                'L': L,
                # 'measure': measure,
                'dt_min': dt_min,
                'dt_max': dt_max,
                # 'deterministic': True,
                'channels': self._kernel_channels,
                **kwargs,
            }
        }
        time_kernel = SSKernel(**kernel_args)
        if normalize:
            with torch.no_grad():
                time_kernel.kernel.C /= (0.5 * time_kernel.kernel.log_dt.exp()[:, None, None])
        self.kernel[-3] = time_kernel


    def forward(self, u, rate=1.0, state=None, **kwargs): # absorbs return_output and transformer src mask
        """
        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as u
        """

        half_precision = False

        # fft can only handle float32
        if u.dtype == torch.float16:
            half_precision = True
            u = u.to(torch.float32)

        assert state is None, f"state not currently supported in S4ND"

        # ensure shape is B, C, L (L can be multi-axis)
        if not self.transposed:
            u = rearrange(u, "b ... h -> b h ...")

        L_input = u.shape[2:]
        # _u = u

        # TODO this is a hack: need more robust way to support resolution logic
        # L_kernel = [l_i if l_k is None else min(l_i, l_k) for l_i, l_k in zip(L_input, self.l_max)]
        # L_kernel = [l_i if l_k is None else l_k for l_i, l_k in zip(L_input, self.l_max)]
        # for l_i, l_k in zip(L_input, L_kernel): assert l_i >= l_k
        L_kernel = [
            l_i if l_k is None else min(l_i, round(l_k / rate)) for l_i, l_k in zip(L_input, self.l_max)
        ]

        # Compute SS Kernel
        # 1 kernel for each axis in L
        k = [kernel(L=l, rate=rate)[0] for kernel, l in zip(self.kernel, L_kernel)]
        # print(k[0], k[0].shape)
        # breakpoint()

        if self.bidirectional: # halves channels
            k = [torch.chunk(_k, 2, dim=-3) for _k in k] # (C H L)
            k = [
                F.pad(k0, (0, l)) + F.pad(k1.flip(-1), (l, 0))
                # for l, (k0, k1) in zip(L_kernel, k) # TODO bug??
                for l, (k0, k1) in zip(L_input, k)
            ]

        # fft can only handle float32
        if u.dtype == torch.float16:
            half_precision = True
            # cast to fp32
            k.dtype = torch.float32

        L_padded = [l_input + l_kernel for l_input, l_kernel in zip(L_input, L_kernel)]
        u_f = torch.fft.rfftn(u, s=tuple([l for l in L_padded])) # (B H L)
        k_f = [torch.fft.fft(_k, n=l) for _k, l in zip(k[:-1], L_padded[:-1])] + [torch.fft.rfft(k[-1], n=L_padded[-1])] # (C H L)

        # Take outer products

        if self.contract_version == 0: # TODO set this automatically if l_max is provided
            k_f = contract('... c h m, ... c h n -> ... c h m n', k_f[0], k_f[1]) # (H L1 L2) # 2D case of next line
            # k_f = self.nd_outer(*k_f)
            # sum over tensor rank
            k_f = reduce(k_f, '(r c) h ... -> c h ...', 'sum', r=self.trank) / self.trank # reduce_mean not available for complex... # TODO does it matter if (r c) or (c r)?
            y_f = contract('bh...,ch...->bch...', u_f, k_f) # k_f.unsqueeze(-4) * u_f.unsqueeze(-3) # (B C H L)

        else:
            contract_str_l = [f'{chr(i+100)}' for i in range(len(L_input))]
            contract_str = 'b ... ' + ' '.join(contract_str_l) + ', ' \
                + ', '.join(['... ' + l for l in contract_str_l]) \
                + ' -> b ... ' \
                + ' '.join(contract_str_l)
            # print(contract_str)
            y_f = contract(contract_str, u_f, *k_f)
            k_f = reduce(y_f, 'b (r c) h ... -> b c h ...', 'sum', r=self.trank) / self.trank # reduce_mean not available for complex... # TODO does it matter if (r c) or (c r)?

        # Contract over channels if not depthwise separable
        if self.out_channels is not None:
            y_f = reduce(y_f, 'b (i c) h ... -> b c i ...', 'sum', i=self.out_channels) # TODO normalization might not be right
            # y_f = reduce(y_f, 'b (i c) h ... -> b i h ...', 'sum', i=self.in_channels) # TODO normalization might not be right
        # y_f = reduce(y_f, 'b (i c) h ... -> b c h ...', 'sum', c=self.channels) # TODO normalization might not be right


        y = torch.fft.irfftn(y_f, s=tuple([l for l in L_padded]))


        # need to cast back to half if used
        if half_precision:
            y = y.to(torch.float16)

        # y = y[..., :self.l_max[0], :self.l_max[1]] # 2D case of next line
        # y = rearrange(y, self.nd_slice, **self.nd_slice_args) # programmatically using einops
        # y = y[0]

        y = multiple_axis_slice(y, L_input)

        # Compute D term in state space equation - essentially a skip connection
        # B, C, H, L (not flat)
        if not self.out_channels:
            y = y + contract('bh...,ch->bch...', u, self.D) # u.unsqueeze(-3) * self.D.unsqueeze(-1)

        # Reshape to flatten channels
        # B, H, L (not flat)
        y = rearrange(y, 'b c h ... -> b (c h) ...')

        if not self.linear:
            y = self.dropout(self.activation(y))

        # ensure output and input shape are the same
        if not self.transposed:
            # B, H, L -> B, H, C
            y = rearrange(y, "b h ... -> b ... h")

        # y = self.norm(y)

        if not self.linear:
            y = self.output_linear(y)

        if self.return_state:
            return y, None
        else: return y

    def default_state(self, *batch_shape, device=None):
        return self._initial_state.repeat(*batch_shape, 1, 1)

    @property
    def d_state(self):
        return self.h * self.n

    @property
    def d_output(self):
        return self.h
        # return self.h if self.out_channels is None else self.out_channels

    @property
    def state_to_tensor(self):
        return lambda state: rearrange('... h n -> ... (h n)', state)


class DownPool2D(nn.Module):
    def __init__(self, in_channels, time_resample_factor):
        super().__init__()
        self.time_resample_factor = time_resample_factor
        out_channels = in_channels * time_resample_factor
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = x[:, :, ::self.time_resample_factor, :]
        return x


class UpPool2D(nn.Module):
    def __init__(self, in_channels, time_resample_factor):
        super().__init__()
        self.time_resample_factor = time_resample_factor
        out_channel = in_channels // time_resample_factor
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channel, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        _, _, freq, time = x.shape
        x = F.interpolate(x, size=(freq*self.time_resample_factor, time), mode='bilinear')
        return x

class FFBlock2D(nn.Module):

    def __init__(self, d_model, ff_bottleneck_expand_factor=2, dropout=0.0):
        """
        Feed-forward block.
        Args:
            d_model: dimension of input
            ff_expand_factor: expansion factor for inverted bottleneck
            dropout: dropout rate
        """
        super().__init__()

        input_linear = nn.Conv2d(d_model, d_model * ff_bottleneck_expand_factor, kernel_size=1)
        act_func = nn.GELU()
        dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        output_linear = nn.Conv2d(d_model * ff_bottleneck_expand_factor, d_model, kernel_size=1)

        self.ff = nn.Sequential(
            input_linear,
            act_func,
            dropout,
            output_linear,
        )

    def forward(self, x):
        return self.ff(x)

class ResidualBlock2D(nn.Module):

    def __init__(
        self,
        d_model,
        layer,
        dropout=0.0,
    ):
        """
        Residual S4 block.
        Args:
            d_model: dimension of the model
            layer: a layer config
            dropout: dropout rate
        """
        super().__init__()

        self.layer = layer
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        """
        Input x is shape (B, d_model, H, W)
        """
        z = x

        z = rearrange(z, "... d h w -> ... h w d")
        z = self.norm(z)
        z = rearrange(z, "...  h w d-> ... d h w")

        z = self.layer(z)
        z = self.dropout(z)
        x = z + x

        return x
