# %%
import os
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional,Any
import flax.linen as nn
from functools import partial

import logging
logging.getLogger("jax").setLevel(logging.ERROR)

np.random.seed(42)


class BNN_sample:
    def __init__(self, layer_size, batch_size):
        self.size = layer_size
        self.batch_size = batch_size

    def hyper_initial(self, sigma_w, sigma_b, key):
        self.params = []

        key, subkey = jax.random.split(key)
        w1 = jax.random.normal(key, (self.batch_size, self.size[0], self.size[1])) * sigma_w
        b1 = jax.random.normal(subkey, (self.batch_size, self.size[1])) * sigma_b
        self.params.append((w1, b1))

        key, subkey = jax.random.split(key)
        w2 = jax.random.normal(key, (self.batch_size, self.size[1], self.size[2])) / jnp.sqrt(self.size[1])
        b2 = jnp.zeros(shape=(self.batch_size, self.size[2]))
        self.params.append((w2, b2))

        return self.params
    

class DeepMLP(nn.Module):
    hidden_size: int
    output_size: int
    hidden_layers: int
    activation: callable

    def setup(self):
        self.dense_layers = [nn.Dense(self.hidden_size) for _ in range(self.hidden_layers)]
        self.output_layer = nn.Dense(self.output_size)

    def __call__(self, x):
        for i in range(self.hidden_layers):
            x = self.activation(self.dense_layers[i](x))
        x = self.output_layer(x)
        return x


@dataclass
class DatasetConfig:
    x_min: float = -1.0
    x_max: float = 1.0
    x_num: int = 256
    dtype: Any = jnp.float32 
    basis_fn_dim: int = 64
    D = 0.1
    v = 0.1
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sigma_list: Any = jnp.array([8, 10, 12, 14, 18, 20])

    @property
    def x(self):
        return jnp.linspace(self.x_min, self.x_max, self.x_num, dtype=self.dtype).reshape((-1, 1))
    
    @property
    def figure_dir(self):
        return os.path.join(self.base_dir, "figure/all_mixture_medium_high_freq")
    
    @property
    def data_dir(self):
        return os.path.join(self.base_dir, "data/all_mixture_medium_high_freq")
    
    @property
    def checkpoint_dir(self):
        return os.path.join(self.base_dir, "checkpoints/all_mixture_medium_high_freq")


data = DatasetConfig()


@jax.jit
def batch_compute_manual_derivatives_xt(x_inputs, params, activation_flag: int = 0):
    B = params[0][0].shape[0]
    N = x_inputs.shape[0]

    W1, b1 = params[0]
    W2, b2 = params[1]
    H = W1.shape[-1]

    a = 1.0
    W1_x = W1[:, 0, :]
    W1_x_2 = W1_x ** 2

    x_input = jnp.tile(jnp.expand_dims(x_inputs, 0), (B, 1, 1))
    z = jnp.matmul(x_input, W1) + jnp.expand_dims(b1, 1)

    def sin_branch(z):
        z2 = z + jnp.pi / 4
        sqrt_2 = jnp.sqrt(2.0)
        h = sqrt_2 * jnp.sin(z2)
        u = (jnp.matmul(h, W2).squeeze(-1) + b2) / a

        phi2 = -jnp.sin(z2)  # second derivative of sin
        d2u_dx2_terms = sqrt_2 * phi2 * W1_x_2[:, None, :]
        d2u_dx2 = jnp.matmul(d2u_dx2_terms, W2).squeeze(-1) / a
        return u, d2u_dx2

    def tanh_branch(z):
        tanh_z = jnp.tanh(z)
        h = tanh_z
        u = (jnp.matmul(h, W2).squeeze(-1) + b2) / a

        phi2 = -2.0 * tanh_z * (1.0 - tanh_z**2)
        d2u_dx2_terms = phi2 * W1_x_2[:, None, :]
        d2u_dx2 = jnp.matmul(d2u_dx2_terms, W2).squeeze(-1) / a
        return u, d2u_dx2

    def sin_plain_branch(z):
        h = jnp.sin(z)
        u = (jnp.matmul(h, W2).squeeze(-1) + b2) / a

        phi2 = -jnp.sin(z)  # second derivative of sin
        d2u_dx2_terms = phi2 * W1_x_2[:, None, :]
        d2u_dx2 = jnp.matmul(d2u_dx2_terms, W2).squeeze(-1) / a
        return u, d2u_dx2

    u, d2u_dx2 = jax.lax.switch(
        activation_flag,
        [sin_branch, tanh_branch, sin_plain_branch],
        z
    )

    return u, d2u_dx2


@jax.jit
def f_func(u, u_xx):
    return data.v * (u**3 - data.D*u_xx)


@dataclass
class BranchNetConfig:
    input_neuron: int = data.x_num
    hidden_layers: int = 5
    hidden_neuron: int = 256
    output_neuron: int = data.basis_fn_dim
    activation_fn: callable = jax.nn.relu


class ResBlock(nn.Module):
    out_dim: int
    activation: callable

    @nn.compact
    def __call__(self, x):
        in_dim = x.shape[-1]
        h = nn.Dense(self.out_dim)(x)
        h = self.activation(h)
        h = nn.Dense(self.out_dim)(h)
        if in_dim != self.out_dim:
            residual = nn.Dense(self.out_dim)(x)
        else:
            residual = x
        return h + residual



class ResMLP(nn.Module):
    hidden_size: int
    output_size: int
    hidden_layers: int
    activation: callable

    @nn.compact
    def __call__(self, x):
        for _ in range(self.hidden_layers):
            x = ResBlock(self.hidden_size, self.activation)(x)
            
        x = nn.Dense(self.output_size)(x)
        return x


class TrunkNet(nn.Module):
    hidden_dim: int
    basis_output_dim: int
    hidden_layers: int
    activation: callable
    sigma: float

    @nn.compact
    def __call__(self, x):
        my_init = nn.initializers.normal(stddev=self.sigma)
        
        sin_x = jnp.sin(nn.Dense(self.hidden_dim // 2, use_bias=False, kernel_init=my_init)(x))
        cos_x = jnp.cos(nn.Dense(self.hidden_dim // 2, use_bias=False, kernel_init=my_init)(x))
        
        x = jnp.concatenate([sin_x, cos_x], axis=-1)

        residual = x

        x = DeepMLP(self.hidden_dim, self.hidden_dim, self.hidden_layers, self.activation)(x)

        x = residual + x

        x = nn.Dense(self.basis_output_dim)(x)

        return x


@dataclass
class DeepONetConfig:
    branch: BranchNetConfig = field(default_factory=BranchNetConfig)


@partial(jax.jit, static_argnames='N')
def sample_interval(key, N, a=data.x_min, b=data.x_max):
    x = jax.random.uniform(key, (N,), minval=a, maxval=b).reshape(-1,1)  # 随机格点
    idx = jnp.arange(N)
    return idx, x
