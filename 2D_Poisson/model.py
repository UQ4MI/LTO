import os

import jax
import jax.numpy as jnp

from flax import linen as nn
from flax.training import train_state


from typing import List, Optional, Callable
from dataclasses import dataclass, field, asdict
from functools import partial
import pickle


def func(input_, k, func_type: str):
    u, du, ddu = input_[0], input_[1], input_[2]   # [B,N,1], [B,N,1,D], [B,N,1,D]

    # 根据 func_type 选择方程
    if func_type == "Poisson":
        # 泊松方程： f = -k * Δu
        f = k * (-jnp.sum(ddu, axis=-1))

    elif func_type == "ReactionSin":
        # 反应扩散方程（带 sin(u) 反应项）：
        # f = -k * Δu + k * sin(u)
        f = k * (-jnp.sum(ddu, axis=-1) + jnp.sin(u))

    elif func_type == "Heat":
        # 导热方程（Heat Equation）
        # 示例：f = k*(du[...,0] + 0.01 * (二阶导的部分求和))
        f = k * (du[..., 0] - 0.01 * jnp.sum(ddu[..., [-1, -2]], axis=-1))
        
    else:
        raise ValueError(f"Unknown func_type: {func_type}")
    return u, f


@dataclass
class ModelConfig:
    Dim: int = None

    
    l: Optional[float] = None
    sigma: Optional[float] = None

    hidden_layers_BNN: int = None

    
    # N_z: int = None
    # N_points: int = None
    
    N_basis: int = None
    Sample_size: int = None

    sigma_B: int = None
    theta_size: int = None
    
    # 使用 field(init=False) 避免在初始化时被修改
    BranchNet_features: List[int] = field(init=False)
    TrunkNet_features: List[int] = field(init=False)

    
    # D: List[int] = field(init=False)
    # G: List[int] = field(init=False)

    function: str = "Poisson"
    function_K: float = 1

    remark: str = "None"
    out_dir: str = "./out/"

    def __post_init__(self):
        
        # # 初始化 l
        # if self.l is None:
        #     self.l = jnp.array([0.2] * self.Dim)
        # elif isinstance(self.l, (float, int)):
        #     self.l = jnp.array([self.l] * self.Dim)
        # else:
        #     self.l = jnp.array(self.l)
        
        # # 初始化 sigma
        # if self.sigma is None:
        #     if len(self.l) != self.Dim:
        #         raise ValueError(f"sigma 长度 {len(self.l)} 和 Dim {self.Dim} 不一致")
        #     self.sigma = 1.0 / self.l
        # else:
        #     if isinstance(self.sigma, (float, int)):
        #         self.sigma = jnp.array([self.sigma] * self.Dim)
        #     else:
        #         self.sigma = jnp.array(self.sigma)

        
        self.sigma = 1.0 / self.l
        
        # 初始化网络特征列表
        self.BranchNet_features = [self.N_basis] * 5
        self.TrunkNet_features = [self.N_basis] * 5
        
        # self.VectorField_features = [self.N_basis*2] * 4 + [self.N_basis]
        # self.G = [self.N_basis*2] * 4 + [self.N_basis]
        # self.D = [self.N_basis*2] * 4 + [1]

    
    # =====================================================
    #    Pickle 保存与加载 (极简版)
    # =====================================================
    def save(self, path: str):
        """保存整个对象状态到 pickle 文件"""
        # 自动创建文件夹
        folder = os.path.dirname(path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
            
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Config saved (Pickle): {path}")

    @staticmethod
    def load(path: str):
        """从 pickle 文件恢复对象"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
            
        with open(path, "rb") as f:
            obj = pickle.load(f)
        
        print(f"Config loaded (Pickle): {path}")
        return obj


    
# 1. BranchNet
class BranchNet_1D(nn.Module):
    features: list[int]
    activation: Callable = nn.relu
    # kernel_init: Callable = nn.initializers.lecun_normal() # tanh
    kernel_init: Callable = nn.initializers.variance_scaling(scale=1.0, mode='fan_avg', distribution='uniform') # relu
    
    # kernel_init: Callable = nn.initializers.normal(stddev=0.1)
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x_res = x  # 残差分支
            
            x = nn.Dense(feat, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
            x = self.activation(x)

            # 如果维度不匹配，用线性映射调整残差
            if x.shape[-1] != x_res.shape[-1]:
                x_res = nn.Dense(feat, kernel_init=self.kernel_init, bias_init=self.bias_init)(x_res)
            x = x + x_res  # 残差连接
            
        # 输出层
        x = nn.Dense(self.features[-1], kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        return x


        
# 2. TrunkNet
class TrunkNet(nn.Module):
    features: list[int]
    activation: Callable = nn.tanh
    
    kernel_init: Callable = nn.initializers.lecun_normal() # tanh
    # kernel_init: Callable = nn.initializers.variance_scaling(scale=1.0, mode='fan_avg', distribution='uniform') # relu
    bias_init: Callable = nn.initializers.zeros
    
    # kernel_init: Callable = nn.initializers.normal(stddev=10)
    # bias_init: Callable = nn.initializers.uniform(10)

    @nn.compact
    def __call__(self, x):
        
        num_freq = self.features[0]//2  # 截止频率 f_max
    
        x_proj = nn.Dense(
            num_freq, 
            kernel_init = nn.initializers.normal(stddev=2), 
            use_bias = False
        )(x)

        
        x_fourier = jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj), x], axis=-1)
        x = x_fourier

        for feat in self.features[:-1]:
            x_res = x  # 残差分支

            x = nn.Dense(feat, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
            x = self.activation(x)

            # 如果维度不匹配，用线性映射调整残差
            if x.shape[-1] != x_res.shape[-1]:
                x_res = nn.Dense(feat, kernel_init=self.kernel_init, bias_init=self.bias_init)(x_res)
            x = x + x_res  # 残差连接
        
        # 输出层
        x = nn.Dense(self.features[-1], kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        return x
        
class VectorField(nn.Module):

    features: list[int]
    activation: Callable = nn.gelu
    # kernel_init: Callable = nn.initializers.lecun_normal()
    kernel_init: Callable = nn.initializers.variance_scaling(scale=1.0, mode='fan_avg', distribution='uniform') # Xavier 初始化
    # kernel_init: Callable = nn.initializers.normal(stddev=0.1)
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, t, xt):
        x = jnp.concatenate([t, xt], axis=-1)
        
        for feat in self.features[:-1]:
            x_res = x  # 残差分支
            
            x = nn.Dense(feat, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
            x = self.activation(x)

            # 如果维度不匹配，用线性映射调整残差
            if x.shape[-1] != x_res.shape[-1]:
                x_res = nn.Dense(feat, kernel_init=self.kernel_init, bias_init=self.bias_init)(x_res)
            x = x + x_res  # 残差连接
            
        # 输出层
        x = nn.Dense(self.features[-1], kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        return x

class mVectorField(nn.Module):

    features: list[int]
    activation: Callable = nn.gelu
    # kernel_init: Callable = nn.initializers.lecun_normal()
    kernel_init: Callable = nn.initializers.variance_scaling(scale=1.0, mode='fan_avg', distribution='uniform') # Xavier 初始化
    # kernel_init: Callable = nn.initializers.normal(stddev=0.1)
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        x_res0 = x  # 残差分支
        
        for feat in self.features[:-1]:
            x_res = x  # 残差分支
            
            x = nn.Dense(feat, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
            x = self.activation(x)

            # 如果维度不匹配，用线性映射调整残差
            if x.shape[-1] != x_res.shape[-1]:
                x_res = nn.Dense(feat, kernel_init=self.kernel_init, bias_init=self.bias_init)(x_res)
            x = x + x_res  # 残差连接
            
        # 输出层
        x = nn.Dense(self.features[-1], kernel_init=self.kernel_init, bias_init=self.bias_init)(x)+x_res0
        return x

        
class MLP_GAN(nn.Module):

    features: list[int]
    activation: Callable = nn.gelu
    # kernel_init: Callable = nn.initializers.lecun_normal()
    kernel_init: Callable = nn.initializers.variance_scaling(scale=1.0, mode='fan_avg', distribution='uniform') # Xavier 初始化
    # kernel_init: Callable = nn.initializers.normal(stddev=0.1)
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            # x_res = x  # 残差分支
            
            x = nn.Dense(feat, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
            x = self.activation(x)

            # # 如果维度不匹配，用线性映射调整残差
            # if x.shape[-1] != x_res.shape[-1]:
            #     x_res = nn.Dense(feat, kernel_init=self.kernel_init, bias_init=self.bias_init)(x_res)
            # x = x + x_res  # 残差连接
            
        # 输出层
        x = nn.Dense(self.features[-1], kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        return x


# 3. DeepONet
class Model(nn.Module):
    Config: ModelConfig

    def setup(self):
        
        # if self.Config.Dim == 1:
        #     self.branch_net = BranchNet_1D(features = self.Config.BranchNet_features)
            
        # elif self.Config.Dim == 2:
        #     self.branch_net = BranchNet_2D(N_points = self.Config.N_points, N_basis = self.Config.N_basis)

        # elif self.Config.Dim == 3:
        #     self.branch_net = BranchNet_3D(N_points = self.Config.N_points, N_basis = self.Config.N_basis)
            
        # else:
        #     pass

        self.branch_net = BranchNet_1D(features = self.Config.BranchNet_features)
        # self.trunk_net = TrunkNet_new(features = self.Config.TrunkNet_features, name='trunk_net')
        
        self.trunk_net = TrunkNet(features = self.Config.TrunkNet_features, name='trunk_net')
        # self.trunk_net = TrunkNet_s(features = self.Config.TrunkNet_features, name='trunk_net')#siren

        # self.vectorfield = VectorField(features = self.Config.VectorField_features, name='vectorfield')
        # self.mvectorfield = mVectorField(features = self.Config.VectorField_features, name='mvectorfield')
        # self.G = MLP_GAN(features = self.Config.G, name='G')
        


    
    def __call__(self, f, x):
        branch_out = self.branch_net(f)  
        u, f = self.pde(x, branch_out)
        return u, f
        

    def pde(self, x, k):
        
        def trunk_fn(x):
            return self.trunk_net(x)

        u = jnp.einsum('bd,sd->bs',k, trunk_fn(x))/jnp.sqrt(self.Config.N_basis)#(128, 1024)

        # tem_d = jax.vmap(jax.jacfwd(trunk_fn))(x) #(1024, 320, 3)
        # tem_dd = jnp.diagonal(jax.vmap(jax.jacfwd(jax.jacfwd(trunk_fn)))(x), axis1=-2, axis2=-1) #(1024, 320, 3, 3)

        # du = jnp.einsum('bd,sdi->bsi',k, tem_d)/jnp.sqrt(self.Config.N_basis)
        # ddu = jnp.einsum('bd,sdi->bsi',k, tem_dd)/jnp.sqrt(self.Config.N_basis)
        
        # f = func([u, du, ddu],self.Config.function_K,self.Config.function)[1]
        return u, None
        
        

    def get_branch_net_output(self,f):
        return self.branch_net(f) 

    def get_trunk_net_output(self,x):
        return self.trunk_net(x)
        
    # def get_G(self, z):
    #     return self.G(z)

    # def get_vectorfield_output(self, t, xt):
    #     return self.vectorfield(t, xt)

    # def get_mvectorfield_output(self, xt):
    #     return self.mvectorfield(xt)



# 1Coordinates[t][x][y]  2 Coordinates[0]=t, Coordinates[1]=x, Coordinates[2]=y 
# 注意matlab拉直和python默认不一致
def make_grid(*arrays):
    """
    arrays: 一个或多个 1D 数组，每个数组代表每个维度的坐标
    返回 ND 网格坐标，每行是一个坐标点
    """
    grids = jnp.meshgrid(*arrays, indexing='ij')  # 保持坐标顺序一致
    # stack 并展平
    grid_points = jnp.stack(grids, -1).reshape(-1, len(arrays))
    return grid_points


def RL2E(predict,ref):
    return jnp.linalg.norm(predict-ref) / jnp.linalg.norm(ref)

    
def sample_time(key, batch):
    return jax.random.uniform(key, (batch, 1), minval=0.0, maxval=1.0)

def interpolate(x0, x1, t):
    return (1.0 - t) * x0 + t * x1

def target_vector_field(x0, x1, t):
    # 对于线性插值路径，最简单的 u*(x_t, t) = x1 - x0
    # （更一般的设定会考虑路径与密度配重，这里用标准最小例子）
    return x1 - x0

def solve_forward_ode(x0, apply_fn, steps):
    def vf(t, y):
        y_in = y[None, :]
        t_in = jnp.array([[t]])
        return apply_fn(t_in, y_in)[0]

    h = 1.0 / steps

    def euler_step(carry, _):
        t, y = carry
        dy = vf(t, y)
        y_next = y + h * dy
        return (t + h, y_next), None

    def single_solve(x_init):
        (t_final, y_final), _ = jax.lax.scan(euler_step, (0.0, x_init), None, length=steps)
        return y_final

    x1 = jax.vmap(single_solve)(x0)
    return x1

def solve_forward_ode_rk4(x0, apply_fn, steps):
    """
    x0: (batch, dim)
    apply_fn: function(t_in, y_in) -> (batch, dim)
    steps: number of integration steps
    """
    h = 1.0 / steps

    def vf(t, y):
        # y: (dim,)
        y_in = y[None, :]          # (1, dim)
        t_in = jnp.array([[t]])    # (1, 1)
        return apply_fn(t_in, y_in)[0]  # (dim,)

    def rk4_step(carry, _):
        t, y = carry

        k1 = vf(t, y)
        k2 = vf(t + h/2, y + h/2 * k1)
        k3 = vf(t + h/2, y + h/2 * k2)
        k4 = vf(t + h, y + h * k3)

        y_next = y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        return (t + h, y_next), None

    def single_solve(x_init):
        (t_final, y_final), _ = jax.lax.scan(rk4_step, (0.0, x_init), None, length=steps)
        return y_final

    # batch 处理
    x1 = jax.vmap(single_solve)(x0)
    return x1
    
def BNN_boundary_(key, sigma, x, sample_size, in_dim, hidden_layers, out_dim):
    
    key, subkey = jax.random.split(key)
    w1 = jax.random.normal(subkey, shape=(sample_size, in_dim, hidden_layers))*sigma
    
    key, subkey = jax.random.split(key)
    b1 = jax.random.normal(subkey, shape=(sample_size, 1, hidden_layers))
    
    key, subkey = jax.random.split(key)
    w2 = jax.random.normal(subkey, shape=(sample_size, hidden_layers, out_dim))

    
    h1 = jnp.sqrt(2)*jnp.cos(jnp.einsum('bih,si->bsh',w1, jnp.concatenate([jnp.sin(x),jnp.cos(x)],-1)) + b1 - jnp.pi/4)
    return jnp.einsum('bho,bsh->bso',w2, h1)/jnp.sqrt(hidden_layers)

BNN_boundary = jax.jit(BNN_boundary_, static_argnums=(3,4,5,6))


def BNN_sin_(key, sigma, x, sample_size, in_dim, hidden_layers, out_dim):
    key, subkey = jax.random.split(key)
    w1 = jax.random.normal(subkey, shape=(sample_size, in_dim, hidden_layers)) * sigma[None,:,None]

    key, subkey = jax.random.split(key)
    b1 = jax.random.normal(subkey, shape=(sample_size, 1, hidden_layers)) * sigma.mean()

    key, subkey = jax.random.split(key)
    w2 = jax.random.normal(subkey, shape=(sample_size, hidden_layers, out_dim))

    if len(x.shape)==2:
        u, du, ddu = jax.vmap(single_sample_forward_second_order_sin, in_axes=(0, 0, 0, None))(w1, b1, w2, x)
    elif len(x.shape)==3:
        u, du, ddu = jax.vmap(single_sample_forward_second_order_sin, in_axes=(0, 0, 0, 0))(w1, b1, w2, x)
    else:
        print('Err shape')
    return u, du, jnp.diagonal(ddu, axis1=-2, axis2=-1)

    
BNN_sin = jax.jit(BNN_sin_, static_argnums=(3,4,5,6))



def EXP_BNN_sin_(key, sigma, x, sample_size, in_dim, hidden_layers, out_dim):
    key, subkey = jax.random.split(key)
    w1 = jax.random.normal(subkey, shape=(sample_size, in_dim, hidden_layers)) * sigma[None,:,None]

    key, subkey = jax.random.split(key)
    b1 = jax.random.normal(subkey, shape=(sample_size, 1, hidden_layers)) * sigma.mean()

    key, subkey = jax.random.split(key)
    w2 = jax.random.normal(subkey, shape=(sample_size, hidden_layers, out_dim))

    if len(x.shape)==2:
        u, du, ddu = jax.vmap(EXP_single_sample_forward_second_order_sin, in_axes=(0, 0, 0, None))(w1, b1, w2, x)
    elif len(x.shape)==3:
        u, du, ddu = jax.vmap(EXP_single_sample_forward_second_order_sin, in_axes=(0, 0, 0, 0))(w1, b1, w2, x)
    else:
        print('Err shape')
    return u, du, jnp.diagonal(ddu, axis1=-2, axis2=-1)

    
EXP_BNN_sin = jax.jit(EXP_BNN_sin_, static_argnums=(3,4,5,6))


# def SIN_(key, sigma, x, sample_size, in_dim=None, hidden_layers=None, out_dim=None):
#     # Sample random amplitude and frequency
#     key, subkey_A = jax.random.split(key)
#     A = jax.random.uniform(subkey_A, (sample_size, 1, 1), minval=1.0, maxval=3.0)

#     key, subkey_Omega = jax.random.split(key)
#     Omega = jax.random.uniform(subkey_Omega, (sample_size, 1, 1), minval=2.0, maxval=12.0)

#     # Compute sin, cos and derivatives
#     if x.ndim not in (2, 3):
#         raise ValueError(f"Unsupported input shape {x.shape}. Expected 2D or 3D.")
        
#     if x.shape[-1] != 1:
#         raise ValueError(f"Unsupported input shape {x.shape}. Expected 2D or 3D.")

#     u = A * jnp.sin(Omega * x)
#     du = Omega * A * jnp.cos(Omega * x)
#     ddu = -Omega**2 * A * jnp.sin(Omega * x)

#     return u, du, jnp.trace(ddu, axis1=-2, axis2=-1)


# SIN = jax.jit(SIN_, static_argnums=(3,4,5,6))



def single_sample_forward_second_order_sin(w1, b1, w2, x):
    """
    单个神经网络样本的前向传播和一阶、二阶导数计算。

    输入：
        w1: (in_dim, h)
        b1: (1, h)
        w2: (h, out_dim)
        x: (s, in_dim)

    输出：
        u:   (s, out_dim)
        du:  (s, out_dim, in_dim)
        ddu: (s, out_dim, in_dim, in_dim)
    """
    z = jnp.einsum('ih,si->sh', w1, x) + b1                   # (s, h)
    sin_term = jnp.sin(z - jnp.pi / 4)                        # (s, h)
    cos_term = jnp.cos(z - jnp.pi / 4)                        # (s, h)
    h = jnp.sqrt(2) * cos_term                                # (s, h)

    u = jnp.einsum('ho,sh->so', w2, h) / jnp.sqrt(h.shape[1]) # (s, out_dim)

    dh_dx = -jnp.sqrt(2) * jnp.einsum('sh,ih->shi', sin_term, w1)      # (s, h, in_dim)
    du_dx = jnp.einsum('shi,ho->soi', dh_dx, w2) / jnp.sqrt(h.shape[1])# (s, out_dim, in_dim)

    d2u_dx2 = -jnp.sqrt(2) * jnp.einsum('sh,ih,jh,ho->soij', cos_term, w1, w1, w2)
    d2u_dx2 /= jnp.sqrt(h.shape[1])                                     # (s, out_dim, in_dim, in_dim)
    
    

    return u, du_dx, d2u_dx2


# # 手写公式：输出取 exp(u_old)
# def EXP_single_sample_forward_second_order_sin(w1, b1, w2, x):
#     z = jnp.einsum('ih,si->sh', w1, x) + b1
#     sin_term = jnp.sin(z - jnp.pi / 4)
#     cos_term = jnp.cos(z - jnp.pi / 4)
#     h = jnp.sqrt(2) * cos_term

#     u_old = jnp.einsum('ho,sh->so', w2, h) / jnp.sqrt(h.shape[1])

#     dh_dx = -jnp.sqrt(2) * jnp.einsum('sh,ih->shi', sin_term, w1)
#     du_old_dx = jnp.einsum('shi,ho->soi', dh_dx, w2) / jnp.sqrt(h.shape[1])

#     d2u_old_dx2 = -jnp.sqrt(2) * jnp.einsum('sh,ih,jh,ho->soij', cos_term, w1, w1, w2)
#     d2u_old_dx2 /= jnp.sqrt(h.shape[1])

#     # exp 变换
#     u = jnp.exp(u_old)
#     du_dx = u[..., None] * du_old_dx
#     d2u_dx2 = u[..., None, None] * (d2u_old_dx2 + jnp.einsum('sok,som->sokm', du_old_dx, du_old_dx))

#     return u, du_dx, d2u_dx2
    
def EXP_single_sample_forward_second_order_sin(w1, b1, w2, x):
    k = 0.5
    z = jnp.einsum('ih,si->sh', w1, x) + b1
    sin_term = jnp.sin(z - jnp.pi / 4)
    cos_term = jnp.cos(z - jnp.pi / 4)
    h = jnp.sqrt(2) * cos_term

    u_old = jnp.einsum('ho,sh->so', w2, h) / jnp.sqrt(h.shape[1])

    dh_dx = -jnp.sqrt(2) * jnp.einsum('sh,ih->shi', sin_term, w1)
    du_old_dx = jnp.einsum('shi,ho->soi', dh_dx, w2) / jnp.sqrt(h.shape[1])

    d2u_old_dx2 = -jnp.sqrt(2) * jnp.einsum('sh,ih,jh,ho->soij', cos_term, w1, w1, w2)
    d2u_old_dx2 /= jnp.sqrt(h.shape[1])

    # ====== EXP(k * GP) 变换 ======
    u = jnp.exp(k * u_old)

    # 一阶导：du/dx = k * u * du_old_dx
    du_dx = k * u[..., None] * du_old_dx

    # 二阶导：d2u/dx2 = k*u*(d2u_old + k * du_old * du_old^T)
    d2u_dx2 = k * u[..., None, None] * (
        d2u_old_dx2 + k * jnp.einsum('sok,som->sokm', du_old_dx, du_old_dx)
    )

    return u, du_dx, d2u_dx2


def single_sample_forward_second_order_tanh(w1, b1, w2, x):
    """
    单个神经网络样本的前向传播和一阶、二阶导数计算，使用 tanh 激活。

    输入：
        w1: (in_dim, h)
        b1: (1, h)
        w2: (h, out_dim)
        x:  (s, in_dim)

    输出：
        u:   (s, out_dim)
        du:  (s, out_dim, in_dim)
        ddu: (s, out_dim, in_dim, in_dim)
    """
    z = jnp.einsum('ih,si->sh', w1, x) + b1       # (s, h)
    h = jnp.tanh(z)                               # (s, h)

    u = jnp.einsum('ho,sh->so', w2, h) / jnp.sqrt(h.shape[1])  # (s, out_dim)

    sech2 = 1.0 - h**2                            # tanh'(z) = 1 - tanh^2(z)
    dh_dx = jnp.einsum('sh,ih->shi', sech2, w1)   # (s, h, in_dim)
    du_dx = jnp.einsum('shi,ho->soi', dh_dx, w2) / jnp.sqrt(h.shape[1])  # (s, out_dim, in_dim)

    # tanh''(z) = -2 tanh(z) * (1 - tanh^2(z))
    d2phi = -2.0 * h * sech2                      # (s, h)
    d2u_dx2 = jnp.einsum('sh,ih,jh,ho->soij', d2phi, w1, w1, w2)
    d2u_dx2 /= jnp.sqrt(h.shape[1])               # (s, out_dim, in_dim, in_dim)

    return u, du_dx, d2u_dx2


