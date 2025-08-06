# adapted from https://github.com/nitzanlab/prototype-equivalences/blob/main/NFDiffeo.py#L1028
# renamed reverse functions to inverse to be able to use in DAA
import torch
from torch import nn
from torch.optim import Adam
import numpy as np
from typing import Tuple


class RFFCoupling(nn.Module):
    """
    An implementation of the invertible Fourier features transform, given by:
                    f(x_1, x_2) = (x_1, x_2 * exp(s(x_1)) + t(x_1))
    where the functions s(x_1) and t(x_1) are both defined according to the Fourier features:
                    t(x) = sum{ a_k cos(2*pi*k*x/R + b_k) } with k between 0 and K
    where a_k are the coefficients of the Fourier features and b_k are the phases. R defines the natural scale of the
    function and K is the number of Fourier components to be used.
    """
    def __init__(self, dim: int, K: int=32, reverse: bool=False):
        """
        Initializes the Fourier-features transform layer
        :param dim: number of dimensions expected for inputs
        :param K: number of Fourier coefficients to use
        :param R: the maximum interval assumed in the data
        :param reverse:
        """
        super().__init__()
        self.reord = reverse
        x1, x2 = self._split(torch.zeros(1, dim))
        self.W_s = nn.Parameter(1e-3*torch.randn(K, x1.shape[-1]))
        self.W_t = nn.Parameter(1e-3*torch.randn(K, x1.shape[-1]))
        self.g_s = nn.Parameter(1e-3*torch.randn(x2.shape[-1], K))
        self.g_t = nn.Parameter(1e-3*torch.randn(x2.shape[-1], K))
        self.b_s = nn.Parameter(torch.rand(K)*2*np.pi*1e-2)
        self.b_t = nn.Parameter(torch.rand(K)*2*np.pi*1e-2)

    @staticmethod
    def _ff(x: torch.Tensor, gamma: torch.Tensor, W: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """
        :param x: a torch tensor with shape [N, dim_in] the input to the transformation
        :param gamma: the output layer, a torch tensor with shape [dim_out, K]
        :param W: the hidden weights matrix, a torch tensor with shape [K, dim_in]
        :param phi: the bias, a torch tensor with shape [K]
        :return: the transformed input x, a tensor with shape [N, dim_out]
        """
        return torch.cos(x@W.T+phi[None])@gamma.T

    @staticmethod
    def _df(x: torch.Tensor, gamma: torch.Tensor, W: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """
        :param x: a torch tensor with shape [N, dim_in] the input to the transformation
        :param gamma: the output layer, a torch tensor with shape [dim_out, K]
        :param W: the hidden weights matrix, a torch tensor with shape [K, dim_in]
        :param phi: the bias, a torch tensor with shape [K]
        :return: the Jacobian of the FF function, with shape [N, dim_in, dim_out]
        """
        sin = torch.sin(x@W.T+phi[None])  # [N, k]
        interm = W.T[None]*sin[:, None, :]  # [N, dim_in, k]
        return - interm@gamma.T  # [N, dim_in, dim_out]

    def _s(self, x1: torch.Tensor, rev: bool=False) -> torch.Tensor:
        return torch.exp(self._ff(x1, self.g_s, self.W_s, self.b_s)*(-1 if rev else 1))

    def _t(self, x1: torch.Tensor) -> torch.Tensor:
        return self._ff(x1, self.g_t, self.W_t, self.b_t)

    def _split(self, x: torch.Tensor) -> Tuple:
        if self.reord:
            x2, x1 = torch.chunk(x, 2, dim=1)
        else:
            x1, x2 = torch.chunk(x, 2, dim=1)
        return x1, x2

    def _cat(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.reord: return torch.cat([x2, x1], dim=1)
        else: return torch.cat([x1, x2], dim=1)

    def freeze_scale(self, freeze: bool=True):
        """
        Freeze all parameters that impact log-determinant
        :param freeze: a boolean indicating whether to freeze (True) or unfreeze (False)
        """
        self.W_s.requires_grad = freeze
        self.b_s.requires_grad = freeze
        self.g_s.requires_grad = freeze

    def rand_init(self, amnt: float):
        """
        Random initialization of the layer
        :param amnt: a float with the strength of the initialization
        """
        self.W_t.data = torch.randn_like(self.W_t)*amnt
        self.b_t.data = torch.randn_like(self.b_t)*amnt
        self.g_t.data = torch.randn_like(self.g_t)*amnt

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the log-abs-determinant of the Jacobian evaluated at the point x
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :return: a torch tensor with shape [N,] of the log-abs-determinants of the Jacobian for each point x
        """
        x1, _ = self._split(x)
        return self._ff(x1, self.g_s, self.W_s, self.b_s).sum(dim=1)

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Jacobian of the transform at points x
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :return: a torch tensor with shape [N, dim, dim] of the Jacobians evaluated at each point x
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :return: a torch tensor with shape [N, dim] of the transformed points
        """
        x1, x2 = self._split(x)
        return self._cat(x1, self._s(x1)*x2 + self._t(x1))

    def jvp_forward(self, x: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates both the forward pass, as well as a Jacobian-vector-product (JVP) and the log-abs-determinant
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector on which to evaluate the JVP, a torch tensor with shape [N, dim] where N is the
                  number of points
        :return: the tuple (y, Jf, logdet) where:
                    - y: a torch tensor with shape [N, dim], which are the transformed xs
                    - Jf: a torch tensor with shape [N, dim], which are the JVP with f
                    - logdet: a torch tensor with shape [N,] which are the log-abs-determinants evaluated at the
                              points x
        """
        x1, x2 = self._split(x)
        s, t = self._s(x1), self._t(x1)
        y = self._cat(x1, s*x2 + t)

        f1, f2 = self._split(f)  # f1 shape: [N, dim_in], f2 shape: [N, dim_out]
        ds = (x2*s)[..., None, :]*self._df(x1, self.g_s, self.W_s, self.b_s)  # [N, dim_in, dim_out]
        dt = self._df(x1, self.g_t, self.W_t, self.b_t)  # [N, dim_in, dim_out]
        Jf2 = ((ds + dt).transpose(-2, -1)@f1[..., None])[..., 0] + f2*s
        Jf = self._cat(f1, Jf2)

        logdet = self._ff(x1, self.g_s, self.W_s, self.b_s).sum(dim=1)

        return y, Jf, logdet

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the reverse transformation
        :param y: a torch tensor with shape [N, dim]
        :return: a torch tensor with shape [N, dim] which are the points after applying the reverse transformation
        """
        y1, y2 = self._split(y)
        return self._cat(y1, self._s(y1, rev=True)*(y2 - self._t(y1)))

    def jvp_reverse(self, y: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the reverse pass as well as the inverse Jacobian-vector-product (iJVP)
        :param y: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector on which to evaluate the iJVP, a torch tensor with shape [N, dim] where N is the
                  number of points
        :return: the tuple (x, inv(J)f, logdet) where:
                    - y: a torch tensor with shape [N, dim], which are the transformed ys
                    - inv(J)f: a torch tensor with shape [N, dim], which is the iJVP with f
        """
        raise NotImplementedError


class FFCoupling(nn.Module):
    """
    An implementation of the invertible Fourier features transform, given by:
                    f(x_1, x_2) = (x_1, x_2 * exp(s(x_1)) + t(x_1))
    where the functions s(x_1) and t(x_1) are both defined according to the Fourier features:
                    t(x) = sum{ a_k cos(2*pi*k*x/R + b_k) } with k between 0 and K
    where a_k are the coefficients of the Fourier features and b_k are the phases. R defines the natural scale of the
    function and K is the number of Fourier components to be used.
    """
    def __init__(self, dim: int, K: int=32, R: int=10, reverse: bool=False, scale_free: bool=False, split_dims=None):
        """
        Initializes the Fourier-features transform layer
        :param dim: number of dimensions expected for inputs
        :param K: number of Fourier coefficients to use
        :param R: the maximum interval assumed in the data
        :param scale_free: return a scaling-free version of the FFCoupling
        :param reverse:
        """
        super().__init__()
        self.reord = reverse
        self.split_dims = split_dims
        x1, x2 = self._split(torch.zeros(1, dim))
        self.a_t = nn.Parameter(1e-3*torch.randn(K, x1.shape[-1], x2.shape[-1]))
        self.b_t = nn.Parameter(torch.rand(K, x1.shape[-1])*2*np.pi*1e-2)
        self.scale_free = scale_free
        self.R = R

        if self.scale_free:
            self.a_s = nn.Parameter(torch.zeros(K, x1.shape[-1], x2.shape[-1]))
            self.b_s = nn.Parameter(torch.zeros(K, x1.shape[-1]))
            self.a_s.requires_grad = False
            self.b_s.requires_grad = False
        else:
            self.a_s = nn.Parameter(1e-3*torch.randn(K, x1.shape[-1], x2.shape[-1]))
            self.b_s = nn.Parameter(torch.rand(K, x1.shape[-1])*2*np.pi*1e-2)

    @staticmethod
    def _ff(x: torch.Tensor, gamma: torch.Tensor, phi: torch.Tensor, R: float) -> torch.Tensor:
        """
        :param x: a torch tensor with shape [N, dim_in] the input to the transformation
        :param gamma: a torch tensor with shape [K, dim_in, dim_out] of the Fourier coefficients
        :param phi: a torch tensor with shape [K, dim_in] the phases of the transformation
        :param R: a float depicting the range
        :return: the transformed input x, a tensor with shape [N, dim_out]
        """
        freqs = torch.arange(0, gamma.shape[0], device=x.device)*2*np.pi/R
        return torch.sum(torch.cos(freqs[None, :, None]*x[:, None] + phi[None])[..., None] * gamma[None], dim=(1, -2))

    @staticmethod
    def _df(x: torch.Tensor, gamma: torch.Tensor, phi: torch.Tensor, R: float) -> torch.Tensor:
        """
        :param x: a torch tensor with shape [N, dim_in] the input to the transformation
        :param gamma: a torch tensor with shape [K, dim_in, dim_out] of the Fourier coefficients
        :param phi: a torch tensor with shape [K, dim_in] the phases of the transformation
        :param R: a float depicting the range
        :return: the Jacobian of the FF function, with shape [N, dim_in, dim_out]
        """
        freqs = torch.arange(0, gamma.shape[0], device=x.device)*2*np.pi/R
        sins = - freqs[None, :, None]*torch.sin(freqs[None, :, None]*x[:, None] + phi[None])  # [N, K, dim_in]
        return torch.sum(sins[..., None]*gamma[None], dim=1)  # [N, dim_in, dim_out]

    def _s(self, x1: torch.Tensor, rev: bool=False) -> torch.Tensor:
        return torch.exp(self._ff(x1, self.a_s, self.b_s, self.R)*(-1 if rev else 1))

    def _t(self, x1: torch.Tensor) -> torch.Tensor:
        return self._ff(x1, self.a_t, self.b_t, self.R)

    def _split(self, x: torch.Tensor) -> Tuple:
        if self.split_dims is None:
            if self.reord:
                x2, x1 = torch.chunk(x, 2, dim=1)
            else:
                x1, x2 = torch.chunk(x, 2, dim=1)
        else:
            outinds = [i for i in range(x.shape[-1]) if i not in self.split_dims]
            x1 = x[:, self.split_dims]
            x2 = x[:, outinds]
        return x1, x2

    def _cat(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.reord: return torch.cat([x2, x1], dim=1)
        else: return torch.cat([x1, x2], dim=1)

    def freeze_scale(self, freeze: bool=True):
        """
        Freeze all parameters that impact log-determinant
        :param freeze: a boolean indicating whether to freeze (True) or unfreeze (False)
        """
        self.a_s.requires_grad = not freeze
        self.b_s.requires_grad = not freeze
        self.a_t.requires_grad = not freeze
        self.b_t.requires_grad = not freeze

        if self.scale_free:
            self.a_s.requires_grad = False
            self.b_s.requires_grad = False

    def rand_init(self, amnt: float):
        """
        Random initialization of the layer
        :param amnt: a float with the strength of the initialization
        """
        self.b_s.data = np.pi*torch.rand_like(self.b_s)*amnt
        self.b_t.data = np.pi*torch.rand_like(self.b_t)*amnt
        self.a_t.data = torch.randn_like(self.a_t)*amnt/np.sum(self.a_t.shape)
        self.a_s.data = torch.randn_like(self.a_t)*amnt*1e-2/np.sum(self.a_s.shape)

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the log-abs-determinant of the Jacobian evaluated at the point x
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :return: a torch tensor with shape [N,] of the log-abs-determinants of the Jacobian for each point x
        """
        x1, _ = self._split(x)
        return self._ff(x1, self.a_s, self.b_s, self.R).sum(dim=1)

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Jacobian of the transform at points x
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :return: a torch tensor with shape [N, dim, dim] of the Jacobians evaluated at each point x
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :return: a torch tensor with shape [N, dim] of the transformed points
        """
        x1, x2 = self._split(x)
        return self._cat(x1, self._s(x1)*x2 + self._t(x1))

    def jvp_forward(self, x: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates both the forward pass, as well as a Jacobian-vector-product (JVP) and the log-abs-determinant
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector on which to evaluate the JVP, a torch tensor with shape [N, dim] where N is the
                  number of points
        :return: the tuple (y, Jf, logdet) where:
                    - y: a torch tensor with shape [N, dim], which are the transformed xs
                    - Jf: a torch tensor with shape [N, dim], which are the JVP with f
                    - logdet: a torch tensor with shape [N,] which are the log-abs-determinants evaluated at the
                              points x
        """
        x1, x2 = self._split(x)
        s, t = self._s(x1), self._t(x1)
        y = self._cat(x1, s*x2 + t)

        f1, f2 = self._split(f)  # f1 shape: [N, dim_in], f2 shape: [N, dim_out]
        ds = (x2*s)[..., None, :]*self._df(x1, self.a_s, self.b_s, self.R)  # [N, dim_in, dim_out]
        dt = self._df(x1, self.a_t, self.b_t, self.R)  # [N, dim_in, dim_out]
        Jf2 = ((ds + dt).transpose(-2, -1)@f1[..., None])[..., 0] + f2*s
        Jf = self._cat(f1, Jf2)

        logdet = self._ff(x1, self.a_s, self.b_s, self.R).sum(dim=1)

        return y, Jf, logdet

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the reverse transformation
        :param y: a torch tensor with shape [N, dim]
        :return: a torch tensor with shape [N, dim] which are the points after applying the reverse transformation
        """
        y1, y2 = self._split(y)
        return self._cat(y1, self._s(y1, rev=True)*(y2 - self._t(y1)))

    def jvp_reverse(self, y: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the reverse pass as well as the inverse Jacobian-vector-product (iJVP)
        :param y: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector on which to evaluate the iJVP, a torch tensor with shape [N, dim] where N is the
                  number of points
        :return: the tuple (x, inv(J)f, logdet) where:
                    - y: a torch tensor with shape [N, dim], which are the transformed ys
                    - inv(J)f: a torch tensor with shape [N, dim], which is the iJVP with f
        """
        y1, y2 = self._split(y)
        s, t = self._s(y1, rev=True), self._t(y1)
        x2 = s*(y2 - self._t(y1))
        x = self._cat(y1, x2)

        f1, f2 = self._split(f)  # f1 shape: [N, dim_in], f2 shape: [N, dim_out]
        ds = x2[:, None]*self._df(y1, self.a_s, self.b_s, self.R)  # [N, dim_in, dim_out]
        dt = s[:, None]*self._df(y1, self.a_t, self.b_t, self.R)  # [N, dim_in, dim_out]
        Jf2 = -((ds+dt).transpose(-2, -1) @ f1[..., None])[..., 0] + f2 * s
        Jf = self._cat(f1, Jf2)

        return x, Jf


class AffineCoupling(nn.Module):

    def __init__(self, dim: int, K: int=32, reverse: bool=False):
        """
        Initializes an AffineCoupling layer with MLPs with single hidden layers
        :param dim: number of dimensions expected for inputs
        :param K: width of MLP
        :param reverse:
        """
        super().__init__()
        self.reord = reverse
        x1, x2 = self._split(torch.ones(1, dim))

        self.s = nn.Sequential(nn.Linear(x1.shape[-1], K), nn.GELU(), nn.Linear(K, K),
                               nn.GELU(), nn.Linear(K, x2.shape[-1]))
        for param in self.s.parameters(): param.data = 1e-3*torch.randn_like(param.data)

        self.t = nn.Sequential(nn.Linear(x1.shape[-1], K), nn.GELU(), nn.Linear(K, K),
                               nn.GELU(), nn.Linear(K, x2.shape[-1]))
        for param in self.t.parameters(): param.data = 1e-3*torch.randn_like(param.data)

    def _s(self, x1: torch.Tensor, rev: bool=False) -> torch.Tensor:
        return torch.exp(self.s(x1)*(-1 if rev else 1))

    def _t(self, x1: torch.Tensor) -> torch.Tensor:
        return self.t(x1)

    def _split(self, x: torch.Tensor) -> Tuple:
        if self.reord:
            x2, x1 = x.split(x.shape[-1]//2, dim=1)
        else:
            x1, x2 = x.split(x.shape[-1]//2, dim=1)
        return x1, x2

    def _cat(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.reord: return torch.cat([x2, x1], dim=1)
        else: return torch.cat([x1, x2], dim=1)

    def freeze_scale(self, freeze: bool=True):
        """
        Freeze all parameters that impact log-determinant
        :param freeze: a boolean indicating whether to freeze (True) or unfreeze (False)
        """
        for param in self.s.parameters():
            param.requires_grad = freeze

    def rand_init(self, amnt: float):
        """
        Random initialization of the layer
        :param amnt: a float with the strength of the initialization
        """
        for param in self.t.parameters(): param.data = torch.randn_like(param)*amnt

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the log-abs-determinant of the Jacobian evaluated at the point x
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :return: a torch tensor with shape [N,] of the log-abs-determinants of the Jacobian for each point x
        """
        x1, _ = self._split(x)
        return self.s(x1).sum(dim=1)

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Jacobian of the transform at points x
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :return: a torch tensor with shape [N, dim, dim] of the Jacobians evaluated at each point x
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :return: a torch tensor with shape [N, dim] of the transformed points
        """
        x1, x2 = self._split(x)
        return self._cat(x1, self._s(x1)*x2 + self._t(x1))

    def _Jf(self, x1: torch.Tensor, x2: torch.Tensor, f: torch.Tensor):
        f1, f2 = self._split(f.clone())
        x = torch.clone(x1).requires_grad_(True)
        jf1 = torch.func.jvp(lambda x: self.s(x)*x2+self.t(x), (x, ), (f1, ))[1]
        return self._cat(f1, jf1 + self.s(x)*f2)

    def jvp_forward(self, x: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates both the forward pass, as well as a Jacobian-vector-product (JVP) and the log-abs-determinant
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector on which to evaluate the JVP, a torch tensor with shape [N, dim] where N is the
                  number of points
        :return: the tuple (y, Jf, logdet) where:
                    - y: a torch tensor with shape [N, dim], which are the transformed xs
                    - Jf: a torch tensor with shape [N, dim], which are the JVP with f
                    - logdet: a torch tensor with shape [N,] which are the log-abs-determinants evaluated at the
                              points x
        """
        x1, x2 = self._split(x)
        s, t = self._s(x1), self._t(x1)
        y = self._cat(x1, s*x2 + t)

        Jf = self._Jf(x1, x2, f)

        logdet = self.s(x1).sum(dim=1)

        return y, Jf, logdet

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the reverse transformation
        :param y: a torch tensor with shape [N, dim]
        :return: a torch tensor with shape [N, dim] which are the points after applying the reverse transformation
        """
        y1, y2 = self._split(y)
        return self._cat(y1, self._s(y1, rev=True)*(y2 - self._t(y1)))

    def jvp_reverse(self, y: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the reverse pass as well as the inverse Jacobian-vector-product (iJVP)
        :param y: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector on which to evaluate the iJVP, a torch tensor with shape [N, dim] where N is the
                  number of points
        :return: the tuple (x, inv(J)f, logdet) where:
                    - y: a torch tensor with shape [N, dim], which are the transformed ys
                    - inv(J)f: a torch tensor with shape [N, dim], which is the iJVP with f
        """
        raise NotImplementedError


class PPT(nn.Module):
    """
    An implementation of the invertible positive-power transform, given by:
                    f(x) = sign(x) * |x| ^ beta
    where beta > 0. When beta < 1, this transform expands everything near the origin and "squishes" everything else, and
    does the opposite when beta > 1. The parameter beta is defined per coordinate and is parameterized as
    beta = exp(alpha) in order for beta to always be positive.
    """
    def __init__(self, dim: int):
        """
        Initializes the positive-power transform layer
        :param dim: number of dimensions expected for inputs
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(dim)*1e-2)

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the log-abs-determinant of the Jacobian evaluated at the point x
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :return: a torch tensor with shape [N,] of the log-abs-determinants of the Jacobian for each point x
        """
        beta = torch.exp(self.alpha)
        return torch.sum((beta[None]-1)*(self.alpha[None] + torch.log(torch.abs(x))), dim=-1)

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Jacobian of the transform at points x
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :return: a torch tensor with shape [N, dim, dim] of the Jacobians evaluated at each point x
        """
        beta = torch.exp(self.alpha)
        return torch.diag_embed(beta[None]*torch.pow(torch.abs(x), beta[None]-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :return: a torch tensor with shape [N, dim] of the transformed points
        """
        beta = torch.exp(self.alpha)
        return torch.sign(x)*torch.pow(torch.abs(x), beta[None, :])

    def jvp_forward(self, x: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates both the forward pass, as well as a Jacobian-vector-product (JVP) and the log-abs-determinant
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector on which to evaluate the JVP, a torch tensor with shape [N, dim] where N is the
                  number of points
        :return: the tuple (y, Jf, logdet) where:
                    - y: a torch tensor with shape [N, dim], which are the transformed xs
                    - Jf: a torch tensor with shape [N, dim], which are the JVP with f
                    - logdet: a torch tensor with shape [N,] which are the log-abs-determinants evaluated at the
                              points x
        """
        beta = torch.exp(self.alpha)
        y = torch.sign(x)*torch.pow(torch.abs(x), beta[None, :])
        J = torch.pow(torch.abs(x), beta[None, :]-1)*beta[None, :]
        return y, J*f, self.logdet(x)

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the reverse transformation
        :param y: a torch tensor with shape [N, dim]
        :return: a torch tensor with shape [N, dim] which are the points after applying the reverse transformation
        """
        beta = torch.exp(-self.alpha)
        return torch.sign(y)*torch.pow(torch.abs(y), beta[None, :])

    def jvp_reverse(self, y: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the reverse pass as well as the inverse Jacobian-vector-product (iJVP)
        :param y: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector on which to evaluate the iJVP, a torch tensor with shape [N, dim] where N is the
                  number of points
        :return: the tuple (x, inv(J)f, logdet) where:
                    - y: a torch tensor with shape [N, dim], which are the transformed ys
                    - inv(J)f: a torch tensor with shape [N, dim], which is the iJVP with f
        """
        raise NotImplementedError


class Affine(nn.Module):
    """
    An affine transformation module of a normalizing flow. To ensure invertibility, this transformation is defined as:
                                        y = WW^Tx + exp[phi] x + mu
    where the learnable parameters are W (with shape [dim, rank]), phi (with shape [dim]), and mu (with shape [dim])
    """
    def __init__(self, dim: int, rank: int=None, data_init: bool=False, mu: torch.Tensor=None):
        super().__init__()
        if rank is None: rank = dim
        self.mu = nn.Parameter(torch.randn(dim)*1e-3) if mu is None else nn.Parameter(mu.clone())
        self.W = nn.Parameter(torch.randn(dim, rank)*1e-3)
        self.phi = nn.Parameter(torch.randn(dim)*1e-3)
        self.register_buffer('data_init', torch.ones(1) if data_init else torch.zeros(1))

    def _data_init(self, x: torch.Tensor):
        q = self.W.shape[-1]
        mean = torch.mean(x, dim=0)
        self.mu.data = mean
        y = x-mean[None]
        U, s, _ = torch.linalg.svd(y.T, full_matrices=False)
        self.W.data = U[:, :q]
        self.phi.data[:] = -3
        self.data_init.data[:] = 0

    def freeze_scale(self, freeze: bool=True):
        """
        Freeze all parameters that impact log-determinant
        :param freeze: a boolean indicating whether to freeze (True) or unfreeze (False)
        """
        pass

    def rand_init(self, amnt: float):
        """
        Random initialization of the layer
        :param amnt: a float with the strength of the initialization
        """
        self.mu.data = torch.randn_like(self.mu)*amnt
        self.phi.data = torch.randn_like(self.phi)*amnt
        W = torch.randn_like(self.W)
        W = W@W.T
        W, _, _ = torch.svd(W)
        self.W.data = W[:, :self.W.shape[-1]]

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        phi = torch.exp(self.phi)
        return torch.linalg.slogdet(self.W@self.W.T + torch.diag(phi))[1]

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        phi = torch.exp(self.phi)
        return (self.W@self.W.T + torch.diag(phi))[None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_init[0] == 1: self._data_init(x)
        phi = torch.exp(self.phi)
        return phi[None]*x + (x@self.W)@self.W.T + self.mu[None]

    def jvp_forward(self, x: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = self.forward(x)
        phi = torch.exp(self.phi)
        J = phi[None]*f + (f@self.W)@self.W.T
        return y, J, self.logdet(x)

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        phi = torch.exp(self.phi)
        M = self.W@self.W.T + torch.diag_embed(phi)
        m = y-self.mu[None]
        return torch.linalg.solve(M, m.T).T

    def jvp_reverse(self, y: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the reverse pass as well as the inverse Jacobian-vector-product (iJVP)
        :param y: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector on which to evaluate the iJVP, a torch tensor with shape [N, dim] where N is the
                  number of points
        :return: the tuple (x, inv(J)f, logdet) where:
                    - y: a torch tensor with shape [N, dim], which are the transformed ys
                    - inv(J)f: a torch tensor with shape [N, dim], which is the iJVP with f
        """
        phi = torch.exp(self.phi)
        M = self.W @ self.W.T + torch.diag_embed(phi)
        m = y - self.mu[None]
        return torch.linalg.solve(M, m.T).T, torch.linalg.solve(M, f.T).T


class LogTransf(nn.Module):

    def __init__(self, precision: float=1e-5):
        super().__init__()
        self.prec = precision

    def freeze_scale(self, freeze: bool = True):
        """
        Freeze all parameters that impact log-determinant
        :param freeze: a boolean indicating whether to freeze (True) or unfreeze (False)
        """
        pass

    def rand_init(self, amnt: float):
        """
        Random initialization of the layer
        :param amnt: a float with the strength of the initialization
        """
        pass

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(-torch.log(x+self.prec), dim=-1)

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        return torch.diag_embed(self._s())[None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x+self.prec)

    def jvp_forward(self, x: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.log(x+self.prec), f/(x+self.prec), self.logdet(x)

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        return torch.exp(y)-self.prec

    def jvp_reverse(self, y: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the reverse pass as well as the inverse Jacobian-vector-product (iJVP)
        :param y: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector on which to evaluate the iJVP, a torch tensor with shape [N, dim] where N is the
                  number of points
        :return: the tuple (x, inv(J)f, logdet) where:
                    - y: a torch tensor with shape [N, dim], which are the transformed ys
                    - inv(J)f: a torch tensor with shape [N, dim], which is the iJVP with f
        """
        raise NotImplementedError


class PolarTransf(nn.Module):

    def __init__(self, precision: float=1e-6):
        super().__init__()
        self.precision = precision

    def freeze_scale(self, freeze: bool = True):
        """
        Freeze all parameters that impact log-determinant
        :param freeze: a boolean indicating whether to freeze (True) or unfreeze (False)
        """
        pass

    def rand_init(self, amnt: float):
        """
        Random initialization of the layer
        :param amnt: a float with the strength of the initialization
        """
        pass

    def _cart2polar(self, x):
        comp = x[:, :2]
        uncomp = x[:, 2:]
        r = torch.sum(x**2, dim=1) + self.precision
        theta = torch.arctan2(comp[:, 1], comp[:, 0])
        return torch.cat([r[:, None], theta[:, None], uncomp], dim=-1)

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        y = self._cart2polar(x)
        return .5*torch.log(y[:, 0])

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._cart2polar(x)

    def jvp_forward(self, x: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = self._cart2polar(x)

        fr = f[:, 0]
        fthet = f[:, 1]
        felse = f[:, 2:]
        # dr/dx = 2x
        jvp1 = 2*x[:, 0]*fr + 2*x[:, 1]*fthet
        # dth/dx = - y/r
        # dth/dy = x/r
        r = torch.sqrt(y[:, 0])
        jvp2 = -x[:, 1]*fr/r + x[:, 0]*fthet/r
        jvp = torch.cat([jvp1[:, None], jvp2[:, None], felse], dim=-1)
        return y, jvp, torch.log(y[:, 0])

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        r = torch.sqrt(torch.clamp(y[:, 0] - self.precision, 0))[:, None]
        theta = y[:, 1][:, None]
        oth = y[:, 2:]
        return torch.cat([r*torch.cos(theta), r*torch.sin(theta), oth], dim=-1)

    def jvp_reverse(self, y: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the reverse pass as well as the inverse Jacobian-vector-product (iJVP)
        :param y: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector on which to evaluate the iJVP, a torch tensor with shape [N, dim] where N is the
                  number of points
        :return: the tuple (x, inv(J)f, logdet) where:
                    - y: a torch tensor with shape [N, dim], which are the transformed ys
                    - inv(J)f: a torch tensor with shape [N, dim], which is the iJVP with f
        """
        raise NotImplementedError


class RevPolarTransf(nn.Module):

    def __init__(self, precision: float=1e-6):
        super().__init__()
        self.precision = precision

    def freeze_scale(self, freeze: bool = True):
        """
        Freeze all parameters that impact log-determinant
        :param freeze: a boolean indicating whether to freeze (True) or unfreeze (False)
        """
        pass

    def rand_init(self, amnt: float):
        """
        Random initialization of the layer
        :param amnt: a float with the strength of the initialization
        """
        pass

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        return -.5*torch.log(x[:, 0])

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.sqrt(torch.clamp(x[:, 0] - self.precision, 0))[:, None]
        theta = x[:, 1][:, None]
        oth = x[:, 2:]
        return torch.cat([r * torch.cos(theta), r * torch.sin(theta), oth], dim=-1)

    def jvp_forward(self, x: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = self.forward(x)

        r = torch.sqrt(torch.clamp(x[:, 0] - self.precision, 0))[:, None]
        theta = x[:, 1][:, None]

        fx = f[:, 0][:, None]
        fy = f[:, 1][:, None]
        felse = f[:, 2:]
        jvp1 = torch.cos(theta)*fx - r*torch.sin(theta)*fy
        jvp2 = torch.sin(theta)*fx + r*torch.cos(theta)*fy
        jvp = torch.cat([jvp1, jvp2, felse], dim=-1)
        return y, jvp, -torch.log(r[:, 0])

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        comp = y[:, :2]
        uncomp = y[:, 2:]
        r = torch.sum(y ** 2, dim=1) + self.precision
        theta = torch.arctan2(comp[:, 1], comp[:, 0])
        return torch.cat([r[:, None], theta[:, None], uncomp], dim=-1)

    def jvp_reverse(self, y: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the reverse pass as well as the inverse Jacobian-vector-product (iJVP)
        :param y: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector on which to evaluate the iJVP, a torch tensor with shape [N, dim] where N is the
                  number of points
        :return: the tuple (x, inv(J)f, logdet) where:
                    - y: a torch tensor with shape [N, dim], which are the transformed ys
                    - inv(J)f: a torch tensor with shape [N, dim], which is the iJVP with f
        """
        raise NotImplementedError


class ActNorm(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer('mu', nn.Parameter(torch.zeros(dim)))
        self.register_buffer('s', nn.Parameter(torch.zeros(dim)))
        self.register_buffer('newinit', torch.ones(1))

    def _newinit(self, x: torch.Tensor):
        self.mu.data = torch.mean(x, dim=0).data
        self.s.data = torch.log(torch.std(x, dim=0).data + 1e-3)
        self.newinit.data[:] = 0

    def freeze_scale(self, freeze: bool = True):
        """
        Freeze all parameters that impact log-determinant
        :param freeze: a boolean indicating whether to freeze (True) or unfreeze (False)
        """
        pass

    def rand_init(self, amnt: float):
        """
        Random initialization of the layer
        :param amnt: a float with the strength of the initialization
        """
        self.newinit.data[:] = 0
        self.s.data = 2*(torch.rand_like(self.s) - 1)*amnt
        self.mu.data = torch.randn_like(self.mu)*amnt

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(-self.s)[None]

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        return torch.diag_embed(torch.exp(-self.s))[None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.newinit[0] == 1: self._newinit(x)
        return torch.exp(-self.s)[None]*(x-self.mu[None])

    def jvp_forward(self, x: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward(x), torch.exp(-self.s)[None]*f, self.logdet(x)

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        if self.newinit[0] == 1: self._newinit(y)
        return torch.exp(self.s)[None]*y + self.mu[None]

    def jvp_reverse(self, y: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the reverse pass as well as the inverse Jacobian-vector-product (iJVP)
        :param y: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector on which to evaluate the iJVP, a torch tensor with shape [N, dim] where N is the
                  number of points
        :return: the tuple (x, inv(J)f, logdet) where:
                    - y: a torch tensor with shape [N, dim], which are the transformed ys
                    - inv(J)f: a torch tensor with shape [N, dim], which is the iJVP with f
        """
        return torch.exp(self.s)[None]*y + self.mu[None], torch.exp(self.s)[None]*f


class NFCompose(nn.Module):
    """
    Composes a number of normalizing-flow layers into one, similar to nn.Sequential
    """
    def __init__(self, *modules: nn.Module):
        super().__init__()
        self.transfs = nn.ModuleList(modules)

    def freeze_scale(self, freeze: bool=True):
        """
        Freeze all parameters that impact log-determinant
        :param freeze: a boolean indicating whether to freeze (True) or unfreeze (False)
        """
        for mod in self.transfs: mod.freeze_scale(freeze)

    def rand_init(self, amnt: float):
        """
        Random initialization of the layer
        :param amnt: a float with the strength of the initialization
        """
        for mod in self.transfs: mod.rand_init(amnt)

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        full_jac = torch.eye(x.shape[-1], device=x.device)[None]
        for mod in self.transfs:
            jac = mod.jacobian(x)
            full_jac = jac@full_jac
            x = mod.forward(x)
        return full_jac

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for mod in self.transfs:
            x = mod.forward(x)
        return x

    def jvp_forward(self, x: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ldets = 0
        for mod in self.transfs:
            x, f, ldet = mod.jvp_forward(x, f)
            ldets = ldets + ldet
        return x, f, ldets

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        for mod in self.transfs[::-1]:
            y = mod.reverse(y)
        return y

    def jvp_reverse(self, y: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for mod in self.transfs[::-1]:
            y, f = mod.jvp_reverse(y, f)
        return y, f


class Diffeo(nn.Module):

    def __init__(self, dim: int, rank: int=2, n_layers: int=4, K: int=15, add_log: bool=False,
                 MLP: bool=False, actnorm: bool=True, RFF: bool=False, affine_init: bool=False,
                 mu: torch.tensor=None, scale_free: bool=False, polar: bool=False):
        """
        Initializes a diffeomorphism, which is a normalizing flow with interleaved Affine transformations and
        AffineCoupling layers
        :param dim: the dimension of the data
        :param rank: the rank used in the Affine transformations (see the Affine object above)
        :param n_layers: number of Affine-Coupling-ReverseCoupling layers to use
        :param K: number of hidden units to use, either for the FFCoupling, MLP or RFF transformations
        :param add_log: whether to add an intial invertible log-transform on the data (a sort of preprocessing)
        :param MLP: if True, an AffineCoupling layer with an MLP will be used instead of the FFCoupling layer
        :param actnorm: whether the first layer is an invertible standardization of the data (a sort of preprocessing)
        :param RFF: if True, an AffineCoupling layer with an RFF will be used instead of the FFCoupling layer
        :param affine_init: whether to use a sort of initialization over the Affine transformation
        :param scale_free: whether to use a scale-free version of the couplings
        """
        super().__init__()

        layers = []
        if add_log: layers.append(LogTransf())
        if actnorm: layers.append(ActNorm(dim))
        layers.append(Affine(dim=dim, rank=dim, data_init=affine_init, mu=mu))
        for i in range(n_layers):
            layers.append(Affine(dim=dim, rank=rank))
            if MLP:
                layers.append(AffineCoupling(dim=dim, K=K))
                layers.append(AffineCoupling(dim=dim, K=K, reverse=True))
            elif RFF:
                layers.append(RFFCoupling(dim=dim, K=K))
                layers.append(RFFCoupling(dim=dim, K=K, reverse=True))
            else:
                layers.append(FFCoupling(dim=dim, K=K, R=10, scale_free=scale_free))
                layers.append(FFCoupling(dim=dim, K=K, R=10, reverse=True, scale_free=scale_free))

        if polar:
            layers.append(PolarTransf())
            layers.append(FFCoupling(dim=dim, K=2, R=1000, split_dims=[0]))
            layers.append(RevPolarTransf())

        self.transf = NFCompose(*layers)

    def freeze_scale(self, freeze: bool=True):
        """
        Freeze all parameters that impact log-determinant
        :param freeze: a boolean indicating whether to freeze (True) or unfreeze (False)
        """
        self.transf.freeze_scale(freeze)

    def rand_init(self, amnt: float):
        """
        Random initialization of the layer
        :param amnt: a float with the strength of the initialization
        """
        self.transf.rand_init(amnt)

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        return self.transf.jacobian(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat, shape = self._flatten(x)
        y_flat = self.transf.forward(x_flat)
        return self._unflatten(y_flat, shape)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y_flat, shape = self._flatten(y)
        x_flat = self.transf.reverse(y_flat)
        return self._unflatten(x_flat, shape)
    
    # def inverse(self, y: torch.Tensor) -> torch.Tensor:
    #     """
    #     Calculates the reverse transformation of the normalizing flow
    #     :param y: inputs as a torch tensor with shape [N, dim]
    #     :return: the transformed inputs, a tensor with shape [N, dim]
    #     """
    #     return self.transf.reverse(y)

    def jvp_forward(self, x: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the forward transformation of the normalizing flow, plus a JVP and the log-determinant
        :param x: inputs as a torch tensor with shape [N, dim]
        :param f: vectors whose JVP should be calculated, a torch tensor with shape [N, dim]
        :return: - the transformed inputs, a tensor with shape [N, dim]
                 - the JVPs of the normalizing flow on the vectors in f, a tensor with shape [N, dim]
                 - the log-determinants of the normalizing flows on x, a tensor with shape [N]
        """
        return self.transf.jvp_forward(x, f)

    def jvp_reverse(self, y: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the reverse pass as well as the inverse Jacobian-vector-product (iJVP)
        :param y: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector on which to evaluate the iJVP, a torch tensor with shape [N, dim] where N is the
                  number of points
        :return: the tuple (x, inv(J)f, logdet) where:
                    - y: a torch tensor with shape [N, dim], which are the transformed ys
                    - inv(J)f: a torch tensor with shape [N, dim], which is the iJVP with f
        """
        return self.transf.jvp_reverse(y, f)
    
    def _flatten(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        shape = x.shape
        return x.view(-1, x.shape[-1]), shape

    def _unflatten(self, x: torch.Tensor, shape: Tuple[int, int, int]) -> torch.Tensor:
        return x.view(*shape)