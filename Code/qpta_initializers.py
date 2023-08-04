"""
@author: poitr_
"""

import numpy as np
import scipy.linalg as la
import scipy.interpolate as spint
from scipy.integrate import cumulative_trapezoid
import torch
from pathlib import Path
from typing import Callable

Tensor = torch.Tensor

import os
current_dir = os.path.dirname(os.path.realpath('__file__'))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

def scale_rot2d(alpha: float, theta: float) -> np.ndarray:
    """2D rotation matrix."""
    return alpha * np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])


def _qpta_tanh_hh(alpha_LB: float = 1 + 1e-3,
                  alpha_UB: float = 10.0, ) -> Callable:
    _LB = 1 + 1e-3
    _UB = 10.0
    assert alpha_LB >= _LB
    assert alpha_UB <= _UB

    with np.load(str(Path(current_dir).joinpath('bifcurve_tanh.npz')),
                 allow_pickle=True) as f:
        alphas = f["alphas"].squeeze()
        thetas = f["thetas"].squeeze()

    LL = spint.interp1d(alphas, thetas, fill_value="extrapolate")

    def prob_alpha(a):
        return 1 - (4 * LL(a)) / np.pi

    A = np.linspace(alpha_LB, alpha_UB, num=50)
    PA = np.array([prob_alpha(a) for a in A])
    cdf = cumulative_trapezoid(PA, A, initial=0)
    cdf /= cdf[-1]
    quantile_fcn_alpha = spint.PchipInterpolator(cdf, A)

    def quantile_fcn_theta(a, u):
        pi = np.pi
        ll = LL(a)
        return np.piecewise(u,
                            [u <= 0.5, u > 0.5],
                            [lambda x: ll + x * 2 * (pi / 2 - 2 * ll),
                             lambda x: pi / 2 + ll + (2 * x - 1.0) * (pi / 2 - 2 * ll)]
                            )

    def init(shape: torch.Size) -> Tensor:
        if len(shape) != 2:
            raise ValueError(f"qpta initializer expects a 2D shape, got: {len(shape)}")
        n_rows, n_cols = shape[0], shape[1]

        if n_rows != n_cols or n_rows % 2 != 0:
            raise ValueError(f"qpta initializer expects a square, even shape, got: {shape}")

        nblocks = n_rows // 2
        us = np.random.uniform(0., 1., size=(nblocks,))
        alphas_hat = np.array([quantile_fcn_alpha(p) for p in np.random.uniform(0, 1, size=(nblocks,))])
        # torch.distributions.Uniform(alpha_LB, alpha_UB).sample(torch.Size([nblocks]))
        thetas_hat = np.zeros_like(alphas_hat)

        for i, (a, u) in enumerate(zip(alphas_hat, us)):
            thetas_hat[i] = quantile_fcn_theta(a, u)
        diags = [scale_rot2d(a, t) for a, t in zip(alphas_hat, thetas_hat)]
        return torch.tensor(la.block_diag(*diags))

    return init


def _qpta_gru_hh(alpha_LB: float = 1e-2,
                 alpha_UB: float = 4.0, ) -> Callable:
    _LB = 1e-2
    _UB = 4.0
    assert alpha_LB >= _LB
    assert alpha_UB <= _UB
    with np.load(str(Path(__file__).parent.joinpath('bifcurve_gru.npz')),
                 allow_pickle=True) as f:
        alphas = f["alphas"].squeeze()
        thetas = f["thetas"].squeeze()
    LL = spint.interp1d(alphas[:61], thetas[:61], fill_value="extrapolate")
    LR = spint.interp1d(alphas[61:141], thetas[61:141], fill_value="extrapolate")
    RL = spint.interp1d(alphas[140:191], thetas[140:191], fill_value="extrapolate")
    RR = spint.interp1d(alphas[191:], thetas[191:], fill_value="extrapolate")

    def offset(w):
        """
        Result of analytical calculation for the Hopf bifurcation.
        :param w:
        :return:
        """
        return -2. * np.cos(w) + np.sqrt(14. + 2. * np.cos(2. * w))

    def prob_alpha(a):
        return (LR(a) - LL(a) + RR(a) - RL(a)) / np.pi

    A = np.linspace(alpha_LB, alpha_UB, num=50)
    PA = np.array([prob_alpha(a) for a in A])
    cdf = cumulative_trapezoid(PA, A, initial=0)
    cdf /= cdf[-1]
    quantile_fcn_alpha = spint.PchipInterpolator(cdf, A)

    def quantile_fcn_theta(alpha, u):
        ll = LL(alpha)
        lr = LR(alpha)
        rl = RL(alpha)
        rr = RR(alpha)
        meas1 = lr - ll
        meas2 = rr - rl
        p1 = meas1 / (meas1 + meas2)
        p2 = meas2 / (meas1 + meas2)
        if u <= p1:
            return ll + u * (meas1 + meas2)
        else:
            return rl + (u - p1) / p2 * meas2

    def init(shape: torch.Size) -> Tensor:
        if len(shape) != 2:
            raise ValueError(f"qpta initializer expects a 2D shape, got: {len(shape)}")
        n_rows, n_cols = shape[0], shape[1]

        if n_rows != n_cols or n_rows % 2 != 0:
            raise ValueError(f"qpta initializer expects a square, even shape, got: {shape}")

        nblocks = n_rows // 2
        us = np.random.uniform(0., 1., size=(nblocks,))
        alphas_hat = np.array([quantile_fcn_alpha(p) for p in np.random.uniform(0, 1, size=(nblocks,))])
        thetas_hat = np.zeros_like(alphas_hat)
        alphas_offset = np.zeros_like(alphas_hat)

        for i, (a, u) in enumerate(zip(alphas_hat, us)):
            thetas_hat[i] = quantile_fcn_theta(a, u)
            alphas_offset[i] = offset(thetas_hat[i]) + a

        diags = [scale_rot2d(a, t) for a, t in zip(alphas_offset, thetas_hat)]
        return torch.tensor(la.block_diag(*diags))

    return init
