import os

import numba as nb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import LogNorm

rho_w = 1000.0
rhow_a = 1.2
g = 9.81
nu = 1.80e-5 / rhow_a

@nb.njit
def G(R, tau_R, r, tau_r, eps, R_l):
    v_k = (nu * eps)**0.25
    tau_k = (nu / eps)**0.5
    eta = (nu**3 / eps)**0.25
    a_o = (11 + 7*R_l) / (205 + R_l)
    a_og = a_o + np.pi / 8.0 * (g * tau_k / v_k)**2
    F = 20.115 * (a_og / R_l)**0.5
    St_R = tau_R / tau_k
    St_r = tau_r / tau_k
    St = np.maximum(St_R, St_r)
    C_1 = -.1988*St**4 + 1.5275*St**3 - 4.2942*St**2 + 5.3406*St / (g * tau_k / v_k)**(.1886*np.exp(20.306/R_l))
    r_c = (eta * np.abs(St_R - St_r) * F)**0.5

    return ((eta**2 + r_c**2) / ((R+r)**2 + r_c**2))**(C_1/2)

@nb.njit
def Phi(tau_R, tau_r, c, e):
    vr = tau_r * g
    vR = tau_R * g
    return (1 / (vr/e - 1/tau_r - 1/c) - 1 / (vR/e + 1/tau_R + 1/c)) *\
          (vR - vr) / (2*e * ((vR - vr)/e + 1/tau_R + 1/tau_r)**2) +\
          (4 / ((vr/e)**2 - (1/tau_r + 1/c)**2) - 1 / (vr/e + 1/tau_r + 1/c)**2 - 1 / (vr/e - 1/tau_r - 1/c)**2) *\
          vr / (2*e * (1/tau_R - 1/c + (1/tau_r + 1/c) * vR/vr)) +\
          (2*e / (vR/e + 1/tau_R + 1/c) - 2*e / ((vr/e - 1/tau_r - 1/c)) -\
           vR / (vR/e + 1/tau_R + 1/c)**2 + vr / (vr/e - 1/tau_r - 1/c)**2) *\
           1 / (2*e * ((vR - vr)/e + 1/tau_R + 1/tau_r))

@nb.njit
def f2(R, temp_beta, L_e):
    return 0.5/temp_beta * ((1 + temp_beta) * np.exp(-2*R/(1 + temp_beta)/L_e) 
                            - (1 - temp_beta) * np.exp(-2*R/(1 - temp_beta)/L_e))

@nb.njit
def t3(R, tau_r, tau_R, up, b1, b2, c1, c2, d1, d2, e1, e2, temp_beta, L_e):
    return up**2 * f2(R, temp_beta, L_e) / tau_r / tau_R * (b1*d1*Phi(tau_R, tau_r, c1, e1) 
                            - b1*d2*Phi(tau_R, tau_r, c1, e2) 
                            - b2*d1*Phi(tau_R, tau_r, c2, e1) 
                            + b2*d2*Phi(tau_R, tau_r, c2, e2))

@nb.njit
def Psi(tau_r, c, e):
    return 1 / (1/tau_r + 1/c + tau_r*g/e) - tau_r * g / (2*e * (1/tau_r + 1/c + tau_r*g/e)**2)

@nb.njit
def t12(tau_r, up, b1, b2, c1, c2, d1, d2, e1, e2):
    return up**2 / tau_r * (b1*d1*Psi(tau_r, c1, e1) 
                            - b1*d2*Psi(tau_r, c1, e2) 
                            - b2*d1*Psi(tau_r, c2, e1) 
                            + b2*d2*Psi(tau_r, c2, e2))    

@nb.njit
def sigma2(r, tau_r, R, tau_R, eps, R_l):
    v_k = (nu * eps)**0.25
    up = R_l**0.5 / 15**0.25 * v_k
    T_L = up**2 / eps
    L_e = 0.5 * up**3 / eps
    a_o = (11 + 7*R_l) / (205 + R_l)
    tau_T = (2*R_l/15**0.5/a_o)**0.5 * v_k
    l = up * (15*nu / eps)**0.5

    z = tau_T / T_L
    beta = 2**0.5 * l / L_e

    temp_z = (1 - 2*z**2)**0.5
    b1 = (1 + temp_z) / (2 * temp_z)
    b2 = (1 - temp_z) / (2 * temp_z)
    c1 = (1 + temp_z) * T_L / 2
    c2 = (1 - temp_z) * T_L / 2
    
    temp_beta = (1 - 2*beta**2)**0.5
    d1 = (1 + temp_beta) / (2 * temp_beta)
    d2 = (1 - temp_beta) / (2 * temp_beta)
    e1 = (1 + temp_beta) * L_e / 2
    e2 = (1 - temp_beta) * L_e / 2

    t1 = t12(tau_r, up, b1, b2, c1, c2, d1, d2, e1, e2)
    t2 = t12(tau_R, up, b1, b2, c1, c2, d1, d2, e1, e2)

    return t1 + t2 - 2 * t3(r+R, tau_r, tau_R, up, b1, b2, c1, c2, d1, d2, e1, e2, temp_beta, L_e)

@nb.njit
def T_L(eps, R_l):
    """Lagrangian integral timescale of the turbulent air motion.

    Same definition as used inside ``sigma2``: T_L = up**2 / eps, where the
    rms turbulent velocity up is set by the Taylor-microscale Reynolds number.
    """
    v_k = (nu * eps)**0.25
    up = R_l**0.5 / 15**0.25 * v_k
    return up**2 / eps

@nb.njit
def tau_next(r, tau_old):
    Re_p = 2 * r * tau_old * g / nu
    return 2.0 / 9.0 * rho_w / rhow_a * r**2 / nu / (1.0 + 0.15*Re_p**0.687)

@nb.njit
def tau(r):
    tau_stokes = 2.0 / 9.0 * rho_w / rhow_a * r**2 * g / nu
    return tau_next(r, tau_next(r, tau_next(r, tau_next(r, tau_stokes))))

@nb.njit
def w(R, tau_R, r, tau_r, eps, R_l):
    return np.sqrt(2.0 / np.pi) * (sigma2(R, tau_R, r, tau_r, eps, R_l) + np.pi / 8.0 * ((tau_R - tau_r)*g)**2)**0.5

@nb.njit
def K(r1, r2, eps=10e-4, R_l=75.0):
    R = np.maximum(r1, r2)
    r = np.minimum(r1, r2)
    tau_R = tau(R)
    tau_r = tau(r)
    return 2 * np.pi * (R+r)**2 * w(R, tau_R, r, tau_r, eps, R_l) * G(R, tau_R, r, tau_r, eps, R_l)

if __name__ == "__main__":
    r = np.linspace(5e-6, 6e-5, 100)
    R = np.meshgrid(r, r)[0]
    eps = 100e-4
    R_l = 75.0

    K_vals = K(R, R.T, eps, R_l)

    plt.pcolormesh(r*1e6, r*1e6, K_vals*1e6, norm=LogNorm())
    plt.colorbar(label='K [cm$^3$/s]')
    plt.savefig('K_Ayala_theory_eps=100.png')
