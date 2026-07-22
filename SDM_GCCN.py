import numpy as np
import pickle
import time
import multiprocessing
import numba as nb
from scipy.stats import lognorm, expon
from scipy.optimize import brentq
from K_Ayala import K as K_Ayala


# ─────────────────── Domain Configuration ────────────────────
Lz = 400.0
Lx_Ly_area = 1.0

n_cells = 16
dz = Lz / n_cells
dv = Lx_Ly_area * dz
total_volume = Lx_Ly_area * Lz
cloud_volume = Lx_Ly_area * 400.0
dt = 1.0
duration = 3600.0

# ─────────────────── Constants & Thermodynamics ────────────────────
g, R, Rd, Rv = 9.81, 8.31, 287.05, 461.5
Ma, Mw, Lv, rho_w = 2.90e-2, 0.01801528, 2.501e6, 1000.0
sigma, D_v, K_th = 0.072, 2.5e-5, 0.025
c_p, T0 = 1005.0, 273.15
rho_air = 1.2
mu_v = 1.80e-5
nu = mu_v / rho_air
kappa = 0.7

T_env, p0 = 300.0, 1.0e5
es0 = 610.94 * np.exp(17.625 * (T_env - T0) / (T_env - 30.11))
Q1 = g * Mw * Lv / (c_p * R * T_env ** 2) - g * Ma / (R * T_env)
Q2 = p0 * Ma / (es0 * Mw) + Mw * Lv ** 2 / (c_p * R * T_env ** 2)
G_thermo = 1.0 / (
    rho_w * R * T_env / (es0 * D_v * Mw) + Lv * rho_w / K_th / T_env * (Lv * Mw / (R * T_env) - 1)
)
aKel = 2.0 * sigma * Mw / (R * T_env * rho_w)

Na = 10e6
s_a, mu_a = 0.3, 1.0e-7

# Load GCCN parameters
Ng_arr, D_arr = np.loadtxt('params.csv', delimiter=',', dtype=str).T
D_arr = D_arr[1::3]
Ng_arr = Ng_arr[1::3]
mask = D_arr != ''
D_arr = D_arr[mask].astype(float)
Ng_arr = Ng_arr[mask].astype(float)
mask = D_arr < 10
D_arr = D_arr[mask]
Ng_arr = Ng_arr[mask]
D_arr *= 1e-6
Ng_arr *= 1e12

N_sd = 20200


@nb.njit
def w_updraft(t):
    if t < 800.0:
        return 1.0 * (1.0 - t / 800.0)
    return 0.0

def r_eq(rd, S_ratio=0.99):
    def f(r):
        return S_ratio - 1.0 - aKel / r + kappa * rd ** 3 / r ** 3
    return brentq(f, rd, rd * 100, xtol=1e-12, rtol=1e-10)

# ─────────────────── Physics Setup ────────────────────
@nb.njit
def v_t(r):
    c1 = 0.0902
    d0 = 9.06
    D = 2*r
    X = 4*rho_w*g*D**3 / (3*rho_air*nu**2)
    if X <= 0:
        return 0.0
    sqrt_X = np.sqrt(X)
    term1 = 1 + c1 * sqrt_X
    sqrt_term1 = np.sqrt(term1)
    b = 0.5*c1*sqrt_X / (sqrt_term1 - 1) / sqrt_term1
    a = d0**2/4*(sqrt_term1 - 1)**2 / X**b
    return a*nu**(1-2*b)*(4/3*g*(rho_w/rho_air - 1))**b*D**(3*b-1)

@nb.njit
def K_Hall(v1, v2):
    r1 = (3.0 * v1 / (4.0 * np.pi))**(1.0/3.0)
    r2 = (3.0 * v2 / (4.0 * np.pi))**(1.0/3.0)
    R = max(r1, r2)
    r = min(r1, r2)
    R_eff = max(R, 1e-9)
    r_eff = max(r, 1e-9)
    E = 1.0 / ((1.0 + (30.6e-6 / R_eff)**5) * (1.0 + (6.27e-6 / r_eff)**4))
    return np.pi * E * (r1 + r2)**2 * np.abs(v_t(R) - v_t(r))

@nb.njit
def K_Ayala_Hall(v1, v2):
    r1 = (3.0 * v1 / (4.0 * np.pi))**(1.0/3.0)
    r2 = (3.0 * v2 / (4.0 * np.pi))**(1.0/3.0)
    R = max(r1, r2)
    r = min(r1, r2)
    R_eff = max(R, 1e-9)
    r_eff = max(r, 1e-9)
    E = 1.0 / ((1.0 + (30.6e-6 / R_eff)**5) * (1.0 + (6.27e-6 / r_eff)**4))
    if (r1>1e-5) & (r2 > 1e-5) & (r1<6e-5) & (r2<6e-5): 
        return E*K_Ayala(r1, r2, eps=50e-4)
    return K_Hall(v1, v2)

@nb.njit
def cell_sdm(idxs, xi, v, v_d, dt, dv):
    n_s = len(idxs)
    if n_s < 2:
        return
    n_pair = n_s // 2
    
    pairs = np.argsort(np.random.rand(n_s))[:2 * n_pair]
    phi = np.random.rand(n_pair)
    
    p_ratio = n_s * (n_s - 1.0) / 2.0 / n_pair
    
    for alpha in range(n_pair):
        j_local = pairs[2 * alpha]
        k_local = pairs[2 * alpha + 1]
        
        j = idxs[j_local]
        k = idxs[k_local]
        
        if xi[j] <= 0 or xi[k] <= 0:
            continue
            
        # p_jk = K_Hall(v[j], v[k]) * dt / dv    # Hall kernel
        p_jk = K_Ayala_Hall(v[j], v[k]) * dt / dv    # Ayala kernel
        if xi[j] < xi[k]:
            j, k = k, j
            
        p_alpha = xi[j] * p_ratio * p_jk
        gamma = np.floor(p_alpha)
        if (p_alpha - gamma) > phi[alpha]:
            gamma += 1.0
            
        if gamma > 0:
            max_gamma = np.floor(xi[j] / xi[k])
            gamma = min(gamma, max_gamma)
            
            if xi[j] - gamma * xi[k] > 0:
                xi[j] -= gamma * xi[k]
                v[k] += gamma * v[j]
                v_d[k] += gamma * v_d[j]
            else:
                xi[j] = np.floor(xi[k] / 2.0)
                xi[k] -= xi[j]
                v[k] += gamma * v[j]
                v[j] = v[k]
                v_d[k] += gamma * v_d[j]
                v_d[j] = v_d[k]

@nb.njit
def sdm_step(z, xi, v, v_d, dt, dv, n_cells, dz, z_max):
    alive_mask = (xi > 0) & (z >= 0) & (z < z_max)
    
    n_total = len(z)
    cell_ids = -np.ones(n_total, dtype=np.int32)
    
    for i in range(n_total):
        if alive_mask[i]:
            cell_ids[i] = int(z[i] / dz)
            
    head = -np.ones(n_cells, dtype=np.int32)
    next_idx = -np.ones(n_total, dtype=np.int32)
    
    for i in range(n_total):
        c = cell_ids[i]
        if c >= 0 and c < n_cells:
            next_idx[i] = head[c]
            head[c] = i
            
    for c in range(n_cells):
        curr = head[c]
        count = 0
        while curr != -1:
            count += 1
            curr = next_idx[curr]
            
        if count < 2:
            continue
            
        idxs = np.empty(count, dtype=np.int32)
        curr = head[c]
        idx_pos = 0
        while curr != -1:
            idxs[idx_pos] = curr
            curr = next_idx[curr]
            idx_pos += 1
            
        cell_sdm(idxs, xi, v, v_d, dt, dv)

@nb.njit
def advect_step(z, v, xi, dt):
    C = 0.0
    R = 0.0
    n_total = len(z)
    for i in range(n_total):
        if xi[i] > 0:
            rad = (3.0 * v[i] / (4.0 * np.pi))**(1.0/3.0)
            vt = v_t(rad)
            z[i] -= vt * dt
            
            if z[i] < 0:
                if rad > 30e-6:
                    R += xi[i] * v[i] * rho_w
                else:
                    C += xi[i] * v[i] * rho_w
                xi[i] = 0.0
    # return sedimentation flux, split into cloud water and rain water.
    return C, R

@nb.njit
def cond_derivatives(t, S, xi, v, v_d, z, dz, n_cells, z_max, dv_cell):
    n_total = len(v)
    dvdt = np.zeros(n_total)
    sum_xi_dvdt = np.zeros(n_cells)
    
    for i in range(n_total):
        if xi[i] > 0 and 0 <= z[i] < z_max:
            c = int(z[i] / dz)
            if 0 <= c < n_cells:
                r = (3.0 * v[i] / (4.0 * np.pi))**(1.0/3.0)
                dvdt[i] = 4.0 * np.pi * r * G_thermo * (S[c] - 1.0 - aKel / r + kappa * v_d[i] / v[i])
                sum_xi_dvdt[c] += xi[i] * dvdt[i]
            
    dSdt = np.zeros(n_cells)
    w_up = w_updraft(t)
    for c in range(n_cells):
        dSdt[c] = Q1 * w_up - rho_w * Q2 * (sum_xi_dvdt[c] / dv_cell)
        
    return dSdt, dvdt

@nb.njit
def cond_step_adaptive(t_sim, dt_step, S_arr, xi, v, v_d, z, dz, n_cells, z_max, dv_cell):
    t = t_sim
    t_end = t_sim + dt_step
    
    rtol, atol = 1e-4, 1e-7
    min_dt = 1e-3
    max_dt = 1.0
    
    dt_sub = 0.1
    n_total = len(v)
    
    S = S_arr.copy()
    
    while t < t_end:
        if t + dt_sub > t_end:
            dt_sub = t_end - t
            
        # Heun's method
        dSdt1, dvdt1 = cond_derivatives(t, S, xi, v, v_d, z, dz, n_cells, z_max, dv_cell)
        
        S_euler = S + dSdt1 * dt_sub
        v_euler = v + dvdt1 * dt_sub
        
        dSdt2, dvdt2 = cond_derivatives(t + dt_sub, S_euler, xi, v_euler, v_d, z, dz, n_cells, z_max, dv_cell)
        
        S_heun = S + 0.5 * (dSdt1 + dSdt2) * dt_sub
        v_heun = v + 0.5 * (dvdt1 + dvdt2) * dt_sub
        
        err_S = 0.0
        for c in range(n_cells):
            e_S = np.abs(S_heun[c] - S_euler[c]) / (atol + rtol * np.abs(S_heun[c]))
            if e_S > err_S:
                err_S = e_S
                
        err_v = 0.0
        for i in range(n_total):
            if xi[i] > 0 and 0 <= z[i] < z_max:
                e = np.abs(v_heun[i] - v_euler[i]) / (atol + rtol * np.abs(v_heun[i]))
                if e > err_v:
                    err_v = e
        
        err = max(err_S, err_v)
        
        if err <= 1.0 or dt_sub <= min_dt:
            t += dt_sub
            S[:] = S_heun[:]
            v[:] = v_heun[:]
            
        factor = 0.9 * (err + 1e-12)**(-0.5)
        factor = min(max(factor, 0.2), 2.0)
        dt_sub = dt_sub * factor
        dt_sub = min(max(dt_sub, min_dt), max_dt)
        
    return S

def run_custom_sdm(seed, i_idx):
    np.random.seed(seed)

    Ng_val = Ng_arr[i_idx]
    D_val = D_arr[i_idx]
    
    p1_val = Na / (Na + Ng_val * D_val)
    p2_val = 1 - p1_val
    r_val_arr = np.linspace(1e-7, 1e-5, 10000)
    pdf_vals = np.where(r_val_arr<1e-6, p1_val*lognorm.pdf(r_val_arr, s_a, scale=mu_a), p2_val*expon.pdf(r_val_arr, scale=D_val/2)/2)
    norm = np.sum(pdf_vals*(r_val_arr[1]-r_val_arr[0]))
    
    def local_pdf(r_loc):
        return np.where(r_loc<1e-6, p1_val*lognorm.pdf(r_loc, s_a, scale=mu_a), p2_val*expon.pdf(r_loc, scale=D_val/2)/2)/norm

    r_d = 10**np.random.uniform(-7, -5, N_sd)
    r0 = np.array([r_eq(r_d_val, 0.99) for r_d_val in r_d])
    v_d = 4.0/3.0 * np.pi * r_d**3
    v = 4.0/3.0 * np.pi * r0**3
    xi = np.array([np.random.poisson((Na + np.exp(-2e-6/D_val)*Ng_val * D_val) * cloud_volume * local_pdf(r_d[j]) * np.log(1e2) * r_d[j] / N_sd) for j in range(N_sd)])
    z = np.random.uniform(Lz-400, Lz, N_sd)
    S = np.ones(n_cells)
    n_steps = int(duration / dt)
    
    C = 0.0
    R = 0.0
    
    for step in range(n_steps):
        t_sim = step * dt
        S = cond_step_adaptive(t_sim, dt, S, xi, v, v_d, z, dz, n_cells, Lz, dv)
        sdm_step(z, xi, v, v_d, dt, dv, n_cells, dz, Lz)
        C_step, R_step = advect_step(z, v, xi, dt)
        C += C_step
        R += R_step
        
    return {
        "multiplicity": xi.copy(),
        "volume": v.copy(),
        "z": z.copy(),
        "S_final": S,
        "C": C,
        "R": R
    }

import traceback

def worker(args):
    seed, i_idx = args
    try:
        res = run_custom_sdm(seed, i_idx)
        return res
    except Exception as e:
        print(f"Simulation {seed} failed: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    total_runs = 16
    
    R_arr = []
    for i_idx in range(len(D_arr)):
        start_time = time.time()
        
        args_list = [(seed, i_idx) for seed in range(total_runs)]
        with multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), 8)) as pool:
            results = pool.map(worker, args_list)
            
        end_time = time.time()
        
        R_vals = []
        
        for r in results:
            if r is not None:
                R_vals.append(r['R'])

        print(f"Ensemble {i_idx} / {len(D_arr)} completed in {end_time - start_time:.2f} seconds.")
        print(f'It had {np.mean(R_vals)} mm of rain.')
        R_arr.append(R_vals)

with open('R_loNa_eps50.pkl', 'wb') as f:
    pickle.dump(R_arr, f)
