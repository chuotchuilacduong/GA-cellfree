# scf_full_fixed_params.py
import numpy as np
from numpy.linalg import pinv, norm
from scipy.linalg import block_diag
import math

def conjT(x):
    return x.conj().T

def generate_network(L=10, K=20, N=2, area_side=200.0, seed=0):
    rng = np.random.default_rng(seed)
    pos_AP = rng.uniform(0, area_side, size=(L,2))
    pos_UE = rng.uniform(0, area_side, size=(K,2))
    alpha = 3.7
    d0 = 10.0
    C0 = 1.0
    R = {}
    beta = np.zeros((K, L))
    for k in range(K):
        for l in range(L):
            d = np.linalg.norm(pos_UE[k]-pos_AP[l]) + 1e-3
            large_scale = C0 * (d0 / (d + d0))**alpha
            rho = 0.5
            idx = np.arange(N)
            corr = rho**np.abs(np.subtract.outer(idx, idx))
            Rkl = large_scale * corr
            Rkl = (Rkl + Rkl.T.conj())/2
            R[(k,l)] = Rkl.astype(complex)
            beta[k,l] = np.real(np.trace(Rkl)/N)
    return pos_AP, pos_UE, R, beta

def form_pilot_sets(K, tau_p):
    St = {t: [] for t in range(tau_p)}
    for k in range(K):
        t = k % tau_p
        St[t].append(k)
    return St

def compute_Psi_for_AP(R, St, l, tau_p, p_dict, sigma2):
    N = next(iter(R.values())).shape[0]
    Psi = {}
    for t in range(tau_p):
        S = np.zeros((N,N), dtype=complex)
        for i in St.get(t, []):
            S += tau_p * p_dict[i] * R[(i,l)]
        S += sigma2 * np.eye(N)
        Psi[t] = S
    return Psi

def mmse_estimate_from_pilot(y_pilot_tl, Rkl, Psi_tl, pk, tau_p):
    return np.sqrt(pk * tau_p) * (Rkl @ pinv(Psi_tl) @ y_pilot_tl)

def three_step_access(new_k, R, beta_new, AP_states, Psi_map, tau_set, neighbor_map):
    master = int(np.argmax(beta_new))
    traces = []
    for t in tau_set:
        Psi = Psi_map.get((master, t))
        traces.append(np.real(np.trace(Psi)) if Psi is not None else np.inf)
    tau = int(np.argmin(traces))
    evicted = []
    curr = AP_states[master]['pilot2ue'].get(tau, None)
    if curr is not None and curr != new_k:
        evicted.append((curr, master, tau))
    AP_states[master]['pilot2ue'][tau] = new_k
    AP_states[master]['pilot2beta'][tau] = beta_new[master]
    for l in neighbor_map.get(master, []):
        curr = AP_states[l]['pilot2ue'].get(tau, None)
        if curr is None:
            AP_states[l]['pilot2ue'][tau] = new_k
            AP_states[l]['pilot2beta'][tau] = beta_new[l]
        else:
            beta_curr = AP_states[l]['pilot2beta'].get(tau, 0.0)
            if beta_new[l] > beta_curr and curr != new_k:
                evicted.append((curr, l, tau))
                AP_states[l]['pilot2ue'][tau] = new_k
                AP_states[l]['pilot2beta'][tau] = beta_new[l]
    return master, tau, evicted

def MR_combiner(hhat_kl):
    return hhat_kl

def LP_MMSE_combiner_at_AP(pk, p_list, hhat_list, C_list, sigma2_ul, hhat_kl):
    N = hhat_kl.shape[0]
    S = np.zeros((N,N), dtype=complex)
    for pi, hhat_il, Ci_l in zip(p_list, hhat_list, C_list):
        S += pi * (np.outer(hhat_il, conjT(hhat_il)) + Ci_l)
    S += sigma2_ul * np.eye(N)
    v = pk * (pinv(S) @ hhat_kl)
    return v

def build_Dk_from_Dl(Dl_per_AP, L, N, K):
    M = L * N
    Dk_dict = {}
    for k in range(K):
        blocks = []
        for l in range(L):
            served = k in Dl_per_AP[l]
            if served:
                blocks.append(np.eye(N, dtype=complex))
            else:
                blocks.append(np.zeros((N,N), dtype=complex))
        Dk_dict[k] = block_diag(*blocks)
    return Dk_dict

def P_MMSE_combiner(pk, Pk_indices, Dk, hhat_dict, C_dict, p_dict, sigma2_ul, hhat_k):
    M = Dk.shape[0]
    S = np.zeros((M,M), dtype=complex)
    for i in Pk_indices:
        hi = hhat_dict[i]
        S += p_dict[i] * (Dk @ np.outer(hi, conjT(hi)) @ Dk)
    sumC = np.zeros((M,M), dtype=complex)
    for i in Pk_indices:
        sumC += p_dict[i] * C_dict[i]
    Zp = Dk @ (sumC + sigma2_ul * np.eye(M)) @ Dk
    S_total = S + Zp
    v = pk * (pinv(S_total) @ (Dk @ hhat_k))
    return v

def compute_ul_sinr_use_and_forget(k, v_k, Dk, p_vec, channel_samples, sigma2_ul):
    num_mc = len(channel_samples)
    E_vhD_h = 0+0j
    E_vhD_hi_sq = np.zeros(len(p_vec), dtype=float)
    E_norm2 = 0.0
    for sample in channel_samples:
        hk = sample[k]
        E_vhD_h += conjT(v_k) @ (Dk @ hk)
        for i in range(len(p_vec)):
            hi = sample[i]
            val = conjT(v_k) @ (Dk @ hi)
            E_vhD_hi_sq[i] += np.abs(val)**2
        E_norm2 += np.real(conjT(Dk @ v_k) @ (Dk @ v_k))
    E_vhD_h /= num_mc
    E_vhD_hi_sq /= num_mc
    E_norm2 /= num_mc
    pk = p_vec[k]
    numer = pk * (np.abs(E_vhD_h)**2)
    denom_sum = 0.0
    for i in range(len(p_vec)):
        denom_sum += p_vec[i] * E_vhD_hi_sq[i]
    denom = denom_sum - pk * (np.abs(E_vhD_h)**2) + sigma2_ul * E_norm2
    if denom <= 0:
        return 1e-12
    return np.real(numer / denom)

def compute_dl_sinr_hardening(k, wbar_dict, rho_vec, Dk, channel_samples, sigma2_dl):
    num_mc = len(channel_samples)
    E_hDW = 0+0j
    K = len(rho_vec)
    E_hDiWi_sq = np.zeros(K, dtype=float)
    for sample in channel_samples:
        hk = sample[k]
        E_hDW += conjT(hk) @ (Dk @ wbar_dict[k])
        for i in range(K):
            wi = wbar_dict[i]
            val = conjT(hk) @ (Dk @ wi)
            E_hDiWi_sq[i] += np.abs(val)**2
    E_hDW /= num_mc
    E_hDiWi_sq /= num_mc
    numer = rho_vec[k] * (np.abs(E_hDW)**2)
    denom = np.sum(rho_vec * E_hDiWi_sq) - rho_vec[k] * (np.abs(E_hDW)**2) + sigma2_dl
    if denom <= 0:
        return 1e-12
    return np.real(numer/denom)

def ul_to_dl_precoder(vi, Di):
    denom = np.real(conjT(vi) @ Di @ vi)
    if denom <= 0:
        denom = np.real(conjT(vi) @ vi)
    wbar = vi / (math.sqrt(denom) + 1e-12)
    wbar = wbar / (norm(wbar) + 1e-12)
    return wbar

def ensure_all_served_and_recompute(K, L, N, tau_p, p_dict, sigma2_ul, sigma2_dl,
                                    pos_AP, R, AP_states, neighbor_map, seed,
                                    h):
    # rebuild Dl_per_AP from AP_states
    Dl_per_AP = []
    for l in range(L):
        served = [u for t,u in AP_states[l]['pilot2ue'].items() if u is not None]
        Dl_per_AP.append(served)

    # build St mapping now from AP_states (the actual pilots per AP)
    St = {t: [] for t in range(tau_p)}
    ue_pilot = {}
    for l in range(L):
        for t,u in AP_states[l]['pilot2ue'].items():
            if u is not None:
                ue_pilot[u] = t
    for ue,tau in ue_pilot.items():
        St[tau].append(ue)

    # recompute Psi_map based on new St
    Psi_map_new = {}
    for l in range(L):
        Psi_l = compute_Psi_for_AP(R, St, l, tau_p, p_dict, sigma2_ul)
        for t, mat in Psi_l.items():
            Psi_map_new[(l,t)] = mat

    # rebuild ypilot using h
    rng = np.random.default_rng(seed+5)
    ypilot = {}
    for l in range(L):
        for t in range(tau_p):
            s = np.zeros((N,), dtype=complex)
            for i in St.get(t, []):
                s += np.sqrt(tau_p * p_dict[i]) * h[(i,l)]
            n = (rng.normal(size=(N,)) + 1j * rng.normal(size=(N,))) / np.sqrt(2) * np.sqrt(sigma2_ul)
            ypilot[(t,l)] = s + n

    # compute hhat and C_error for all (k,l) in St
    hhat = {}
    C_error = {}
    for l in range(L):
        for t in range(tau_p):
            Psi_tl = Psi_map_new[(l,t)]
            for k in St.get(t, []):
                Rkl = R[(k,l)]
                y = ypilot[(t,l)]
                hhat[(k,l)] = mmse_estimate_from_pilot(y, Rkl, Psi_tl, p_dict[k], tau_p)
                A = np.sqrt(p_dict[k]*tau_p) * Rkl @ pinv(Psi_tl)
                Ck = Rkl - A @ (np.sqrt(p_dict[k]*tau_p) * Rkl)
                C_error[(k,l)] = (Ck + Ck.conj().T)/2

    return St, Psi_map_new, ypilot, hhat, C_error, Dl_per_AP

def unit_tests_and_demo():
    L = 8
    K = 16
    N = 2
    tau_p = 4

    # adjusted noise and power for visible SINR
    sigma2_ul = 1e-6
    sigma2_dl = 1e-6
    P_ul = 0.5
    rho_ap = 0.5
    seed = 7

    pos_AP, pos_UE, R, beta = generate_network(L=L, K=K, N=N, area_side=200.0, seed=seed)

    p_dict = {i: P_ul for i in range(K)}
    p_vec = np.array([p_dict[i] for i in range(K)])

    # temporary Psi init
    tmp_St = form_pilot_sets(K, tau_p)
    Psi_map = {}
    for l in range(L):
        Psi_l = compute_Psi_for_AP(R, tmp_St, l, tau_p, p_dict, sigma2_ul)
        for t, mat in Psi_l.items():
            Psi_map[(l,t)] = mat

    AP_states = [{'pilot2ue': {t: None for t in range(tau_p)}, 'pilot2beta': {}} for _ in range(L)]

    neighbor_map = {}
    for l in range(L):
        dists = np.linalg.norm(pos_AP - pos_AP[l], axis=1)
        idxs = np.argsort(dists).tolist()
        neighbor_map[l] = [i for i in idxs if i != l][:3]

    # generate small-scale fading h
    rng = np.random.default_rng(seed+1)
    h = {}
    for k in range(K):
        for l in range(L):
            Rkl = R[(k,l)]
            vals, vecs = np.linalg.eigh(Rkl)
            vals[vals < 0] = 0.0
            sqrtR = vecs @ np.diag(np.sqrt(vals)) @ conjT(vecs)
            z = (rng.normal(size=(N,)) + 1j * rng.normal(size=(N,))) / np.sqrt(2)
            h[(k,l)] = sqrtR @ z

    # three-step access
    ue_pilot = {}
    ue_master = {}

    ue_order = np.argsort(-np.max(beta, axis=1))
    for k in ue_order:
        beta_k = beta[k, :]
        master, tau_assigned, evicted = three_step_access(k, R, beta_k, AP_states, Psi_map, list(range(tau_p)), neighbor_map)
        ue_pilot[k] = tau_assigned
        ue_master[k] = master
        for (ev_u, ev_ap, ev_tau) in evicted:
            if ev_u != k:
                if ev_u in ue_pilot: del ue_pilot[ev_u]
                if ev_u in ue_master: del ue_master[ev_u]
                beta_ev = beta[ev_u, :]
                m2, tau2, ev2 = three_step_access(ev_u, R, beta_ev, AP_states, Psi_map, list(range(tau_p)), neighbor_map)
                ue_pilot[ev_u] = tau2
                ue_master[ev_u] = m2

    # build Dl_per_AP
    Dl_per_AP = []
    for l in range(L):
        served = [u for t,u in AP_states[l]['pilot2ue'].items() if u is not None]
        Dl_per_AP.append(served)

    # retry loop
    served_union = {u for lst in Dl_per_AP for u in lst}
    missing = [k for k in range(K) if k not in served_union]

    max_rounds = K * 5
    round_idx = 0
    while missing and round_idx < max_rounds:
        next_missing = set()
        for k in missing:
            beta_k = beta[k, :]
            master, tau_assigned, evicted = three_step_access(k, R, beta_k, AP_states, Psi_map, list(range(tau_p)), neighbor_map)
            ue_pilot[k] = tau_assigned
            ue_master[k] = master
            for (ev_u, ev_ap, ev_tau) in evicted:
                if ev_u != k:
                    if ev_u in ue_pilot: del ue_pilot[ev_u]
                    if ev_u in ue_master: del ue_master[ev_u]
                    next_missing.add(ev_u)
        Dl_per_AP = []
        for l in range(L):
            served = [u for t,u in AP_states[l]['pilot2ue'].items() if u is not None]
            Dl_per_AP.append(served)
        served_union = {u for lst in Dl_per_AP for u in lst}
        missing = [k for k in range(K) if k not in served_union]
        for ev in list(next_missing):
            if ev not in missing:
                missing.append(ev)
        round_idx += 1

    # enforce all served and recompute estimates
    St, Psi_map, ypilot, hhat, C_error, Dl_per_AP = ensure_all_served_and_recompute(
        K, L, N, tau_p, p_dict, sigma2_ul, sigma2_dl,
        pos_AP, R, AP_states, neighbor_map, seed, h
    )

    M = L * N
    def stack_h_k(k, use_hat=False):
        vec = np.zeros((M,), dtype=complex)
        for l in range(L):
            if use_hat:
                v = hhat.get((k,l), np.zeros((N,), dtype=complex))
            else:
                v = h[(k,l)]
            vec[l*N:(l+1)*N] = v
        return vec

    hhat_collective = {k: stack_h_k(k, use_hat=True) for k in range(K)}
    h_collective = {k: stack_h_k(k, use_hat=False) for k in range(K)}

    C_collective = {}
    for i in range(K):
        blocks = [C_error.get((i,l), np.zeros((N,N), dtype=complex)) for l in range(L)]
        C_collective[i] = block_diag(*blocks)

    Dk_dict = build_Dk_from_Dl(Dl_per_AP, L, N, K)

    def overlapping(Da, Db):
        return np.any(np.abs(Da @ Db) > 1e-12)

    k0 = 3
    Dk = Dk_dict[k0]

    # small debug prints to verify state (non-intrusive)
    print("DEBUG: number of APs serving k0:", sum([1 for l in range(L) if k0 in Dl_per_AP[l]]))
    print("DEBUG: ||hhat_collective[k0]|| =", np.linalg.norm(hhat_collective[k0]))
    print("DEBUG: ||h_collective[k0]|| =", np.linalg.norm(h_collective[k0]))

    Pk = []
    for i in range(K):
        if overlapping(Dk, Dk_dict[i]):
            Pk.append(i)

    hhat_dict = {i: hhat_collective[i] for i in range(K)}
    C_dict = {i: C_collective[i] for i in range(K)}
    v_pmmse = P_MMSE_combiner(p_dict[k0], Pk, Dk, hhat_dict, C_dict, p_dict, sigma2_ul, hhat_collective[k0])

    v_lp_local = {}
    for l in range(L):
        if k0 not in Dl_per_AP[l]:
            v_lp_local[l] = np.zeros((N,), dtype=complex)
            continue
        Dl = Dl_per_AP[l]
        p_list = [p_dict[i] for i in Dl]
        hhat_list = [hhat.get((i,l), np.zeros((N,), dtype=complex)) for i in Dl]
        C_list = [C_error.get((i,l), np.zeros((N,N), dtype=complex)) for i in Dl]
        vloc = LP_MMSE_combiner_at_AP(p_dict[k0], p_list, hhat_list, C_list, sigma2_ul, hhat.get((k0,l), np.zeros((N,), dtype=complex)))
        v_lp_local[l] = vloc
    v_lp_collective = np.zeros((M,), dtype=complex)
    for l in range(L):
        v_lp_collective[l*N:(l+1)*N] = v_lp_local[l]

    v_mr_collective = hhat_collective[k0]

    num_mc = 2000
    rng3 = np.random.default_rng(seed+3)
    channel_samples = []
    for mc in range(num_mc):
        sample = {}
        for k in range(K):
            big = np.zeros((M,), dtype=complex)
            for l in range(L):
                Rkl = R[(k,l)]
                vals, vecs = np.linalg.eigh(Rkl)
                vals[vals<0] = 0.0
                sqrtR = vecs @ np.diag(np.sqrt(vals)) @ conjT(vecs)
                z = (rng3.normal(size=(N,)) + 1j * rng3.normal(size=(N,))) / np.sqrt(2)
                big[l*N:(l+1)*N] = sqrtR @ z
            sample[k] = big
        channel_samples.append(sample)

    # extra debug before SINR
    print("DEBUG: Dl_per_AP:", Dl_per_AP)
    print("DEBUG: Nonzero in Dk:", np.count_nonzero(Dk))
    print("DEBUG: ||Dk||:", np.linalg.norm(Dk))

    sinr_mr = compute_ul_sinr_use_and_forget(k0, v_mr_collective, Dk, p_vec, channel_samples, sigma2_ul)
    sinr_lp = compute_ul_sinr_use_and_forget(k0, v_lp_collective, Dk, p_vec, channel_samples, sigma2_ul)
    sinr_pmmse = compute_ul_sinr_use_and_forget(k0, v_pmmse, Dk, p_vec, channel_samples, sigma2_ul)

    tau_c = 200
    tau_u = 50
    uplink_SE_mr = (tau_u/tau_c) * math.log2(1+sinr_mr)
    uplink_SE_lp = (tau_u/tau_c) * math.log2(1+sinr_lp)
    uplink_SE_pmmse = (tau_u/tau_c) * math.log2(1+sinr_pmmse)

    Di_dict = {i: Dk_dict[i] for i in range(K)}
    wbar_dict = {}
    rho_vec = np.zeros(K)
    for i in range(K):
        vi = hhat_collective[i]
        Di = Di_dict[i]
        wbar = ul_to_dl_precoder(vi, Di)
        wbar_dict[i] = wbar
        rho_vec[i] = rho_ap / tau_p

    dl_sinr_k0 = compute_dl_sinr_hardening(k0, wbar_dict, rho_vec, Dk, channel_samples, sigma2_dl)
    dl_se = ( (tau_c - tau_u - tau_p) / tau_c ) * math.log2(1 + dl_sinr_k0)

    print("UL SINR estimates for UE", k0, "MR={:.6e}, LP-MMSE={:.6e}, P-MMSE={:.6e}".format(sinr_mr, sinr_lp, sinr_pmmse))
    print("UL SE (bps/Hz) MR={:.6e}, LP-MMSE={:.6e}, P-MMSE={:.6e}".format(uplink_SE_mr, uplink_SE_lp, uplink_SE_pmmse))
    print("DL SINR (hardening bound) for UE", k0, "{:.6e}, DL SE approx {:.6e} bps/Hz".format(dl_sinr_k0, dl_se))

if __name__ == "__main__":
    unit_tests_and_demo()
