# scalable_cellfree_sim.py
# Full Python simulation framework implementing the algorithms in
# "Scalable Cell-Free Massive MIMO" (Bjornson et al.) — corrected and aligned with the paper.
# Implementations:
#  - MMSE channel estimation (eq (3))
#  - Pilot assignment & 3-step access (Sec V-A) — master AP always serves; neighbors follow paper rule
#  - P-MMSE combining (eq (23)-(24))
#  - LP-MMSE combining (eq (29))
#  - MR combining/precoding
#  - UL/DL SE (use-and-forget / hardening bound approximations)
#  - DL distributed power allocation (eq (43))
#
# Notes about the corrections from the earlier draft:
#  - The initial-access procedure processes UEs in descending order of their best large-scale
#    coefficient (max_l beta_{k,l}) to avoid eviction cycles.
#  - A robust re-assignment loop handles evicted UEs until convergence (with a safety limit).
#
# Requirements: numpy, scipy
# Run: python scalable_cellfree_sim.py

import numpy as np
from numpy.linalg import pinv, norm
from scipy.linalg import block_diag
import math

# ---------------------------
# Utilities & small helpers
# ---------------------------
def conjT(x):
    return x.conj().T

# ---------------------------
# Channel & network generation
# ---------------------------
def generate_network(L=10, K=20, N=2, area_side=2000.0, seed=0):
    """
    Generate AP and UE coordinates and large-scale correlation matrices R_{k,l}.
    Returns pos_AP, pos_UE, R (dict (k,l)->NxN), beta (K,L).
    """
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

# ---------------------------
# Pilot sets and MMSE estimation
# ---------------------------
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

# ---------------------------
# Initial access & cluster formation (3-step) - Sec V-A
# ---------------------------
def three_step_access(new_k, R, beta_new, AP_states, Psi_map, tau_set, neighbor_map):
    """
    Implements Steps 1-3 (Sec V-A) with Master AP always serving the new UE (replace if needed).
    Returns master, tau, evicted list (evicted entries are (ue, ap, tau)).
    """
    master = int(np.argmax(beta_new))
    # Step2: pick tau minimizing trace(Psi) at master
    traces = []
    for t in tau_set:
        Psi = Psi_map.get((master, t))
        traces.append(np.real(np.trace(Psi)) if Psi is not None else np.inf)
    tau = int(np.argmin(traces))

    evicted = []

    # Master MUST serve new UE on pilot tau, evict if needed
    curr = AP_states[master]['pilot2ue'].get(tau, None)
    if curr is not None and curr != new_k:
        evicted.append((curr, master, tau))
    AP_states[master]['pilot2ue'][tau] = new_k
    AP_states[master]['pilot2beta'][tau] = beta_new[master]

    # Neighbors: serve only if free OR new beta stronger
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

# ---------------------------
# Combine/precoding functions
# ---------------------------
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

# ---------------------------
# P-MMSE combining (centralized partial MMSE) - eq (23)-(24)
# ---------------------------
def P_MMSE_combiner(pk, Pk_indices, Dk, hhat_dict, C_dict, p_dict, sigma2_ul, hhat_k):
    M = Dk.shape[0]
    S = np.zeros((M,M), dtype=complex)
    for i in Pk_indices:
        hi = hhat_dict[i]
        S += p_dict[i] * (Dk @ np.outer(hi, conjT(hi)) @ Dk)
    Zp = Dk @ (sum(p_dict[i] * C_dict[i] for i in Pk_indices) + sigma2_ul * np.eye(M)) @ Dk
    S_total = S + Zp
    v = pk * (pinv(S_total) @ (Dk @ hhat_k))
    return v

# ---------------------------
# UL and DL SE computations (approximate via Monte-Carlo)
# ---------------------------
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
        return 1e9
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
        return 1e9
    return np.real(numer/denom)

# ---------------------------
# DL precoder from UL combiner (UL-DL duality eq (36))
# ---------------------------
def ul_to_dl_precoder(vi, Di):
    denom = np.real(conjT(vi) @ Di @ vi)
    if denom <= 0:
        denom = np.real(conjT(vi) @ vi)
    wbar = vi / (math.sqrt(denom) + 1e-12)
    wbar = wbar / (norm(wbar) + 1e-12)
    return wbar

# ---------------------------
# DL distributed power allocation (43)
# ---------------------------
def distribute_rho_at_AP(rho_ap, Dl_list, beta_l):
    denom = np.sum(np.sqrt(np.maximum(0, beta_l[Dl_list])))
    rho_alloc = {}
    for k in Dl_list:
        if denom <= 0:
            rho_alloc[k] = 0.0
        else:
            rho_alloc[k] = rho_ap * np.sqrt(max(0, beta_l[k])) / denom
    return rho_alloc

# ---------------------------
# Main demo + unit tests (corrected initial-access ordering + robust fallback)
# ---------------------------
def unit_tests_and_demo():
    print("Running unit tests and a small demo Monte-Carlo to validate implementation...")
    L = 8
    K = 16
    N = 2
    tau_p = 4
    sigma2_ul = 1e-3
    sigma2_dl = 1e-3
    P_ul = 0.1
    rho_ap = 0.2
    seed = 7

    pos_AP, pos_UE, R, beta = generate_network(L=L, K=K, N=N, seed=seed)
    St = form_pilot_sets(K, tau_p)
    p_dict = {i: P_ul for i in range(K)}
    p_vec = np.array([p_dict[i] for i in range(K)])

    Psi_map = {}
    for l in range(L):
        Psi_l = compute_Psi_for_AP(R, St, l, tau_p, p_dict, sigma2_ul)
        for t, mat in Psi_l.items():
            Psi_map[(l,t)] = mat

    AP_states = [{'pilot2ue': {t: None for t in range(tau_p)}, 'pilot2beta': {}} for _ in range(L)]

    neighbor_map = {}
    for l in range(L):
        dists = np.linalg.norm(pos_AP - pos_AP[l], axis=1)
        idxs = np.argsort(dists).tolist()
        neighbor_map[l] = [i for i in idxs if i != l][:3]

    # --- CORRECTION: process UEs in descending order of their best beta (strongest first)
    ue_order = np.argsort(-np.max(beta, axis=1))
    for k in ue_order:
        beta_k = beta[k, :]
        master, tau_assigned, evicted = three_step_access(k, R, beta_k, AP_states, Psi_map, list(range(tau_p)), neighbor_map)
        # one immediate retry for any evicted UE (rare with sorted order)
        for (ev_u, ev_ap, ev_tau) in evicted:
            beta_ev = beta[ev_u, :]
            three_step_access(ev_u, R, beta_ev, AP_states, Psi_map, list(range(tau_p)), neighbor_map)

    # Build Dl lists
    Dl_per_AP = []
    for l in range(L):
        served = [u for t,u in AP_states[l]['pilot2ue'].items() if u is not None]
        Dl_per_AP.append(served)

    # Robust re-assignment loop for any still-missing UEs:
    served_union = set()
    for lst in Dl_per_AP:
        served_union.update(lst)

    missing = [k for k in range(K) if k not in served_union]
    max_rounds = K * 5
    round_idx = 0
    while missing and round_idx < max_rounds:
        next_missing_set = set()
        for k in missing:
            beta_k = beta[k, :]
            master, tau_assigned, evicted = three_step_access(k, R, beta_k, AP_states, Psi_map, list(range(tau_p)), neighbor_map)
            # Any evicted UEs become candidates to reassign
            for (ev_u, ev_ap, ev_tau) in evicted:
                if ev_u != k:
                    next_missing_set.add(ev_u)
        # rebuild Dl and missing
        Dl_per_AP = []
        for l in range(L):
            served = [u for t,u in AP_states[l]['pilot2ue'].items() if u is not None]
            Dl_per_AP.append(served)
        served_union = set()
        for lst in Dl_per_AP:
            served_union.update(lst)
        missing = [kk for kk in range(K) if kk not in served_union]
        # prepare next round: include evicted UEs discovered
        # ensure we include those detected as evicted to attempt to reassign
        for ev in list(next_missing_set):
            if ev not in missing:
                missing.append(ev)
        round_idx += 1

    # Final forced fallback (should rarely be needed): forcibly assign any remaining missing UEs to master by overwriting
    if missing:
        print("Fallback: some UEs are unserved after reassign attempts. Forcibly assigning them to their master AP (overwriting current occupant).")
        for k in missing:
            beta_k = beta[k, :]
            master = int(np.argmax(beta_k))
            traces = []
            for t in range(tau_p):
                traces.append(np.real(np.trace(Psi_map[(master,t)])))
            tau = int(np.argmin(traces))
            AP_states[master]['pilot2ue'][tau] = k
            AP_states[master]['pilot2beta'][tau] = beta_k[master]
        # rebuild Dl_per_AP
        Dl_per_AP = []
        for l in range(L):
            served = [u for t,u in AP_states[l]['pilot2ue'].items() if u is not None]
            Dl_per_AP.append(served)
        served_union = set()
        for lst in Dl_per_AP:
            served_union.update(lst)

    if len(served_union) < K:
        missing_final = [k for k in range(K) if k not in served_union]
        print(f"[Warning] Even after forced assignment, {len(missing_final)} UEs remain unserved: {missing_final}")
        print("Continuing simulation: unserved UEs will have Dk=0 and SE=0.\n")

    # --- Continue with MMSE estimation and rest of demo (unchanged) ---
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

    ypilot = {}
    for l in range(L):
        for t in range(tau_p):
            s = np.zeros((N,), dtype=complex)
            for i in St[t]:
                s += np.sqrt(tau_p * p_dict[i]) * h[(i,l)]
            n = (rng.normal(size=(N,)) + 1j * rng.normal(size=(N,))) / np.sqrt(2) * np.sqrt(sigma2_ul)
            ypilot[(t,l)] = s + n

    hhat = {}
    C_error = {}
    for l in range(L):
        for t in range(tau_p):
            Psi_tl = Psi_map[(l, t)]
            for k in St[t]:
                Rkl = R[(k,l)]
                y = ypilot[(t,l)]
                hhat[(k,l)] = mmse_estimate_from_pilot(y, Rkl, Psi_tl, p_dict[k], tau_p)
                A = np.sqrt(p_dict[k]*tau_p) * Rkl @ pinv(Psi_tl)
                Ck = Rkl - A @ (np.sqrt(p_dict[k]*tau_p) * Rkl)
                Ck = (Ck + conjT(Ck))/2
                C_error[(k,l)] = Ck

    M = L*N
    def stack_h_k(k, use_hat=False):
        vec = np.zeros((M,), dtype=complex)
        for l in range(L):
            v = hhat[(k,l)] if use_hat else h[(k,l)]
            vec[l*N:(l+1)*N] = v
        return vec

    hhat_collective = {k: stack_h_k(k, use_hat=True) for k in range(K)}
    h_collective = {k: stack_h_k(k, use_hat=False) for k in range(K)}

    C_collective = {}
    for i in range(K):
        blocks = [C_error[(i,l)] for l in range(L)]
        C_collective[i] = block_diag(*blocks)

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

    k0 = 3
    Pk = []
    Dk = Dk_dict[k0]
    for i in range(K):
        Di = Dk_dict[i]
        if np.any(np.abs(Dk @ Di) > 1e-12):
            Pk.append(i)

    hhat_dict = {i: hhat_collective[i] for i in range(K)}
    C_dict = {i: C_collective[i] for i in range(K)}
    p_dict = {i: p_dict[i] for i in range(K)}

    v_pmmse = P_MMSE_combiner(p_dict[k0], Pk, Dk, hhat_dict, C_dict, p_dict, sigma2_ul, hhat_collective[k0])

    v_lp_local = {}
    for l in range(L):
        if k0 not in Dl_per_AP[l]:
            v_lp_local[l] = np.zeros((N,), dtype=complex)
            continue
        Dl = Dl_per_AP[l]
        p_list = [p_dict[i] for i in Dl]
        hhat_list = [hhat[(i,l)] for i in Dl]
        C_list = [C_error[(i,l)] for i in Dl]
        vloc = LP_MMSE_combiner_at_AP(p_dict[k0], p_list, hhat_list, C_list, sigma2_ul, hhat[(k0,l)])
        v_lp_local[l] = vloc
    v_lp_collective = np.zeros((M,), dtype=complex)
    for l in range(L):
        v_lp_collective[l*N:(l+1)*N] = v_lp_local[l]

    v_mr_collective = hhat_collective[k0]

    assert v_pmmse.shape == (M,)
    assert v_lp_collective.shape == (M,)
    assert v_mr_collective.shape == (M,)

    print("Combiner shapes OK. Sizes: M =", M)

    num_mc = 200
    rng2 = np.random.default_rng(seed+2)
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
                z = (rng2.normal(size=(N,)) + 1j * rng2.normal(size=(N,))) / np.sqrt(2)
                big[l*N:(l+1)*N] = sqrtR @ z
            sample[k] = big
        channel_samples.append(sample)

    sinr_mr = compute_ul_sinr_use_and_forget(k0, v_mr_collective, Dk, p_vec, channel_samples, sigma2_ul)
    sinr_lp = compute_ul_sinr_use_and_forget(k0, v_lp_collective, Dk, p_vec, channel_samples, sigma2_ul)
    sinr_pmmse = compute_ul_sinr_use_and_forget(k0, v_pmmse, Dk, p_vec, channel_samples, sigma2_ul)

    print(f"UL SINR estimates for UE {k0}: MR={sinr_mr:.4f}, LP-MMSE={sinr_lp:.4f}, P-MMSE={sinr_pmmse:.4f}")

    tau_c = 200
    tau_u = 50
    uplink_SE_mr = (tau_u/tau_c) * math.log2(1+sinr_mr)
    uplink_SE_lp = (tau_u/tau_c) * math.log2(1+sinr_lp)
    uplink_SE_pmmse = (tau_u/tau_c) * math.log2(1+sinr_pmmse)
    print(f"UL SE (bps/Hz) MR={uplink_SE_mr:.4f}, LP-MMSE={uplink_SE_lp:.4f}, P-MMSE={uplink_SE_pmmse:.4f}")

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
    print(f"DL SINR (hardening bound) for UE {k0}: {dl_sinr_k0:.4f}, DL SE approx {dl_se:.4f} bps/Hz")

    print("Demo complete. You can now use this framework to run larger experiments similar to Section VI in the paper.")

if __name__ == "__main__":
    unit_tests_and_demo()
