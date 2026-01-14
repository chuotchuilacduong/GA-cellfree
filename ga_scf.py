import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm, toeplitz
from scipy.integrate import quad

# Hàm chuyển đổi dB sang công suất
def db2pow(db):
    return 10**(db/10)

# Hàm tạo ma trận tương quan không gian (Local Scattering Model)
def functionRlocalscattering(M, theta, ASDdeg, antennaSpacing=0.5):
    ASD = ASDdeg * np.pi / 180
    firstRow = np.zeros(M, dtype=complex)
    for col in range(M):
        distance = antennaSpacing * col
        # Tính tích phân thực và ảo
        def integrand_real(Delta):
            return np.cos(2*np.pi*distance*np.sin(theta+Delta)) * np.exp(-Delta**2/(2*ASD**2))/(np.sqrt(2*np.pi)*ASD)
        def integrand_imag(Delta):
            return np.sin(2*np.pi*distance*np.sin(theta+Delta)) * np.exp(-Delta**2/(2*ASD**2))/(np.sqrt(2*np.pi)*ASD)
        
        val_real, _ = quad(integrand_real, -20*ASD, 20*ASD)
        val_imag, _ = quad(integrand_imag, -20*ASD, 20*ASD)
        firstRow[col] = val_real + 1j*val_imag
        
    R = toeplitz(firstRow)
    return R

# Hàm tạo thiết lập mạng (AP, UE, kênh truyền)
def generateSetup(L, K, N, tau_p):
    squareLength = 2000
    B_bw = 20e6
    noiseFigure = 7
    noiseVariancedBm = -174 + 10*np.log10(B_bw) + noiseFigure
    alpha = 3.76
    sigma_sf = 10
    constantTerm = -35.3
    antennaSpacing = 0.5
    ASDdeg = 20
    threshold = -40

    # Vị trí ngẫu nhiên
    APpositions = (np.random.rand(L) + 1j*np.random.rand(L)) * squareLength
    UEpositions = (np.random.rand(K) + 1j*np.random.rand(K)) * squareLength

    # Xử lý wrap-around để mô phỏng mạng vô hạn
    wrapHorizontal = np.array([-squareLength, 0, squareLength])
    wrapVertical = wrapHorizontal
    wrapLocations = (wrapHorizontal[:, None] + 1j*wrapVertical).flatten()
    
    APpositionsWrapped = APpositions[:, None] + wrapLocations
    
    dist_matrix = np.zeros((L, K))
    gainOverNoisedB = np.zeros((L, K))
    R = np.zeros((N, N, L, K), dtype=complex)
    D = np.zeros((L, K))
    pilotIndex = np.zeros(K, dtype=int)
    masterAPs = np.zeros(K, dtype=int)

    for k in range(K):
        # Tính khoảng cách tới AP gần nhất (xét cả wrap-around)
        dists = np.abs(APpositionsWrapped - UEpositions[k])
        min_dist_indices = np.argmin(dists, axis=1)
        distances = np.sqrt(10**2 + np.take_along_axis(dists, min_dist_indices[:, None], axis=1).flatten()**2)
        dist_matrix[:, k] = distances
        
        # Tính Large-scale fading (Pathloss + Shadowing)
        gainOverNoisedB[:, k] = constantTerm - alpha*10*np.log10(distances) + sigma_sf*np.random.randn(L) - noiseVariancedBm
        
        # Chọn Master AP
        master = np.argmax(gainOverNoisedB[:, k])
        D[master, k] = 1
        masterAPs[k] = master
        
        # Phân bổ Pilot (Pilot Assignment)
        if k < tau_p:
            pilotIndex[k] = k
        else:
            pilotInterference = np.zeros(tau_p)
            for t in range(tau_p):
                # --- KHẮC PHỤC LỖI INDEX ERROR TẠI ĐÂY ---
                # Chỉ lấy gain của các user từ 0 đến k-1 ([:k]) sau đó mới lọc theo pilot t
                gains_of_existing_users = gainOverNoisedB[master, :k]
                mask_pilot_t = (pilotIndex[:k] == t)
                pilotInterference[t] = np.sum(db2pow(gains_of_existing_users[mask_pilot_t]))
                # -----------------------------------------
            pilotIndex[k] = np.argmin(pilotInterference)
            
        # Tạo ma trận tương quan không gian R
        for l in range(L):
            best_pos = min_dist_indices[l]
            angletoUE = np.angle(UEpositions[k] - APpositionsWrapped[l, best_pos])
            R[:, :, l, k] = db2pow(gainOverNoisedB[l, k]) * functionRlocalscattering(N, angletoUE, ASDdeg, antennaSpacing)

    # AP Selection (Dynamic Cooperation Cluster)
    for l in range(L):
        for t in range(tau_p):
            pilotUEs = np.where(pilotIndex == t)[0]
            # Nếu AP chưa phục vụ ai dùng pilot t (tức là chưa là Master AP của ai dùng pilot t)
            if np.sum(D[l, pilotUEs]) == 0 and len(pilotUEs) > 0:
                gains = gainOverNoisedB[l, pilotUEs]
                best_idx = np.argmax(gains)
                bestUE = pilotUEs[best_idx]
                # Kiểm tra điều kiện ngưỡng để tham gia phục vụ
                if gains[best_idx] - gainOverNoisedB[masterAPs[bestUE], bestUE] >= threshold:
                    D[l, bestUE] = 1
                    
    return gainOverNoisedB, R, pilotIndex, D

# Hàm ước lượng kênh (MMSE Estimation)
def functionChannelEstimates(R, nbrOfRealizations, L, K, N, tau_p, pilotIndex, p):
    H = (np.random.randn(L*N, nbrOfRealizations, K) + 1j*np.random.randn(L*N, nbrOfRealizations, K)) * np.sqrt(0.5)
    
    # Áp dụng tương quan không gian vào kênh truyền
    for l in range(L):
        for k in range(K):
            Rsqrt = sqrtm(R[:, :, l, k])
            H_lk = H[l*N:(l+1)*N, :, k]
            H[l*N:(l+1)*N, :, k] = Rsqrt @ H_lk
            
    Hhat = np.zeros_like(H)
    B = np.zeros_like(R)
    C = np.zeros_like(R)
    Np = (np.random.randn(N, nbrOfRealizations, L, tau_p) + 1j*np.random.randn(N, nbrOfRealizations, L, tau_p)) * np.sqrt(0.5)
    eyeN = np.eye(N)
    
    for l in range(L):
        for t in range(tau_p):
            pilotUEs = np.where(pilotIndex == t)[0]
            # Tín hiệu pilot nhận được tại AP l
            yp = np.sqrt(p)*tau_p * np.sum(H[l*N:(l+1)*N, :, pilotUEs], axis=2) + np.sqrt(tau_p)*Np[:, :, l, t]
            # Ma trận hiệp phương sai tín hiệu nhận
            Psi = p*tau_p * np.sum(R[:, :, l, pilotUEs], axis=2) + eyeN
            PsiInv = np.linalg.inv(Psi)
            
            # Ước lượng kênh cho từng UE
            for k in pilotUEs:
                RPsi = R[:, :, l, k] @ PsiInv
                Hhat[l*N:(l+1)*N, :, k] = np.sqrt(p) * RPsi @ yp
                B[:, :, l, k] = p*tau_p * RPsi @ R[:, :, l, k]
                C[:, :, l, k] = R[:, :, l, k] - B[:, :, l, k]
                
    return Hhat, H, B, C

# Hàm tính hiệu suất phổ (Spectral Efficiency)
def compute_se_downlink_fast(Hhat, H, D, B, C, tau_c, tau_p, nbrOfRealizations, N, K, L, p, rho_dist, R, pilotIndex, rho_central):
    prelogFactor = (1 - tau_p/tau_c)
    
    # --- Tính MR (Maximum Ratio) ---
    signal_MR = np.zeros(K)
    interf_MR = np.zeros(K)
    cont_MR = np.zeros((K, K))
    
    # Tính theo công thức đóng (Corollary 3 trong bài báo)
    for l in range(L):
        servedUEs = np.where(D[l, :] == 1)[0]
        for k in servedUEs:
            signal_MR[k] += np.sqrt(rho_dist[l, k] * np.trace(B[:, :, l, k].real))
            for i in range(K):
                term = rho_dist[l, k] * np.trace(B[:, :, l, k] @ R[:, :, l, i]).real / np.trace(B[:, :, l, k].real)
                interf_MR[i] += term
                if pilotIndex[k] == pilotIndex[i]:
                    cont_MR[i, k] += np.sqrt(rho_dist[l, k]) * np.trace((B[:, :, l, k] @ np.linalg.inv(R[:, :, l, k])) @ R[:, :, l, i]).real / np.sqrt(np.trace(B[:, :, l, k].real))

    SE_MR = prelogFactor * np.log2(1 + (np.abs(signal_MR)**2) / (interf_MR + np.sum(np.abs(cont_MR)**2, axis=1) - np.abs(signal_MR)**2 + 1))
    
    # --- Tính P-MMSE (Scalable) bằng Monte Carlo ---
    signal_P_MMSE = np.zeros(K)
    interf_P_MMSE_vec = np.zeros(K)
    
    for n in range(nbrOfRealizations):
        interf_P_MMSE_n_realization = np.zeros((K, K), dtype=complex)
        
        for k in range(K):
            servingAPs = np.where(D[:, k] == 1)[0]
            La = len(servingAPs)
            if La == 0: continue

            servedUEs_in_cluster = np.where(np.sum(D[servingAPs, :], axis=0) >= 1)[0]
            
            Hhat_active = np.zeros((N*La, K), dtype=complex)
            H_active = np.zeros((N*La, K), dtype=complex)
            C_tot_partial = np.zeros((N*La, N*La), dtype=complex)
            
            idx_start = 0
            for l_idx in range(La):
                l = servingAPs[l_idx]
                Hhat_active[idx_start:idx_start+N, :] = Hhat[l*N:(l+1)*N, n, :]
                H_active[idx_start:idx_start+N, :] = H[l*N:(l+1)*N, n, :]
                C_tot_partial[idx_start:idx_start+N, idx_start:idx_start+N] = np.sum(C[:, :, l, servedUEs_in_cluster], axis=2)
                idx_start += N
            
            # Tính vector precoding P-MMSE
            # w = p * inv(...) * hhat_k
            try:
                inv_term = np.linalg.inv(p * (Hhat_active[:, servedUEs_in_cluster] @ Hhat_active[:, servedUEs_in_cluster].conj().T + C_tot_partial) + np.eye(La*N))
                w = p * inv_term @ Hhat_active[:, k]
            except np.linalg.LinAlgError:
                w = np.zeros(N*La, dtype=complex) # Fallback nếu ma trận kỳ dị

            # Chuẩn hóa công suất
            norm_w = np.linalg.norm(w)
            if norm_w > 0:
                w = w / norm_w * np.sqrt(rho_central[k]) 
            
            # Tín hiệu mong muốn
            h_k = H_active[:, k]
            signal_P_MMSE[k] += (h_k.conj().T @ w).real / nbrOfRealizations
            
            # Nhiễu gây ra cho các UE khác
            for i in range(K):
                h_i = H_active[:, i]
                interf_P_MMSE_n_realization[i, k] = h_i.conj().T @ w
        
        interf_P_MMSE_vec += np.sum(np.abs(interf_P_MMSE_n_realization)**2, axis=1) / nbrOfRealizations

    SE_P_MMSE = prelogFactor * np.log2(1 + (np.abs(signal_P_MMSE)**2) / (interf_P_MMSE_vec - np.abs(signal_P_MMSE)**2 + 1))
    
    return SE_MR, SE_P_MMSE

# Thuật toán Di truyền (GA) để tối ưu công suất
def genetic_algorithm(Hhat, H, D, B, C, tau_c, tau_p, nbrOfRealizations, N, K, L, p, R, pilotIndex, rho_tot):
    population_size = 10
    generations = 5 # Số thế hệ
    mutation_rate = 0.1
    
    # Khởi tạo quần thể: Các hệ số công suất cho P-MMSE
    population = np.random.rand(population_size, K)
    
    best_fitness = -np.inf
    best_rho = None
    
    print("Running GA optimization...")
    for gen in range(generations):
        fitness = []
        for i in range(population_size):
            # Chuẩn hóa để tổng công suất hợp lý 
            rho_test = population[i, :] * (rho_tot / np.mean(population[i, :])) 
            
            # Tính SE với bộ tham số này 
            _, SE_P = compute_se_downlink_fast(Hhat, H, D, B, C, tau_c, tau_p, 5, N, K, L, p, np.zeros((L,K)), R, pilotIndex, rho_test)
            
            # Hàm mục tiêu: Tối đa hóa SE tối thiểu
            min_se = np.min(SE_P)
            fitness.append(min_se)
            
            if min_se > best_fitness:
                best_fitness = min_se
                best_rho = rho_test
        
        print(f"Gen {gen+1}: Best Min SE = {best_fitness:.4f}")
        
        # Chọn lọc và Lai ghép
        sorted_idx = np.argsort(fitness)[::-1]
        parents = population[sorted_idx[:population_size//2]]
        
        new_pop = []
        new_pop.extend(parents)
        while len(new_pop) < population_size:
            idx1, idx2 = np.random.randint(len(parents)), np.random.randint(len(parents))
            p1, p2 = parents[idx1], parents[idx2]
            mask = np.random.rand(K) > 0.5
            child = np.where(mask, p1, p2)
            
            # Đột biến
            if np.random.rand() < mutation_rate:
                child += np.random.randn(K) * 0.1
                child = np.abs(child) 
            new_pop.append(child)
        population = np.array(new_pop)
        
    return best_rho

# === MAIN SIMULATION ===
if __name__ == "__main__":
    selectSimulationSetup = 2 # 1 = Hình 6a, 2 = Hình 6b
    
    if selectSimulationSetup == 1:
        L, N = 400, 1
    elif selectSimulationSetup == 2:
        L, N = 100, 4

    K = 100
    tau_c = 200
    tau_p = 10
    p = 100
    rho_tot = 1000 # mW

    # Cấu hình số vòng lặp (Giảm nhỏ để chạy demo nhanh)
    # Tăng lên (nbrOfSetups=25, nbrOfRealizations=1000) để có kết quả mượt như báo
    nbrOfSetups = 25 
    nbrOfRealizations = 1000

    SE_MR_tot = []
    SE_P_MMSE_tot = []
    SE_P_MMSE_GA_tot = []

    print(f"Starting simulation with L={L}, N={N}, K={K}...")
    
    for n in range(nbrOfSetups):
        print(f"Setup {n+1} out of {nbrOfSetups}")
        
        gainOverNoisedB, R, pilotIndex, D = generateSetup(L, K, N, tau_p)
        Hhat, H, B, C = functionChannelEstimates(R, nbrOfRealizations, L, K, N, tau_p, pilotIndex, p)
        
        # --- Heuristic Power Allocation (Theo bài báo) ---
        rho_central = (rho_tot/tau_p)*np.ones(K)
        
        # Power Allocation cho Distributed MR (Eq. 43)
        rho_dist = np.zeros((L, K))
        gainOverNoise = db2pow(gainOverNoisedB)
        for l in range(L):
            servedUEs = np.where(D[l, :] == 1)[0]
            normAP = np.sum(np.sqrt(gainOverNoise[l, servedUEs]))
            if normAP > 0:
                rho_dist[l, servedUEs] = rho_tot * np.sqrt(gainOverNoise[l, servedUEs]) / normAP

        # Tính SE cơ bản
        SE_MR, SE_P_MMSE = compute_se_downlink_fast(Hhat, H, D, B, C, tau_c, tau_p, nbrOfRealizations, N, K, L, p, rho_dist, R, pilotIndex, rho_central)
        
        SE_MR_tot.extend(SE_MR)
        SE_P_MMSE_tot.extend(SE_P_MMSE)
        
        # --- Chạy GA để tối ưu P-MMSE ---
        best_rho_central = genetic_algorithm(Hhat, H, D, B, C, tau_c, tau_p, nbrOfRealizations, N, K, L, p, R, pilotIndex, rho_tot)
        
        # Tính lại SE với bộ tham số tối ưu từ GA
        _, SE_P_MMSE_GA = compute_se_downlink_fast(Hhat, H, D, B, C, tau_c, tau_p, nbrOfRealizations, N, K, L, p, rho_dist, R, pilotIndex, best_rho_central)
        SE_P_MMSE_GA_tot.extend(SE_P_MMSE_GA)

    # Vẽ biểu đồ CDF
    plt.figure(figsize=(10, 6))
    
    # Hàm vẽ CDF
    def plot_cdf(data, style, label):
        sorted_data = np.sort(data)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        plt.plot(sorted_data, yvals, style, linewidth=2, label=label)

    plot_cdf(SE_P_MMSE_GA_tot, 'g-', 'P-MMSE (GA Optimized)')
    plot_cdf(SE_P_MMSE_tot, 'r-.', 'P-MMSE (Scalable Heuristic)')
    plot_cdf(SE_MR_tot, 'k-', 'MR (Scalable)')

    plt.xlabel('Spectral efficiency [bit/s/Hz]')
    plt.ylabel('CDF')
    plt.legend(loc='lower right')
    plt.xlim([0, 10])
    plt.grid(True)
    plt.title(f'CDF of Downlink Spectral Efficiency (Setup {selectSimulationSetup})')
    plt.show()