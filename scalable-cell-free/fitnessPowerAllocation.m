function f_matrix = fitnessPowerAllocation(pop_matrix, L, K, D, B, R, pilotIndex, rho_tot, tau_c, tau_p)
    % Hàm wrapper để tính fitness cho cả quần thể (vectorized)
    % pop_matrix: [PopSize x NumVars]
    % Output: [PopSize x 2] (f1 = -SumSE, f2 = -MinSE)

    [Np, ~] = size(pop_matrix);
    f_matrix = zeros(Np, 2);
    
    % Hằng số prelog
    prelogFactor = (1 - tau_p / tau_c);
    
    % Cache các vị trí cần thiết để truy xuất nhanh
    servedUEs_per_AP = cell(L, 1);
    numServed_per_AP = zeros(L, 1);
    for l = 1:L
        servedUEs_per_AP{l} = find(D(l,:) == 1);
        numServed_per_AP(l) = length(servedUEs_per_AP{l});
    end
    
    % Duyệt qua từng cá thể trong quần thể
    for i = 1:Np
        x = pop_matrix(i, :);
        
        % 1. Decode biến x thành ma trận công suất rho_dist
        rho_dist = zeros(L, K);
        current_idx = 1;

        for l = 1:L
            numServed = numServed_per_AP(l);
            if numServed > 0
                % Lấy trọng số từ nhiễm sắc thể
                weights = x(current_idx : current_idx + numServed - 1);
                current_idx = current_idx + numServed;
                
                % Chuẩn hóa công suất tại mỗi AP
                sum_w = sum(weights);
                if sum_w > 1e-9
                    power_alloc = (weights / sum_w) * rho_tot;
                else
                    power_alloc = (1/numServed) * rho_tot * ones(size(weights));
                end
                
                rho_dist(l, servedUEs_per_AP{l}) = power_alloc;
            end
        end
        
        % 2. Tính toán SE (Closed-form approximation for MR)
        % Sử dụng công thức đóng dựa trên thống kê kênh (R, B) để tính nhanh
        
        signal_MR = zeros(K,1);
        interf_MR = zeros(K,1);
        cont_MR  = zeros(K,K); % Contamination
        
        % Tính toán các thành phần Signal, Interference, Contamination
        for l = 1:L
            servedUEs = servedUEs_per_AP{l};
            for idx = 1:length(servedUEs)
                k = servedUEs(idx);
                
                val_trace_B = real(trace(B(:,:,l,k)));
                sqrt_rho = sqrt(rho_dist(l,k));
                
                % Signal term (Coherent accumulation)
                signal_MR(k) = signal_MR(k) + sqrt_rho * sqrt(val_trace_B);
                
                for u = 1:K
                    % Interference term
                    val_trace_BR = real(trace(B(:,:,l,k) * R(:,:,l,u)));
                    interf_MR(u) = interf_MR(u) + rho_dist(l,k) * val_trace_BR / val_trace_B;
                    
                    % Pilot contamination term
                    if pilotIndex(k) == pilotIndex(u) && k ~= u
                         val_trace_cont = real(trace( (B(:,:,l,k)/R(:,:,l,k)) * R(:,:,l,u) ));
                         cont_MR(u,k) = cont_MR(u,k) + sqrt_rho * val_trace_cont / sqrt(val_trace_B);
                    end
                end
            end
        end
        
        % Tính SINR và SE
        numerator = abs(signal_MR).^2;
        % Lưu ý: cont_MR(u,k) là contamination từ k gây ra cho u
        contamination_power = sum(abs(cont_MR).^2, 2); 
        
        denominator = interf_MR + contamination_power - numerator + 1; % +1 là Noise normalized
        denominator(denominator < 1e-9) = 1e-9;
        
        SINR = numerator ./ denominator;
        SE_MR = prelogFactor * log2(1 + SINR);
        
        % 3. Gán giá trị mục tiêu (Minimize negative values)
        f_matrix(i, 1) = -sum(SE_MR); % Max Sum Rate
        f_matrix(i, 2) = -min(SE_MR); % Max Min Rate
    end
end