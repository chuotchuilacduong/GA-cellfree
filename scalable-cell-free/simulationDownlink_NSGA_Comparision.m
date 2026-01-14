% Script mô phỏng so sánh Downlink Power Allocation: 
% 1. Equal Allocation (Chia đều công suất) - Benchmark
% 2. NSGA-II Optimized Allocation - Proposed
% Tương tự phong cách của simulationFigure5.m

close all;
clear;
clc;

% Thêm đường dẫn tới thư mục NSGA code (Sửa lại tên folder nếu khác)
addpath('NSGA_CODE'); 

%% 1. CẤU HÌNH MÔ PHỎNG
% Để test nhanh, giảm số lượng setup và thông số NSGA
% Khi muốn kết quả đẹp mịn, tăng nbrOfSetups = 50, pop = 100, gen = 100

nbrOfSetups = 5;          % Số lượng Monte Carlo Setups (Chạy lâu thì giảm xuống 1-5 để test)
nbrOfRealizations = 100;  % Số lượng hiện thực hóa kênh nhỏ (Small-scale fading)

L = 100;        % Số lượng AP
N = 1;          % Số lượng anten mỗi AP
K = 20;         % Số lượng UE
tau_c = 200;    % Độ dài coherence block
tau_p = 10;     % Độ dài pilot
p = 100;        % Uplink power (cho pilot training)
rho_tot = 200;  % Downlink power per AP (mW)

% Tham số NSGA-II
nsga_params.Np = 50;      % Population size
nsga_params.maxgen = 30;  % Generations
nsga_params.pc = 0.9;
nsga_params.pm = 0.1;
nsga_params.ms = 0.05;

% Biến lưu kết quả để vẽ CDF
SE_Equal_MR_tot = [];
SE_NSGA_MR_tot = [];

fprintf('Bắt đầu mô phỏng với %d Setups...\n', nbrOfSetups);

%% 2. VÒNG LẶP MONTE CARLO (Setups)
for n = 1:nbrOfSetups
    
    disp(['--- Setup ', num2str(n), ' / ', num2str(nbrOfSetups), ' ---']);
    
    % A. Tạo vị trí ngẫu nhiên và Large-scale Fading
    [gainOverNoisedB, R, pilotIndex, D] = generateSetup(L, K, N, tau_p, 1);
    
    % B. Ước lượng thống kê kênh (Tính ma trận B)
    % Ta cần B và R để chạy tối ưu hóa nhanh
    [Hhat_dummy, H_dummy, B, C] = functionChannelEstimates(R, 1, L, K, N, tau_p, pilotIndex, p);
    
    % Tạo kênh thực tế (Small-scale fading) dùng để tính SE chính xác sau cùng
    [Hhat, H, ~, ~] = functionChannelEstimates(R, nbrOfRealizations, L, K, N, tau_p, pilotIndex, p);
    
    %% C. PHƯƠNG PHÁP 1: EQUAL POWER ALLOCATION (BENCHMARK)
    % Chia đều công suất rho_tot cho các UE được phục vụ tại mỗi AP
    rho_dist_equal = zeros(L, K);
    for l = 1:L
        servedUEs = find(D(l,:) == 1);
        numServed = length(servedUEs);
        if numServed > 0
            rho_dist_equal(l, servedUEs) = rho_tot / numServed;
        end
    end
    
    % Tính SE chính xác bằng hàm có sẵn
    % Lưu ý: Hàm functionComputeSE_downlink cần tham số rho_central, ta để dummy
    rho_central_dummy = zeros(K,1); 
    [SE_MR_Equal, ~, ~, ~, ~, ~] = functionComputeSE_downlink(Hhat, H, D, B, C, tau_c, tau_p, nbrOfRealizations, N, K, L, p, rho_dist_equal, R, pilotIndex, rho_central_dummy);
    
    SE_Equal_MR_tot = [SE_Equal_MR_tot; SE_MR_Equal]; % Tích lũy kết quả
    
    %% D. PHƯƠNG PHÁP 2: NSGA-II POWER ALLOCATION
    disp('   Running NSGA-II...');
    
    % Xác định biến quyết định: Mỗi liên kết AP-UE active là 1 biến trọng số
    % Biến D quyết định liên kết nào active
    active_links_indices = find(D == 1); % Chỉ mục tuyến tính
    numVars = length(active_links_indices);
    
    if numVars > 0
        % Cấu hình bài toán NSGA
        MultiObj.nVar = numVars;
        MultiObj.var_min = 0.01 * ones(1, numVars);
        MultiObj.var_max = 1.00 * ones(1, numVars);
        
        % Wrapper gọi hàm fitness
        MultiObj.fun = @(pop) fitnessPowerAllocation(pop, L, K, D, B, R, pilotIndex, rho_tot, tau_c, tau_p);
        
        % Chạy NSGA
        % Tắt output text của NSGA để đỡ rối
        [~, ~, old_warning_state] = evalc('warning("off");'); 
        [x_opt_pop, fval_pop] = NSGAII(nsga_params, MultiObj);
        warning(old_warning_state);
        
        % Chọn giải pháp tốt nhất từ Pareto Front
        % Tiêu chí: Maximize Min SE (Max-Min Fairness) -> Cột 2 của fval_pop là -MinSE
        % Ta tìm giá trị nhỏ nhất của cột 2 (tức là Max Min SE lớn nhất)
        [~, best_idx] = min(fval_pop(:, 2));
        x_best = x_opt_pop(best_idx, :);
        
        % Decode x_best thành rho_dist_nsga
        rho_dist_nsga = zeros(L, K);
        current_idx = 1;
        for l = 1:L
            servedUEs = find(D(l,:) == 1);
            numServed = length(servedUEs);
            if numServed > 0
                weights = x_best(current_idx : current_idx + numServed - 1);
                current_idx = current_idx + numServed;
                sum_w = sum(weights);
                rho_dist_nsga(l, servedUEs) = (weights / sum_w) * rho_tot;
            end
        end
        
        % Tính SE chính xác với Power đã tối ưu
        [SE_MR_NSGA, ~, ~, ~, ~, ~] = functionComputeSE_downlink(Hhat, H, D, B, C, tau_c, tau_p, nbrOfRealizations, N, K, L, p, rho_dist_nsga, R, pilotIndex, rho_central_dummy);
        
        SE_NSGA_MR_tot = [SE_NSGA_MR_tot; SE_MR_NSGA];
        
    else
        % Trường hợp không có kết nối nào (hiếm)
        SE_NSGA_MR_tot = [SE_NSGA_MR_tot; SE_MR_Equal];
    end
    
end

%% 3. VẼ BIỂU ĐỒ CDF SO SÁNH
disp('Vẽ biểu đồ so sánh...');

figure;
hold on; box on; grid on;

% Sắp xếp dữ liệu để vẽ CDF
sorted_Equal = sort(SE_Equal_MR_tot);
sorted_NSGA = sort(SE_NSGA_MR_tot);
num_points = length(sorted_Equal);
y_axis = linspace(0, 1, num_points);

% Vẽ
plot(sorted_Equal, y_axis, 'k--', 'LineWidth', 2);
plot(sorted_NSGA, y_axis, 'r-', 'LineWidth', 2);

xlabel('Spectral Efficiency [bit/s/Hz]', 'Interpreter', 'Latex');
ylabel('CDF', 'Interpreter', 'Latex');
title('Downlink SE: Equal Power vs. NSGA-II Optimized');
legend({'MR (Equal Power)', 'MR (NSGA-II Power Opt)'}, 'Interpreter', 'Latex', 'Location', 'SouthEast');
xlim([0 max(sorted_NSGA)*1.1]);

disp('Hoàn tất.');