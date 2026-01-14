import numpy as np

def calculate_lp_mmse_combiner(N, sigma2, target_user_k, served_users_indices, p_uplink, H_hat, C_error):
    """
    Tính toán vector kết hợp LP-MMSE cho một AP và một người dùng cụ thể (UE k).
    Dựa trên Công thức (29) trong bài báo Scalable Cell-Free Massive MIMO.

    Tham số:
    - N: Số lượng ăng-ten của AP.
    - sigma2: Công suất nhiễu (noise variance).
    - target_user_k: ID của người dùng k mà ta muốn tính vector kết hợp.
    - served_users_indices: Danh sách ID các người dùng thuộc tập D_l (AP đang phục vụ).
    - p_uplink: Mảng chứa công suất phát của từng UE (p_i).
    - H_hat: Dictionary chứa ước lượng kênh h_hat_il của các UE (kích thước N x 1).
    - C_error: Dictionary chứa ma trận hiệp phương sai sai số C_il (kích thước N x N).

    Trả về:
    - v_kl: Vector kết hợp (kích thước N x 1).
    """
    
    # Bước 1: Khởi tạo ma trận tổng bên trong dấu ngoặc (Phần can thiệp + Nhiễu)
    # Tương ứng với: sum( p_i * (...) ) + sigma^2 * I_N
    # Kích thước ma trận là N x N
    A_matrix = np.zeros((N, N), dtype=complex)

    # Bước 2: Duyệt qua tất cả người dùng i trong tập D_l (served_users_indices)
    for i in served_users_indices:
        # Lấy ước lượng kênh h_hat của người dùng i
        h_hat_i = H_hat[i] # Shape (N, 1)
        
        # Tính tích ngoài: h_hat * h_hat^H
        # .conj().T là chuyển vị liên hợp (Hermitian transpose)
        h_outer = h_hat_i @ h_hat_i.conj().T
        
        # Lấy ma trận sai số ước lượng C_il
        C_i = C_error[i]
        
        # Lấy công suất phát p_i
        p_i = p_uplink[i]
        
        # Cộng dồn vào ma trận A: p_i * (h_hat * h_hat^H + C_il)
        term_i = p_i * (h_outer + C_i)
        A_matrix += term_i

    # Bước 3: Cộng thêm thành phần tạp âm nhiệt (Regularization)
    # sigma^2 * I_N
    A_matrix += sigma2 * np.eye(N)

    # Bước 4: Nghịch đảo ma trận A
    try:
        A_inv = np.linalg.inv(A_matrix)
    except np.linalg.LinAlgError:
        # Sử dụng giả nghịch đảo nếu ma trận suy biến (hiếm gặp do đã cộng nhiễu)
        A_inv = np.linalg.pinv(A_matrix)

    # Bước 5: Tính vector kết hợp v_kl
    # Công thức: p_k * A_inv * h_hat_k
    p_k = p_uplink[target_user_k]
    h_hat_k = H_hat[target_user_k]
    
    v_kl = p_k * (A_inv @ h_hat_k)
    
    return v_kl

# --- VÍ DỤ SỬ DỤNG ---

# 1. Thiết lập thông số giả lập
np.random.seed(42) # Để kết quả cố định
N_antennas = 4      # AP có 4 ăng-ten
K_users = 10        # Tổng số UE trong mạng
Sigma_sq = 1.0      # Công suất nhiễu

# Giả sử AP l này phục vụ 3 người dùng: UE 0, UE 2, và UE 5
# Đây chính là tập D_l trong bài báo
my_served_users = [0, 2, 5] 

# Công suất phát Uplink (giả sử ai cũng phát mức 100 mW)
p_uplink = np.ones(K_users) * 0.1 

# Tạo dữ liệu kênh giả lập (H_hat) và ma trận sai số (C)
H_hat_data = {}
C_error_data = {}

for k in my_served_users:
    # Kênh Rayleigh ngẫu nhiên (phần thực + ảo)
    H_hat_data[k] = (np.random.randn(N_antennas, 1) + 1j * np.random.randn(N_antennas, 1)) / np.sqrt(2)
    # Ma trận sai số ngẫu nhiên (Hermitian positive semi-definite)
    temp_mat = np.random.randn(N_antennas, N_antennas) + 1j * np.random.randn(N_antennas, N_antennas)
    C_error_data[k] = 0.01 * (temp_mat @ temp_mat.conj().T) # Sai số nhỏ

# 2. Chọn người dùng mục tiêu để tính vector kết hợp (ví dụ: UE 2)
target_ue = 2

# 3. Gọi hàm tính toán
if target_ue in my_served_users:
    v_vector = calculate_lp_mmse_combiner(
        N=N_antennas,
        sigma2=Sigma_sq,
        target_user_k=target_ue,
        served_users_indices=my_served_users,
        p_uplink=p_uplink,
        H_hat=H_hat_data,
        C_error=C_error_data
    )

    print(f"Vector kết hợp LP-MMSE cho UE {target_ue} (kích thước {v_vector.shape}):")
    print(v_vector)
    
    # Kiểm tra nhanh: Vector này dùng để nhân với tín hiệu thu y_l: v^H * y_l
else:
    print(f"UE {target_ue} không được phục vụ bởi AP này.")