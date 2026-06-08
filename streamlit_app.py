import numpy as np
import pandas as pd
import streamlit as st

# ================= CÁC HÀM THUẬT TOÁN ĐƠN HÌNH HAI PHA TỐI ƯU =================


def giai_phase_1(tableau, header_vars, basis, num_constraints):
    logs = []
    tables_step = []
    iteration = 0

    while True:
        g_row = tableau[-1, :-1]
        # Nếu tất cả hệ số dòng mục tiêu >= -1e-9 -> Đạt tối ưu Pha 1
        if np.all(g_row >= -1e-9):
            break

        iteration += 1

        # --- ÁP DỤNG QUY TẮC BLAND (DÒNG VÀO) ---
        candidate_cols = np.where(g_row < -1e-9)[0]
        col_vao = candidate_cols[0]

        tiso = []
        for i in range(num_constraints):
            val = tableau[i, col_vao]
            rhs = tableau[i, -1]
            if val > 1e-9:
                tiso.append(rhs / val)
            else:
                tiso.append(np.inf)

        tiso = np.array(tiso)
        if np.all(tiso == np.inf):
            return (
                None,
                ["Pha 1 kết thúc: Hệ ràng buộc không giới hạn (Vô nghiệm)"],
                [],
                basis,
            )

        # --- ÁP DỤNG QUY TẮC BLAND (DÒNG RA) ---
        min_val = np.min(tiso)
        candidate_rows = np.where(np.abs(tiso - min_val) < 1e-9)[0]

        row_ra = candidate_rows[0]
        best_basis_var = basis[row_ra]
        for r in candidate_rows:
            if basis[r] < best_basis_var:
                best_basis_var = basis[r]
                row_ra = r

        bien_ra_ten = header_vars[basis[row_ra]]
        basis[row_ra] = col_vao

        # Biến đổi Gauss-Jordan xoay trục quanh phần tử pivot
        pt_quay = tableau[row_ra, col_vao]
        tableau[row_ra, :] /= pt_quay

        for i in range(len(tableau)):
            if i != row_ra:
                tableau[i, :] -= tableau[i, col_vao] * tableau[row_ra, :]

        logs.append(
            f"**Pha 1 - Bước {iteration}** -> Biến vào: `{header_vars[col_vao]}`, Biến ra: `{bien_ra_ten}` (Dòng {row_ra + 1})"
        )
        # Lưu bản sao ma trận (Ẩn dòng f gốc khi xem Pha 1)
        tables_step.append(
            (f"Pha 1 - Bước {iteration}", np.delete(tableau, -2, axis=0).copy())
        )

    # Kiểm tra tính chấp nhận được của bài toán ban đầu (g_opt phải bằng 0)
    if abs(tableau[-1, -1]) > 1e-5:
        return None, ["Bài toán ban đầu VÔ NGHIỆM!"], [], basis

    return tableau, logs, tables_step, basis


def giai_simplex_hai_pha_tong_quat(c_new, A_mat, b_list, slack_rows, header_vars):
    num_constraints = len(b_list)
    num_artificial = sum(1 for sign in slack_rows if sign in [">=", "="])
    total_vars = len(header_vars) + num_artificial

    # Thiết lập bảng đơn hình tổng quát: gồm dòng f gốc và dòng g giả
    tableau = np.zeros((num_constraints + 2, total_vars + 1))
    tableau[:num_constraints, : len(header_vars)] = A_mat
    tableau[:num_constraints, -1] = b_list
    tableau[-2, : len(header_vars)] = c_new

    art_vars_indices = []
    header_vars_with_art = list(header_vars)
    basis = [-1] * num_constraints

    # Xác định cơ sở xuất phát và cấu trúc biến giả R
    art_idx = len(header_vars)
    slack_cnt = 0
    for i, sign in enumerate(slack_rows):
        if sign == "<=":
            for idx, h in enumerate(header_vars):
                if h.startswith("s_") and idx >= (
                    len(header_vars) - len(slack_rows) + slack_cnt
                ):
                    basis[i] = idx
                    slack_cnt += 1
                    break
        else:
            tableau[i, art_idx] = 1
            header_vars_with_art.append(f"R_{len(art_vars_indices) + 1}")
            art_vars_indices.append(art_idx)
            basis[i] = art_idx
            tableau[-1, :] -= tableau[i, :]  # Triệt tiêu hệ số cơ sở tại dòng g
            art_idx += 1

    all_logs = []
    all_tables = []

    # --- CHẠY PHA 1 ---
    if num_artificial > 0:
        all_tables.append(
            (
                "Pha 1 - Bảng khởi tạo (Bước 0)",
                np.delete(tableau, -2, axis=0).copy(),
            )
        )
        res = giai_phase_1(
            tableau, header_vars_with_art, basis, num_constraints
        )
        if res[0] is None:
            return None, res[1], [], header_vars_with_art, basis
        tableau, p1_logs, p1_tables, basis = res
        all_logs.extend(p1_logs)
        all_tables.extend(p1_tables)

        # --- ĐOẠN KHỬ BIẾN GIẢ KHỎI CƠ SỞ ---
        drive_out_cnt = 0
        for i, b_idx in enumerate(basis):
            if b_idx in art_vars_indices:
                # Tìm một biến thực j không nằm trong cơ sở để thế chỗ biến giả
                for j in range(len(header_vars)):
                    if j not in basis and abs(tableau[i, j]) > 1e-9:
                        pt_quay = tableau[i, j]
                        tableau[i, :] /= pt_quay
                        for r in range(len(tableau)):
                            if r != i:
                                tableau[r, :] -= (
                                    tableau[r, j] * tableau[i, :]
                                )
                        
                        drive_out_cnt += 1
                        all_logs.append(
                            f"**Đuổi biến giả**: Thay thế `{header_vars_with_art[b_idx]}` bằng `{header_vars_with_art[j]}` tại cơ sở dòng {i + 1}."
                        )
                        basis[i] = j
                        
                        # Lưu lại ma trận ngay sau bước biến đổi đuổi biến giả này (Ẩn dòng f gốc)
                        all_tables.append(
                            (
                                f"Pha 1 - Bước khử biến giả phụ {drive_out_cnt}",
                                np.delete(tableau, -2, axis=0).copy(),
                            )
                        )
                        break

        # giải phóng các cột chứa biến giả R
        tableau = np.delete(tableau, art_vars_indices, axis=1)
        for idx in sorted(art_vars_indices, reverse=True):
            del header_vars_with_art[idx]
            for b_i in range(len(basis)):
                if basis[b_i] > idx:
                    basis[b_i] -= 1

    # --- CHẠY PHA 2 ---
    tableau_p2 = tableau[:-1, :].copy()
    all_tables.append(("Pha 2 - Bảng khởi tạo (Bước 0)", tableau_p2.copy()))

    iteration = 0
    while True:
        c_row = tableau_p2[-1, :-1]
        if np.all(c_row >= -1e-9):
            break

        iteration += 1

        # Áp dụng quy tắc Bland cho Pha 2
        candidate_cols = np.where(c_row < -1e-9)[0]
        col_vao = candidate_cols[0]

        tiso = []
        for i in range(num_constraints):
            val = tableau_p2[i, col_vao]
            rhs = tableau_p2[i, -1]
            if val > 1e-9:
                tiso.append(rhs / val)
            else:
                tiso.append(np.inf)

        tiso = np.array(tiso)
        if np.all(tiso == np.inf):
            all_logs.append("Bài toán vô hạn nghiệm ở Pha 2!")
            return None, all_logs, [], header_vars_with_art, basis

        # Quy tắc Bland chọn dòng ra cho Pha 2
        min_val = np.min(tiso)
        candidate_rows = np.where(np.abs(tiso - min_val) < 1e-9)[0]

        row_ra = candidate_rows[0]
        best_basis_var = basis[row_ra]
        for r in candidate_rows:
            if basis[r] < best_basis_var:
                best_basis_var = basis[r]
                row_ra = r

        bien_ra_ten = header_vars_with_art[basis[row_ra]]
        basis[row_ra] = col_vao

        pt_quay = tableau_p2[row_ra, col_vao]
        tableau_p2[row_ra, :] /= pt_quay

        for i in range(len(tableau_p2)):
            if i != row_ra:
                tableau_p2[i, :] -= (
                    tableau_p2[i, col_vao] * tableau_p2[row_ra, :]
                )

        all_logs.append(
            f"**Pha 2 - Bước {iteration}** -> Biến vào: `{header_vars_with_art[col_vao]}`, Biến ra: `{bien_ra_ten}` (Dòng {row_ra + 1})"
        )
        all_tables.append((f"Pha 2 - Bước {iteration}", tableau_p2.copy()))

    return tableau_p2, all_logs, all_tables, header_vars_with_art, basis


# ================= GIAO DIỆN WEB STREAMLIT =================
st.set_page_config(page_title="Giải QHTT Tổng Quát", layout="wide")
st.title("Chương trình Giải bài toán Quy hoạch tuyến tính tổng quát")
st.subheader("Sử dụng thuật toán Đơn hình Hai pha")

col1, col2 = st.columns(2)
with col1:
    num_vars = st.number_input(
        "Nhập số lượng biến ban đầu:", min_value=1, value=2, step=1
    )
with col2:
    target_type = st.selectbox("Hàm mục tiêu:", ["Max", "Min"]).lower()

st.markdown("---")
st.write("### 1. Hàm mục tiêu và Điều kiện dấu của biến")

c_orig = []
var_types = []
cols_vars = st.columns(int(num_vars))

for i in range(int(num_vars)):
    with cols_vars[i]:
        st.write(f"**Biến $x_{i+1}$**")
        hso = st.number_input(f"Hệ số c_{i+1}:", value=1.0, key=f"c_{i}")
        c_orig.append(hso)
        dau = st.selectbox(
            f"Dấu của $x_{i+1}$:",
            [">= 0", "<= 0", "Tùy ý"],
            key=f"type_{i}",
        )
        if dau == ">= 0":
            var_types.append(">=0")
        elif dau == "<= 0":
            var_types.append("<=0")
        else:
            var_types.append("tu_y")

# Ánh xạ đổi biến sang dạng chính quy chuẩn
header_vars = []
mapping = {}
current_idx = 0
for i in range(int(num_vars)):
    if var_types[i] == ">=0":
        header_vars.append(f"x_{i+1}")
        mapping[i] = ("pos", current_idx)
        current_idx += 1
    elif var_types[i] == "<=0":
        header_vars.append(f"x_{i+1}'")
        mapping[i] = ("neg", current_idx)
        current_idx += 1
    else:
        header_vars.append(f"x_{i+1}+")
        header_vars.append(f"x_{i+1}-")
        mapping[i] = ("free", current_idx, current_idx + 1)
        current_idx += 2

c_new = np.zeros(len(header_vars))
for i in range(int(num_vars)):
    hso = c_orig[i] if target_type == "min" else -c_orig[i]
    if var_types[i] == ">=0":
        c_new[mapping[i][1]] = hso
    elif var_types[i] == "<=0":
        c_new[mapping[i][1]] = -hso
    else:
        c_new[mapping[i][1]] = hso
        c_new[mapping[i][2]] = -hso

st.markdown("---")
st.write("### 2. Các ràng buộc điều kiện")
num_constraints = st.number_input(
    "Nhập số lượng ràng buộc:", min_value=1, value=2, step=1
)

A_list = []
b_list = []
slack_rows = []

for i in range(int(num_constraints)):
    st.write(f"**Ràng buộc {i+1}:**")
    cols_constraint = st.columns(int(num_vars) + 2)

    row_orig = []
    for j in range(int(num_vars)):
        with cols_constraint[j]:
            val = st.number_input(
                f"Hệ số x_{j+1}:", value=1.0, key=f"A_{i}_{j}"
            )
            row_orig.append(val)

    with cols_constraint[-2]:
        sign = st.selectbox(
            "Dấu:", ["<=", ">=", "="], key=f"sign_{i}", index=0
        )
    with cols_constraint[-1]:
        rhs = st.number_input("Vế phải b:", value=0.0, key=f"b_{i}")

    # Chuẩn hóa b không âm
    if rhs < 0:
        row_orig = [-x for x in row_orig]
        rhs = -rhs
        if sign == "<=":
            sign = ">="
        elif sign == ">=":
            sign = "<="

    new_row = np.zeros(len(header_vars))
    for j in range(int(num_vars)):
        if var_types[j] == ">=0":
            new_row[mapping[j][1]] = row_orig[j]
        elif var_types[j] == "<=0":
            new_row[mapping[j][1]] = -row_orig[j]
        else:
            new_row[mapping[j][1]] = row_orig[j]
            new_row[mapping[j][2]] = -row_orig[j]

    A_list.append(new_row)
    b_list.append(rhs)
    slack_rows.append(sign)

A_mat = np.array(A_list)
header_vars_with_slack = list(header_vars)
c_new_with_slack = np.array(c_new)

# Thêm biến s_i (Biến bù / biến phụ)
slack_idx = 1
for i, sign in enumerate(slack_rows):
    if sign == "<=":
        col_slack = np.zeros((int(num_constraints), 1))
        col_slack[i] = 1
        A_mat = np.hstack((A_mat, col_slack))
        header_vars_with_slack.append(f"s_{slack_idx}")
        c_new_with_slack = np.append(c_new_with_slack, 0)
        slack_idx += 1
    elif sign == ">=":
        col_slack = np.zeros((int(num_constraints), 1))
        col_slack[i] = -1
        A_mat = np.hstack((A_mat, col_slack))
        header_vars_with_slack.append(f"s_{slack_idx}")
        c_new_with_slack = np.append(c_new_with_slack, 0)
        slack_idx += 1

st.markdown("---")
if st.button("BẮT ĐẦU GIẢI BÀI TOÁN", type="primary"):
    res_tableau, logs, all_tables, final_headers, final_basis = (
        giai_simplex_hai_pha_tong_quat(
            -c_new_with_slack,
            A_mat,
            np.array(b_list),
            slack_rows,
            header_vars_with_slack,
        )
    )

    if res_tableau is not None:
        st.success("ĐÃ TÌM THẤY PHƯƠNG ÁN TỐI ƯU!")

        # --- IN THÀNH PHẦN BỘ LẶP DƯỚI DẠNG DATAFRAME ---
        st.write("### Các bảng đơn hình trong bộ lặp:")
        for title, table in all_tables:
            with st.expander(f" Xem {title}"):
                if "Pha 1" in title:
                    num_current_vars = table.shape[1] - 1
                    num_r = num_current_vars - len(header_vars_with_slack)
                    cols_table = (
                        header_vars_with_slack
                        + [f"R_{k+1}" for k in range(num_r)]
                        + ["RHS"]
                    )
                else:
                    cols_table = final_headers + ["RHS"]

                df_display = pd.DataFrame(table, columns=cols_table)
                st.dataframe(
                    df_display.style.format("{:.2f}"), use_container_width=True
                )

        st.write("### Nhật ký dịch chuyển cơ sở (Quy tắc Bland áp dụng)")
        for log in logs:
            st.write(log)

        # --- GIẢI MÃ KHÔI PHỤC NGHIỆM GỐC ---
        var_values = {h: 0.0 for h in final_headers}
        for i, b_idx in enumerate(final_basis):
            if b_idx < len(final_headers):
                var_values[final_headers[b_idx]] = res_tableau[i, -1]

        x_optimal = []
        for i in range(int(num_vars)):
            if var_types[i] == ">=0":
                val = var_values.get(f"x_{i+1}", 0.0)
            elif var_types[i] == "<=0":
                val = -var_values.get(f"x_{i+1}'", 0.0)
            else:
                val = var_values.get(f"x_{i+1}+", 0.0) - var_values.get(
                    f"x_{i+1}-", 0.0
                )
            x_optimal.append(val)

        # Trả kết quả nghiệm
        st.write("### Phương án tối ưu cuối cùng (Nghiệm gốc):")
        cols_res = st.columns(int(num_vars))
        for i in range(int(num_vars)):
            with cols_res[i]:
                st.metric(label=f"Biến x_{i+1}", value=f"{x_optimal[i]:.4f}")

        f_opt = (
            res_tableau[-1, -1] if target_type == "min" else -res_tableau[-1, -1]
        )
        st.markdown("---")
        st.metric(
            label=f"Giá trị tối ưu cực trị f(x) [{target_type.upper()}]",
            value=f"{f_opt:.4f}",
        )
    else:
        st.error(
            "Không tìm thấy phương án tối ưu hợp lệ (Bài toán Vô nghiệm hoặc Không giới hạn)"
        )
