import torch
import torch.nn as nn
import numpy as np
import time
import os

# Resolve OpenMP duplicate library conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==========================================
# 0. Global parameters
# ==========================================
FORCE_MINS = [3.0, 2.0, 2.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2.0, 2.0, 3.0]
FORCE_MAXS = [3.5, 2.5, 2.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 3.5]
GB_MIN = -65.0
GB_MAX = -55.0

# ==========================================
# 1. Geometry: symmetric half-beam model
# ==========================================
model_data = {
    'Beam-EA': 1.7e8,
    'Beam-EI': 7.0e7,
    'Num-beamnodes': 31,
    'Beam-length': 300.0,
    'Num-cables': 7,
}

def get_exact_geometry(data):
    n_bn = data['Num-beamnodes']
    beam_x = np.linspace(-150.0, 150.0, n_bn)
    beam_y = np.ones(n_bn) * 20.0
    beam_nodes = np.column_stack((beam_x, beam_y))

    L_cb_beam = np.column_stack((np.linspace(-140.0, -20.0, 7), np.ones(7)*20.0))
    L_cb_tower = np.column_stack((np.zeros(7), np.linspace(90.0, 72.0, 7)))
    R_cb_beam = np.column_stack((np.linspace(20.0, 140.0, 7), np.ones(7)*20.0))
    R_cb_tower = np.column_stack((np.zeros(7), np.linspace(72.0, 90.0, 7)))

    CB_indices = np.concatenate((np.arange(1, 14, 2), np.arange(17, 30, 2)))
    all_beam_pts = np.vstack((L_cb_beam, R_cb_beam))
    all_tower_pts = np.vstack((L_cb_tower, R_cb_tower))

    vectors = all_tower_pts - all_beam_pts
    lengths = np.sqrt(np.sum(vectors**2, axis=1, keepdims=True))
    unit_vectors = vectors / lengths
    cb_list = np.column_stack((CB_indices, unit_vectors))
    return beam_nodes, cb_list

# ==========================================
# 2. Precompute system matrices K and Fg
# ==========================================
def precompute_system(data):
    n_bn = data['Num-beamnodes']
    L_e = data['Beam-length'] / (n_bn - 1)
    EA, EI = data['Beam-EA'], data['Beam-EI']

    total_dof = 3 * n_bn
    K = torch.zeros((total_dof, total_dof), dtype=torch.float64)
    for i in range(n_bn - 1):
        k_e = torch.zeros((6, 6), dtype=torch.float64)
        k_e[0,0] = k_e[3,3] = EA/L_e; k_e[0,3] = k_e[3,0] = -EA/L_e
        k_e[1,1] = k_e[4,4] = 12*EI/(L_e**3); k_e[1,4] = k_e[4,1] = -12*EI/(L_e**3)
        k_e[1,2] = k_e[2,1] = k_e[1,5] = k_e[5,1] = 6*EI/(L_e**2)
        k_e[4,2] = k_e[2,4] = k_e[4,5] = k_e[5,4] = -6*EI/(L_e**2)
        k_e[2,2] = k_e[5,5] = 4*EI/L_e; k_e[2,5] = k_e[5,2] = 2*EI/L_e
        K[3*i:3*i+6, 3*i:3*i+6] += k_e

    # Boundary conditions: pin left, roller right
    cons = [0, 1, 3*n_bn-3, 3*n_bn-1]
    for d in cons:
        K[d, :] = 0; K[:, d] = 0; K[d, d] = 1.0

    scale_factor = 1e7
    K = K / scale_factor

    # Compute Fg per unit Gb (scaled)
    Fg_unit = torch.zeros((total_dof, 1), dtype=torch.float64)
    for i in range(n_bn):
        val = 1.0 * L_e
        if i == 0 or i == n_bn - 1: val *= 0.5
        Fg_unit[3*i + 1, 0] = val
    Fg_unit[2, 0] = 1.0 * (L_e**2) / 12.0
    Fg_unit[3*(n_bn-1) + 2, 0] = -1.0 * (L_e**2) / 12.0
    Fg_unit = Fg_unit / scale_factor

    return K, Fg_unit, cons

# ==========================================
# 3. Network architecture and loss function
# ==========================================
class BeamSurrogateNet(nn.Module):
    def __init__(self, out_dim, cons_dofs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(15, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, out_dim)
        )
        self.register_buffer('mask', torch.ones(out_dim, dtype=torch.float64))
        for d in cons_dofs:
            self.mask[d] = 0.0

    def forward(self, x):
        return self.net(x) * self.mask


class PhysicsLoss:
    def __init__(self, K, Fg_unit, CB, cons):
        self.K = K.cuda() if torch.cuda.is_available() else K
        self.Fg_unit = Fg_unit.cuda() if torch.cuda.is_available() else Fg_unit
        self.CB = torch.tensor(CB).cuda() if torch.cuda.is_available() else torch.tensor(CB)
        self.cons = cons
        self.scale_factor = 1e7

    def __call__(self, p_forces, D_pred, Gb):
        bs = p_forces.shape[0]
        device = p_forces.device

        T = (torch.abs(p_forces) * 1000.0) / self.scale_factor

        F_c = torch.zeros((bs, self.K.shape[0], 1), dtype=torch.float64, device=device)
        for i in range(14):
            node_idx = self.CB[i, 0].long()
            cosa, sina = self.CB[i, 1], self.CB[i, 2]
            F_c[:, 3*node_idx, 0] += T[:, i] * cosa
            F_c[:, 3*node_idx+1, 0] += T[:, i] * sina

        F_total = Gb.unsqueeze(-1) * self.Fg_unit.unsqueeze(0) + F_c
        for d in self.cons: F_total[:, d, 0] = 0.0

        D_pred = D_pred.reshape(bs, -1, 1)
        K_batch = self.K.unsqueeze(0).expand(bs, -1, -1)

        # Normalized loss: D_pred ≈ K^{-1} * F  (divide both sides of K*D = F by K)
        D_exact = torch.linalg.solve(K_batch, F_total)
        loss = torch.mean(torch.square(D_pred - D_exact))

        return loss


# ==========================================
# [Core] Constrained force sampling
# ==========================================
def generate_sampled_forces(bs, mins_t, maxs_t, cb_list, device):
    """
    Generate cable force samples with vertical resultant
    constrained to [0.9, 1.1] * 19080 kN.
    """
    # 1. Uniform random sampling in raw bounds
    p_base = mins_t + torch.rand(bs, 14, device=device, dtype=torch.float64) * (maxs_t - mins_t)

    # 2. Compute current Y-direction resultant force (kN)
    sina = torch.tensor(cb_list[:, 2], device=device, dtype=torch.float64)
    current_fy_sum_kn = torch.sum(p_base * 1000.0 * sina, dim=1, keepdim=True)

    # 3. Random target resultant within [0.9, 1.1] * 19080 kN
    TARGET_FY_SUM = 19080.0
    target_min = 0.9 * TARGET_FY_SUM
    target_max = 1.1 * TARGET_FY_SUM
    target_fy_sum_kn = target_min + torch.rand(bs, 1, device=device, dtype=torch.float64) * (target_max - target_min)

    # 4. Proportional scaling to match target
    scale_factor = target_fy_sum_kn / current_fy_sum_kn
    p_scaled = p_base * scale_factor

    # 5. Clamp to relaxed bounds to prevent extreme values
    p_final = torch.clamp(p_scaled, mins_t * 0.8, maxs_t * 1.2)

    return p_final


# ==========================================
# 4. Validation: displacement error
# ==========================================
def check_val_error(model, K_sys, Fg_unit, CB, cons, device, gb_min, gb_max):
    model.eval()
    scale_factor = 1e7
    with torch.no_grad():
        bs = 100
        mins_t = torch.tensor(FORCE_MINS, device=device, dtype=torch.float64)
        maxs_t = torch.tensor(FORCE_MAXS, device=device, dtype=torch.float64)

        p_sample = generate_sampled_forces(bs, mins_t, maxs_t, CB, device)
        gb_sample = gb_min + torch.rand(bs, 1, device=device, dtype=torch.float64) * (gb_max - gb_min)
        x_sample = torch.cat([p_sample, gb_sample], dim=1)
        D_pred = model(x_sample).reshape(bs, -1, 3)

        T = (torch.abs(p_sample) * 1000.0) / scale_factor
        K_dev = K_sys.to(device)
        F_c = torch.zeros((bs, K_dev.shape[0], 1), dtype=torch.float64, device=device)
        CB_tensor = torch.tensor(CB, device=device)

        for i in range(14):
            node_idx = CB_tensor[i, 0].long()
            cosa, sina = CB_tensor[i, 1], CB_tensor[i, 2]
            F_c[:, 3*node_idx, 0] += T[:, i] * cosa
            F_c[:, 3*node_idx+1, 0] += T[:, i] * sina

        F_tot = gb_sample.unsqueeze(-1) * Fg_unit.to(device).unsqueeze(0) + F_c
        for d in cons:
            F_tot[:, d, 0] = 0.0

        K_batch = K_dev.unsqueeze(0).expand(bs, -1, -1)
        D_exact = torch.linalg.solve(K_batch, F_tot).reshape(bs, -1, 3)

        mae_uy = torch.mean(torch.abs(D_pred[:, :, 1] - D_exact[:, :, 1])).item() * 1000.0

    model.train()
    return mae_uy


# ==========================================
# 5. Manual force prediction
# ==========================================
def predict_from_manual_forces(model, manual_forces_kn, Gb=-63.6):
    model.eval()
    device = next(model.parameters()).device

    target_scaled = np.array(manual_forces_kn, dtype=np.float64) / 1000.0
    fitted_forces = torch.from_numpy(target_scaled).unsqueeze(0).to(device)
    gb_tensor = torch.tensor([[Gb]], device=device, dtype=torch.float64)
    x_input = torch.cat([fitted_forces, gb_tensor], dim=1)

    with torch.no_grad():
        raw_output = model(x_input)
        displacement_matrix = raw_output.cpu().numpy().reshape(-1, 3)

    return displacement_matrix, target_scaled, fitted_forces


# ==========================================
# 6. Main: Adam + LBFGS training
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    beam_coords, cb_list = get_exact_geometry(model_data)
    K_sys, Fg_unit, cons_dofs = precompute_system(model_data)

    model = BeamSurrogateNet(out_dim=3*model_data['Num-beamnodes'], cons_dofs=cons_dofs).to(device).double()
    loss_fn = PhysicsLoss(K_sys, Fg_unit, cb_list, cons_dofs)

    mins_t = torch.tensor(FORCE_MINS, device=device, dtype=torch.float64)
    maxs_t = torch.tensor(FORCE_MAXS, device=device, dtype=torch.float64)
    gb_min_t = torch.tensor(GB_MIN, device=device, dtype=torch.float64)
    gb_max_t = torch.tensor(GB_MAX, device=device, dtype=torch.float64)

    print(">>> Phase 1: Adam optimizer (coarse tuning)...")
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=4e-4)
    for epoch in range(30001):
        p_batch = generate_sampled_forces(128, mins_t, maxs_t, cb_list, device)
        gb_batch = gb_min_t + torch.rand(128, 1, device=device, dtype=torch.float64) * (gb_max_t - gb_min_t)
        x_batch = torch.cat([p_batch, gb_batch], dim=1)

        optimizer_adam.zero_grad()
        loss = loss_fn(p_batch, model(x_batch), gb_batch)
        loss.backward()
        optimizer_adam.step()

        if epoch % 2000 == 0:
            mae_mm = check_val_error(model, K_sys, Fg_unit, cb_list, cons_dofs, device, gb_min_t, gb_max_t)
            print(f"Adam Epoch {epoch:5d} | Loss: {loss.item():.6e} | Uy MAE: {mae_mm:.4f} mm")

    print("\n>>> Phase 2: LBFGS optimizer (fine tuning)...")
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=0.1,
        max_iter=150,
        history_size=100,
        tolerance_grad=1e-7,
        line_search_fn="strong_wolfe"
    )

    p_fixed = generate_sampled_forces(2000, mins_t, maxs_t, cb_list, device)
    gb_fixed = gb_min_t + torch.rand(2000, 1, device=device, dtype=torch.float64) * (gb_max_t - gb_min_t)
    x_fixed = torch.cat([p_fixed, gb_fixed], dim=1)

    def closure():
        optimizer_lbfgs.zero_grad()
        loss = loss_fn(p_fixed, model(x_fixed), gb_fixed)
        loss.backward()
        return loss

    for i in range(301):
        loss = optimizer_lbfgs.step(closure)
        mae_mm = check_val_error(model, K_sys, Fg_unit, cb_list, cons_dofs, device, gb_min_t, gb_max_t)
        print(f"LBFGS Step {i:2d} | Loss: {loss.item():.6e} | Uy MAE: {mae_mm:.4f} mm")

    # ==========================================
    # 7. Phase 3: Surrogate-based cable force optimization
    # ==========================================
    print("\n" + "="*50)
    print(">>> Phase 3: Frozen surrogate — optimize 14 cable forces")
    print("    to minimize mid-span vertical displacement.")
    print("="*50)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Precompute cable direction sines for resultant-force constraint
    sina_t = torch.tensor(cb_list[:, 2], device=device, dtype=torch.float64)

    def obj_and_grad(p_numpy):
        gb_val = torch.tensor([[-63.6]], device=device, dtype=torch.float64)  # fixed at true value
        p_tensor = torch.tensor(p_numpy, dtype=torch.float64, device=device).unsqueeze(0)
        p_tensor.requires_grad_(True)

        x_tensor = torch.cat([p_tensor, gb_val], dim=1)
        disp = model(x_tensor).reshape(-1, 3)
        uy_all = disp[:, 1]

        # 1. Primary objective: minimize vertical displacement
        loss_disp = torch.mean(torch.square(uy_all)) * 1e6

        # 2. Soft constraint: cable resultant should approach 19080 kN
        forces_kn = p_tensor * 1000.0
        fy_sum_kn = torch.sum(forces_kn * sina_t)
        loss_fy = torch.square(fy_sum_kn - 19080.0) * 1e-4

        # 3. Smoothness regularization: penalize adjacent force oscillation
        left_diff = forces_kn[:, 1:7] - forces_kn[:, 0:6]
        right_diff = forces_kn[:, 8:14] - forces_kn[:, 7:13]
        loss_smooth = (torch.sum(torch.square(left_diff)) + torch.sum(torch.square(right_diff))) * 1e-5

        total_loss = loss_disp + loss_fy + loss_smooth + 1e-6 * torch.norm(p_tensor)

        total_loss.backward()

        obj_val = total_loss.item()
        grad_val = p_tensor.grad.cpu().numpy().astype(np.float64).flatten()
        return obj_val, grad_val

    bounds = list(zip(FORCE_MINS, FORCE_MAXS))

    p0_guess = np.array(FORCE_MINS) + (np.array(FORCE_MAXS) - np.array(FORCE_MINS)) / 2.0

    from scipy.optimize import minimize

    print("Running L-BFGS-B optimization...")
    t_start = time.time()
    res = minimize(
        fun=obj_and_grad,
        x0=p0_guess,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'disp': False, 'ftol': 1e-12, 'gtol': 1e-12}
    )
    t_end = time.time()

    best_p = res.x
    print(f"Optimization finished! Time: {t_end - t_start:.4f} s")
    print(f"Optimizer status: {res.message}")

    best_p_tensor = torch.tensor(best_p, dtype=torch.float64, device=device).unsqueeze(0)
    gb_opt = torch.tensor([[-63.6]], device=device, dtype=torch.float64)
    x_opt = torch.cat([best_p_tensor, gb_opt], dim=1)
    with torch.no_grad():
        opt_disp = model(x_opt).reshape(-1, 3).cpu().numpy()

    print("-" * 50)
    print(f"Optimized mid-span Uy: {opt_disp[15, 1]:.6e} m ({opt_disp[15, 1]*1000:.4f} mm)")

    opt_forces_kn = best_p * 1000.0

    print("\nOptimal cable forces (kN):")
    for i, f in enumerate(opt_forces_kn):
        print(f"Cable {i+1:<2} : {f:>10.2f} kN")
    print("="*50)
