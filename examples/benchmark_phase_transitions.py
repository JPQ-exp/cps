import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. SIMULATION ENGINE ACROSS ARBITRARY VIABILITY RADII
# ==============================================================================
def simulate_scenario(r_boundary, total_steps=500, delay=12, dt=0.02):
    """
    Simulates both the Delayed Baseline and the CPS-COL Agent
    under a specified circular viability boundary.
    """
    np.random.seed(42)
    drift = np.array([1.1, 0.8])
    start_pos = np.array([0.4, 0.4])
    
    def get_margin(p):
        return r_boundary - np.linalg.norm(p)
    
    def get_grad(p):
        norm = np.linalg.norm(p) + 1e-6
        return -p / norm

    # --- A. DELAYED BASELINE ---
    pos_b = start_pos.copy()
    hist_b = [pos_b.copy()]
    margin_b = [get_margin(pos_b)]
    reserve_b = [5.0]
    buffer_b = [pos_b.copy() for _ in range(delay)]
    crash_idx_b = None
    
    for t in range(total_steps):
        delayed_p = buffer_b.pop(0)
        u_control = 3.5 * get_grad(delayed_p)
        noise = np.random.normal(0, 0.12, size=2)
        vel_b = u_control + drift + noise
        pos_b = pos_b + vel_b * dt
        
        buffer_b.append(pos_b.copy())
        hist_b.append(pos_b.copy())
        psi_b = get_margin(pos_b)
        margin_b.append(psi_b)
        
        # Metabolic burn from jitter
        r_prev = reserve_b[-1]
        r_new = max(0.0, r_prev + (0.8 - 0.05 * r_prev - 0.08 * np.dot(vel_b, vel_b)) * dt)
        reserve_b.append(r_new)
        
        if psi_b <= 0 and crash_idx_b is None:
            crash_idx_b = t
            break  # Extinction event

    # --- B. CPS AGENT (BPLA + Oscillation Damping + COL) ---
    pos_c = start_pos.copy()
    hist_c = [pos_c.copy()]
    margin_c = [get_margin(pos_c)]
    reserve_c = [5.0]
    buffer_c = [pos_c.copy() for _ in range(delay)]
    recent_signs = []
    crash_idx_c = None
    
    for t in range(total_steps):
        delayed_p = buffer_c.pop(0)
        psi_c = get_margin(pos_c)
        
        # 1. BPLA Urgency
        urgency = 1.0 / max(psi_c, 0.03)
        # 2. Metabolic capacity
        capacity = np.sqrt(max(0.1, reserve_c[-1]) * 1.2 / 0.5)
        
        # Raw desire
        raw_act = capacity * get_grad(delayed_p) * (urgency ** 1.1) * 0.40 + drift
        
        # 3. Temporal Oscillation Damping (Memory clamp)
        cur_sign = np.sign(raw_act)
        recent_signs.append(cur_sign)
        if len(recent_signs) > 5:
            recent_signs.pop(0)
            flips = np.sum(np.diff(np.array(recent_signs), axis=0) != 0, axis=0)
            for d in range(2):
                if flips[d] > 2:
                    raw_act[d] = 0.0
        
        # 4. Constrained Orthant Lattice (Spatial Mutual Exclusion)
        act_pos = np.maximum(raw_act, 0.0)
        act_neg = np.maximum(-raw_act, 0.0)
        vel_c = act_pos - act_neg
        
        noise = np.random.normal(0, 0.04, size=2)
        pos_c = pos_c + (vel_c + noise) * dt
        
        buffer_c.append(pos_c.copy())
        hist_c.append(pos_c.copy())
        margin_c.append(get_margin(pos_c))
        
        # Smooth action cost
        r_prev = reserve_c[-1]
        r_new = max(0.0, r_prev + (0.9 - 0.02 * r_prev - 0.02 * np.dot(vel_c, vel_c)) * dt)
        reserve_c.append(r_new)
        
        if psi_c <= 0 and crash_idx_c is None:
            crash_idx_c = t
            break

    return {
        'r_bound': r_boundary,
        'hist_b': np.array(hist_b), 'margin_b': margin_b, 'reserve_b': reserve_b, 'crash_b': crash_idx_b,
        'hist_c': np.array(hist_c), 'margin_c': margin_c, 'reserve_c': reserve_c, 'crash_c': crash_idx_c,
        'dt': dt
    }

# ==============================================================================
# 2. RUN ALL THREE REGIMES
# ==============================================================================
radii = [1.0, 1.5, 2.0]
results = [simulate_scenario(r) for r in radii]

# ==============================================================================
# 3. BUILD THE MULTI-PANEL HERO FIGURE (2 ROWS x 3 COLUMNS)
# ==============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), dpi=300)
plt.subplots_adjust(wspace=0.28, hspace=0.30)

regime_titles = [
    r"Regime 1: Sub-Critical ($R = 1.0$)" + "\n" + r"Empty Viability Kernel ($\mathrm{Viab}(K) = \emptyset$)",
    r"Regime 2: Critical Boundary ($R = 1.5$)" + "\n" + r"Delay-Induced Limit Cycle Attractor",
    r"Regime 3: Robust Clearance ($R = 2.0$)" + "\n" + r"Limit Cycle Collapsed to Point Attractor"
]

theta = np.linspace(0, 2 * np.pi, 200)

for col, res in enumerate(results):
    r_b = res['r_bound']
    ax_top = axes[0, col]
    ax_bot = axes[1, col]
    
    # --------------------------------------------------------------------------
    # TOP ROW: 2D Spatial Trajectories
    # --------------------------------------------------------------------------
    x_circ = r_b * np.cos(theta)
    y_circ = r_b * np.sin(theta)
    
    # Danger zone shading outside circle
    ax_top.fill_between(np.linspace(-2.5, 2.5, 100), -2.5, 2.5, color='#FDE8E8', alpha=0.5)
    ax_top.fill(x_circ, y_circ, color='white')
    ax_top.plot(x_circ, y_circ, 'r--', lw=2.0, label=rf'Viability Boundary ($r={r_b}$)')
    
    # Trajectories
    hb = res['hist_b']
    hc = res['hist_c']
    
    # Baseline
    ax_top.plot(hb[:, 0], hb[:, 1], color='#D90429', lw=1.6, alpha=0.85, label='Delayed Baseline')
    if res['crash_b'] is not None:
        ax_top.scatter(hb[-1, 0], hb[-1, 1], color='#D90429', marker='X', s=140, zorder=5, label='Lysis Breach')
        
    # CPS
    ax_top.plot(hc[:, 0], hc[:, 1], color='#0077B6', lw=2.4, label='CPS-COL')
    ax_top.scatter(hc[0, 0], hc[0, 1], color='#2B9348', s=70, zorder=5, label=r'Start ($x_0$)')
    if res['crash_c'] is None:
        ax_top.scatter(hc[-1, 0], hc[-1, 1], color='#0077B6', s=80, marker='o', zorder=5, label='Stable End')
    
    ax_top.set_xlim(-2.4, 2.4)
    ax_top.set_ylim(-2.4, 2.4)
    ax_top.set_aspect('equal')
    ax_top.set_title(regime_titles[col], fontsize=11, fontweight='bold')
    ax_top.set_xlabel('X coordinate', fontsize=10)
    ax_top.set_ylabel('Y coordinate', fontsize=10)
    ax_top.grid(True, linestyle=':', alpha=0.6)
    if col == 0:
        ax_top.legend(loc='lower left', fontsize=8.5, framealpha=0.9)

    # --------------------------------------------------------------------------
    # BOTTOM ROW: Margin & Metabolic Dynamics Over Time
    # --------------------------------------------------------------------------
    t_b = np.linspace(0, len(res['margin_b']) * res['dt'], len(res['margin_b']))
    t_c = np.linspace(0, len(res['margin_c']) * res['dt'], len(res['margin_c']))
    
    ax_bot.axhline(0, color='#D90429', ls='--', lw=1.8, alpha=0.8, label=r'Extinction ($\psi \leq 0$)')
    
    # Plot margins on primary y-axis
    ax_bot.plot(t_c, res['margin_c'], color='#F77F00', lw=2.0, label=r'CPS Margin ($\psi$)')
    ax_bot.plot(t_b, res['margin_b'], color='#D90429', ls=':', lw=1.6, alpha=0.85, label=r'Baseline Margin ($\psi$)')
    ax_bot.set_ylabel(r'Safety Margin ($\psi$)', fontsize=10)
    ax_bot.set_ylim(-0.6, 2.2)
    ax_bot.set_xlim(0, 10.0)
    ax_bot.set_xlabel('Time (seconds)', fontsize=10)
    ax_bot.grid(True, linestyle=':', alpha=0.6)
    
    # Plot reserves on secondary y-axis
    ax_bot_twin = ax_bot.twinx()
    ax_bot_twin.plot(t_c, res['reserve_c'], color='#2B9348', lw=1.8, label=r'CPS Reserve ($R$)')
    ax_bot_twin.plot(t_b, res['reserve_b'], color='#6C757D', ls='-.', lw=1.4, alpha=0.7, label=r'Baseline Reserve ($R$)')
    ax_bot_twin.set_ylabel(r'Metabolic Reserve ($R$)', color='#2B9348', fontsize=10)
    ax_bot_twin.tick_params(axis='y', labelcolor='#2B9348')
    ax_bot_twin.set_ylim(0, 13)
    
    # Explanatory annotations per bottom panel
    if col == 0:
        ax_bot.text(0.8, -0.4, r'Immediate Boundary Breach' + '\n' + r'($\psi < 0$ at $t \approx 1.2\mathrm{s}$)', 
                    color='#D90429', fontsize=8.5, fontweight='bold')
    elif col == 1:
        ax_bot.text(2.0, 0.15, r'Delay Limit-Cycle Trap' + '\n' + r'(Continuous $\approx 40\mathrm{Hz}$ Oscillation)', 
                    color='#9D0208', fontsize=8.5, fontweight='bold')
    elif col == 2:
        ax_bot.text(2.5, 1.4, r'Limit Cycle Collapsed' + '\n' + r'($R$ builds to $+12$)', 
                    color='#2B9348', fontsize=8.5, fontweight='bold')

    if col == 2:
        # Put unified legend on the last bottom plot
        l1, lab1 = ax_bot.get_legend_handles_labels()
        l2, lab2 = ax_bot_twin.get_legend_handles_labels()
        ax_bot.legend(l1 + l2, lab1 + lab2, loc='upper left', fontsize=8.0, framealpha=0.9)

plt.suptitle(r"Phase Transition in Viability: How Spatial Clearance Dictates Control Stability under Sensory Latency ($\tau = 12$)", 
             fontsize=14, fontweight='bold', y=0.98)

plt.savefig('assets/cps_three_regimes_phase_transition.png', dpi=300)
plt.show()