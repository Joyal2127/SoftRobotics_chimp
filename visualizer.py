"""
visualizer.py — Chimpanzee Voxelbot 3D Animator
Reads best_parameters.csv + fitness_log.csv produced by the GA.

Usage:
    python visualizer.py           # animate (interactive)
    python visualizer.py --save    # also write chimp_locomotion.gif

Requirements:  numpy  matplotlib
Optional:      pillow  (only for --save)
"""

import sys
import os
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

# ── Mirror C++ constants exactly ──────────────────────────────────────────────
TWO_PI      = 2.0 * np.pi
GROUND_Z    = -0.5
OMEGA       = 8.0
DT          = 0.005
SPRING_K    = 3000.0
SPRING_DAMP = 30.0
NODE_MASS   = 0.1
GRAVITY     = -9.81
RESTITUTION = 0.4
FRICTION    = 0.7

VIZ_TOTAL_TIME = 6.0   # longer window than training — show a full gait
FRAME_EVERY    = 10    # record 1 frame every N physics steps → ~20 fps


# ── Robot geometry (mirrors create_voxelbot) ──────────────────────────────────
_NODE_POS = np.array([
    [ 1.0,  0.5,  0.8],  # 0 Left Shoulder
    [ 1.0, -0.5,  0.8],  # 1 Right Shoulder
    [-1.0,  0.4,  0.3],  # 2 Left Hip
    [-1.0, -0.4,  0.3],  # 3 Right Hip
    [ 1.5,  0.7, -0.5],  # 4 Left Hand
    [ 1.5, -0.7, -0.5],  # 5 Right Hand
    [-1.5,  0.5, -0.5],  # 6 Left Foot
    [-1.5, -0.5, -0.5],  # 7 Right Foot
], dtype=float)

# (n1, n2, is_actuator) — same insertion order as C++
_TOPOLOGY = [
    (0,1,False),(2,3,False),(0,2,False),(1,3,False),(0,3,False),(1,2,False),
    (4,0,True), (5,1,True), (4,2,False),(5,3,False),
    (6,2,True), (7,3,True), (6,0,False),(7,1,False),
    (4,1,False),(5,0,False),(6,3,False),(7,2,False),
]

_PAIRS     = np.array([(n1, n2) for n1, n2, _ in _TOPOLOGY])
_IS_ACT    = np.array([m          for  _,  _, m in _TOPOLOGY], dtype=bool)
_REST_LEN  = np.array([
    np.linalg.norm(_NODE_POS[n2] - _NODE_POS[n1]) for n1, n2, _ in _TOPOLOGY
])


# ── Physics (vectorised, mirrors simulate_step) ───────────────────────────────
def make_state():
    """Return mutable physics state arrays."""
    return {
        'pos':   _NODE_POS.copy(),
        'vel':   np.zeros_like(_NODE_POS),
        'force': np.zeros_like(_NODE_POS),
    }


def simulate_step(state, amplitude, phase, t, dt):
    pos, vel, force = state['pos'], state['vel'], state['force']
    n1 = _PAIRS[:, 0]
    n2 = _PAIRS[:, 1]

    # Gravity
    force[:] = 0.0
    force[:, 2] = NODE_MASS * GRAVITY

    # Spring + dashpot (vectorised over all springs)
    delta = pos[n2] - pos[n1]                            # (S, 3)
    L     = np.linalg.norm(delta, axis=1)                # (S,)
    valid = L > 1e-9
    Lsafe = np.where(valid, L, 1.0)
    d     = delta / Lsafe[:, None]                       # unit vectors
    d[~valid] = 0.0

    target_L        = _REST_LEN.copy()
    target_L[_IS_ACT] += amplitude[_IS_ACT] * np.sin(OMEGA * t + phase[_IS_ACT])

    spring_f = SPRING_K * (L - target_L)                 # (S,)
    rel_vel  = vel[n2] - vel[n1]                         # (S, 3)
    damp_f   = SPRING_DAMP * np.einsum('ij,ij->i', rel_vel, d)  # (S,)

    fv = d * ((spring_f + damp_f) * valid)[:, None]      # (S, 3)
    np.add.at(force, n1, fv)
    np.add.at(force, n2, -fv)

    # Symplectic Euler
    vel += (force / NODE_MASS) * dt
    pos += vel * dt

    # Ground contact
    below = pos[:, 2] < GROUND_Z
    if below.any():
        pos[below, 2] = GROUND_Z
        bouncing = below & (vel[:, 2] < 0)
        vel[bouncing, 2] *= -RESTITUTION

        nf  = np.maximum(0.0, -force[below, 2])
        fdv = FRICTION * nf * dt / NODE_MASS
        spd = np.hypot(vel[below, 0], vel[below, 1])
        safe = spd > 1e-9
        scl  = np.where(safe, np.maximum(0.0, spd - fdv) / np.where(safe, spd, 1.0), 1.0)
        vel[below, 0] *= scl
        vel[below, 1] *= scl


# ── Pre-run simulation and collect frames ─────────────────────────────────────
def run_simulation(amplitude, phase):
    state  = make_state()
    frames = []   # list of node position arrays
    exts   = []   # per-spring extension ratio (for colouring)
    steps  = int(VIZ_TOTAL_TIME / DT)

    for step in range(steps):
        if step % FRAME_EVERY == 0:
            frames.append(state['pos'].copy())
            L    = np.linalg.norm(state['pos'][_PAIRS[:,1]] - state['pos'][_PAIRS[:,0]], axis=1)
            ext  = np.where(_IS_ACT, (L - _REST_LEN) / (_REST_LEN + 1e-9), 0.0)
            exts.append(ext.copy())
        simulate_step(state, amplitude, phase, step * DT, DT)

    return frames, exts


# ── Load output files ─────────────────────────────────────────────────────────
def load_spring_params(path='best_parameters.csv'):
    amplitude = np.zeros(len(_TOPOLOGY))
    phase     = np.zeros(len(_TOPOLOGY))
    with open(path) as f:
        for row in csv.DictReader(f):
            i = int(row['SpringIndex'])
            amplitude[i] = float(row['Amplitude'])
            phase[i]     = float(row['Phase'])
    return amplitude, phase


def load_fitness_log(path='fitness_log.csv'):
    gens, bests, means = [], [], []
    i0, i1, i2, i3 = [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            gens.append(int(row['Generation']))
            bests.append(float(row['GlobalBest']))
            means.append(float(row['GlobalMean']))
            if 'I0Best' in row:
                i0.append(float(row['I0Best']))
                i1.append(float(row['I1Best']))
                i2.append(float(row['I2Best']))
                i3.append(float(row['I3Best']))
    return (np.array(gens), np.array(bests), np.array(means),
            np.array(i0), np.array(i1), np.array(i2), np.array(i3))


# ── Colour helpers ────────────────────────────────────────────────────────────
def spring_color(ext, is_act):
    if not is_act:
        return '#4a6070'     # passive: steel blue-grey
    if ext > 0:              # contracted: warm orange-red
        r = min(1.0, 0.5 + ext * 3.5)
        return (r, max(0.1, 0.45 - ext * 2), 0.1)
    else:                    # extended: cool cyan
        b = min(1.0, 0.5 + abs(ext) * 3.5)
        return (0.1, max(0.1, 0.45 - abs(ext) * 2), b)


_NODE_COLORS = ['#ffffff'] * 4 + ['#ff6644', '#ff6644', '#44aaff', '#44aaff']
# hands (4,5) red-orange; feet (6,7) cyan; torso white


# ── Animation ─────────────────────────────────────────────────────────────────
def build_and_run(amplitude, phase, save_gif):
    print("Running simulation …")
    frames, exts = run_simulation(amplitude, phase)
    print(f"  {len(frames)} frames at {1/(FRAME_EVERY*DT):.0f} fps  ({VIZ_TOTAL_TIME}s)")

    has_log = os.path.exists('fitness_log.csv')

    # Figure layout
    fig = plt.figure(figsize=(14 if has_log else 9, 7), facecolor='#0d1117')
    if has_log:
        ax3d   = fig.add_subplot(1, 2, 1, projection='3d')
        ax_evo = fig.add_subplot(1, 2, 2)
        ax_evo.set_facecolor('#161b22')
        _draw_evolution(ax_evo)
    else:
        ax3d   = fig.add_subplot(1, 1, 1, projection='3d')
        ax_evo = None
    ax3d.set_facecolor('#161b22')

    init_x = frames[0][:, 0].mean()

    def update(fi):
        ax3d.cla()
        pos = frames[fi]
        ext = exts[fi]
        t   = fi * FRAME_EVERY * DT
        com_x = pos[:, 0].mean()

        # Ground plane — slides with robot to keep it centred
        gx = np.array([com_x - 3.0, com_x + 3.0])
        gy = np.array([-2.2, 2.2])
        GX, GY = np.meshgrid(gx, gy)
        ax3d.plot_surface(GX, GY, np.full_like(GX, GROUND_Z),
                          alpha=0.15, color='#8b7355', zorder=0, linewidth=0)

        # Springs
        for i, (n1i, n2i) in enumerate(_PAIRS):
            p1, p2 = pos[n1i], pos[n2i]
            col = spring_color(ext[i], _IS_ACT[i])
            lw  = 3.2 if _IS_ACT[i] else 1.6
            ax3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                      color=col, linewidth=lw, alpha=0.95)

        # Nodes
        for ni, p in enumerate(pos):
            ax3d.scatter([p[0]], [p[1]], [p[2]],
                         color=_NODE_COLORS[ni], s=90, zorder=6, depthshade=False)

        # Axes
        ax3d.set_xlim(com_x - 2.8, com_x + 2.8)
        ax3d.set_ylim(-2.2, 2.2)
        ax3d.set_zlim(-0.9, 2.2)
        ax3d.set_xlabel('X forward', color='#8b9dc3', labelpad=4, fontsize=8)
        ax3d.set_ylabel('Y lateral', color='#8b9dc3', labelpad=4, fontsize=8)
        ax3d.set_zlabel('Z up',      color='#8b9dc3', labelpad=4, fontsize=8)
        ax3d.tick_params(colors='#555', labelsize=7)
        ax3d.xaxis.pane.fill = ax3d.yaxis.pane.fill = ax3d.zaxis.pane.fill = False
        ax3d.xaxis.pane.set_edgecolor('#1f2937')
        ax3d.yaxis.pane.set_edgecolor('#1f2937')
        ax3d.zaxis.pane.set_edgecolor('#1f2937')
        ax3d.set_title(
            f'Chimpanzee Voxelbot   t = {t:.2f} s   Δx = {com_x - init_x:+.3f} m',
            color='white', pad=8, fontsize=10
        )
        ax3d.view_init(elev=20, azim=-55 + fi * 0.07)  # slow camera pan

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=50, blit=False
    )
    plt.tight_layout(pad=1.2)

    if save_gif:
        out = 'chimp_locomotion.gif'
        print(f"Saving {out} …")
        try:
            ani.save(out, writer='pillow', fps=20, dpi=90)
            print(f"Saved {out}")
        except Exception as e:
            print(f"GIF save failed: {e}\n  Install pillow:  pip install pillow")

    plt.show()


def _draw_evolution(ax):
    try:
        gens, bests, means, i0, i1, i2, i3 = load_fitness_log()
    except Exception:
        ax.text(0.5, 0.5, 'fitness_log.csv\nnot found',
                ha='center', va='center', color='white', transform=ax.transAxes)
        return

    island_colors = ['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff']
    labels        = ['Island 0', 'Island 1', 'Island 2', 'Island 3']

    if len(i0):
        for data, col, lbl in zip([i0, i1, i2, i3], island_colors, labels):
            ax.plot(gens, data, color=col, lw=0.8, alpha=0.55, label=lbl)

    ax.plot(gens, bests, color='white',  lw=2.0, label='Global best', zorder=5)
    ax.plot(gens, means, color='#8b9dc3',lw=1.0, ls='--', alpha=0.7, label='Mean')
    ax.axhline(0, color='#444', ls=':', lw=0.8)

    ax.set_xlabel('Generation', color='#8b9dc3', fontsize=9)
    ax.set_ylabel('Fitness  (m)', color='#8b9dc3', fontsize=9)
    ax.set_title('Evolution Progress', color='white', fontsize=10)
    ax.legend(fontsize=7, framealpha=0.25, labelcolor='white')
    ax.tick_params(colors='#8b9dc3', labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor('#2d3748')
    ax.grid(True, alpha=0.15, color='white')


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    save = '--save' in sys.argv

    if not os.path.exists('best_parameters.csv'):
        print("best_parameters.csv not found — run the GA first:\n"
              "  g++ -O3 -march=native -fopenmp -std=c++17 PE_M_GA.cpp -o chimp_ga && ./chimp_ga")
        sys.exit(1)

    print("Loading best_parameters.csv …")
    amplitude, phase = load_spring_params()

    build_and_run(amplitude, phase, save)


if __name__ == '__main__':
    main()
