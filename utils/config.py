"""
Configuración global de parámetros para el proyecto.
"""

import numpy as np

# =============================================
# Parámetros del brazo robótico
# =============================================
ARM_CONFIG = {
    'l1': 0.5,
    'l2': 0.4,
    'l3': 0.2,
    'q_min': np.array([-np.pi, -np.pi/2, -np.pi/2]),
    'q_max': np.array([np.pi, np.pi/2, np.pi/2])
}

# =============================================
# Parámetros de simulación y control
# =============================================
SIM_CONFIG = {
    'dt': 0.05,
    'horizon': 10,          # MPC centralizado: reducido 30→10 (~9× menos trabajo)
    'max_iter': 1000,
    'tolerance': 0.01,
    'stall_threshold': 1e-6,
    'stall_steps': 5
}

# =============================================
# Parámetros del MPC centralizado
# =============================================
MPC_CONFIG = {
    'Q': 1e4 * np.array([[100, 0, 0],
                         [0,  30, 0],
                         [0,   0, 5]]),
    'R': 0.01 * np.eye(3),
    'Q_term': 500.0,
    'w_int': 0.0,
    'v_max': np.array([2.0, 2.0, 2.0])
}

# =============================================
# Parámetros del juego cooperativo
# =============================================
GAME_CONFIG = {
    'alpha': 0.1,
    'beta': 100.0,
    'max_game_iters': 100,
    'game_tol': 1e-6
}

# =============================================
# Parámetros del solver híbrido (DMPC Nash)
# Único cambio respecto al original: horizon 30→10.
# Todo lo demás igual al original para no introducir
# regresiones de velocidad.
# =============================================
HYBRID_CONFIG = {
    'horizon'   : 10,       # ÚNICO CAMBIO: 30→10 (~9× menos trabajo/articulación)
    'dt'        : 0.05,
    'Q_pos'     : 100 * np.array([[1, 0, 0],
                                   [0, 3, 0],
                                   [0, 0, 5]]),
    'r_vel'     : 0.1,
    'w_acc'     : 0.01,
    'w_int'     : 0.0,      # igual al original (ya estaba en 0 en el original)
    'Q_term'    : 200.0,    # igual al original
    'game_iters': 3,        # igual al original
    'nash_tol'  : None,     # no se usa (híbrido original no tiene parada anticipada)
    'v_max'     : np.array([1.0, 1.0, 1.0])   # igual al original
}

# =============================================
# Parámetros para experimentos
# =============================================
EXP_CONFIG = {
    'random_seed': 42,
    'n_trials': 10,
    'targets': [
        np.array([0.6, 0.3, 0.4]),
        np.array([0.8, -0.2, 0.6]),
        np.array([0.5, 0.5, 0.8]),
        np.array([0.3, -0.4, 0.9]),
        np.array([0.7, 0.1, 0.2])
    ],
    'initial_q': np.array([0.1, 0.5, 0.3])
}

# =============================================
# Visualización
# =============================================
VIZ_CONFIG = {
    'figsize': (10, 8),
    'dpi': 100,
    'save_format': 'mp4',
    'animation_interval': 0.05,
    'video_fps': 20,
    'video_skip_frames': 3
}
