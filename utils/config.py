"""
Configuración global de parámetros para el proyecto.
"""

import numpy as np

# =============================================
# Parámetros del brazo robótico
# =============================================
ARM_CONFIG = {
    'l1': 0.5,          # Longitud del eslabón 1 (hombro a codo) [m]
    'l2': 0.4,          # Longitud del eslabón 2 (codo a muñeca) [m]
    'l3': 0.2,          # Longitud del eslabón 3 (muñeca a mano) [m]
    'q_min': np.array([-np.pi, -np.pi/2, -np.pi/2]),   # Límites inferiores [rad]
    'q_max': np.array([np.pi, np.pi/2, np.pi/2])       # Límites superiores [rad]
}

# =============================================
# Parámetros de simulación y control
# =============================================
SIM_CONFIG = {
    'dt': 0.05,                 # Paso de tiempo para simulaciones [s]
    'horizon': 30,              # Horizonte de predicción para MPC
    'max_iter': 1000,            # Máximo de iteraciones para solvers iterativos
    'tolerance': 0.01,           # Tolerancia para convergencia
    'stall_threshold': 1e-6,   # Umbral de cambio de vector estado. Si es menor, se empieza a contar
    'stall_steps': 5            # Pasos sin acción
}

# =============================================
# Parámetros del MPC centralizado
# =============================================
MPC_CONFIG = {
    'Q': 1e11 * np.array([[100,0,0],[0,30,0],[0,0,5]]),      # Peso error proporcional
    'R': 0.01 * np.eye(3),      # Peso esfuerzo de control
    'Q_term': 500.0,            
    'w_int': 1e6,
    'v_max': np.array([2.0, 2.0, 2.0])
}

# =============================================
# Parámetros del juego cooperativo
# =============================================
GAME_CONFIG = {
    'alpha': 0.1,               # Peso del costo de movimiento individual
    'beta': 100.0,               # Peso del error global de posición
    'max_game_iters': 100,       # Iteraciones máximas del juego
    'game_tol': 1e-6            # Tolerancia para equilibrio de Nash
}

# =============================================
# Parámetros del solver híbrido
# =============================================

HYBRID_CONFIG = {
    'horizon': 30,
    'dt': 0.05,
    'Q_pos': 100 * np.array([[1,0,0],[0,3,0],[0,0,5]]),
    'r_vel': 0.1,
    'w_acc': 0.01,
    'w_int': 1e6,          
    'Q_term': 200.0,       
    'game_iters': 100,
    'v_max': np.array([1.0, 1.0, 1.0])
}

# =============================================
# Parámetros para experimentos
# =============================================
EXP_CONFIG = {
    'random_seed': 42,          # Semilla para reproducibilidad
    'n_trials': 10,             # Número de repeticiones por experimento
    'targets': [                # Puntos objetivo de prueba (x, y, z) [m]
        np.array([0.6, 0.3, 0.4]),
        np.array([0.8, -0.2, 0.6]),
        np.array([0.5, 0.5, 0.8]),
        np.array([0.3, -0.4, 0.9]),
        np.array([0.7, 0.1, 0.2])
    ],
    'initial_q': np.array([0.1, 0.5, 0.3])  # Configuración inicial típica
}

# =============================================
# Visualización
# =============================================
VIZ_CONFIG = {
    'figsize': (10, 8),
    'dpi': 100,
    'save_format': 'png',
    'animation_interval': 0.05   # segundos entre frames
}