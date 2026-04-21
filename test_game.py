"""
Prueba del controlador basado en juego cooperativo.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arm_model.three_dof_arm import ThreeDOFArm
from arm_model.visualizer import ArmVisualizer
from solvers.game_cooperative import CooperativeGameSolver
from utils.config import ARM_CONFIG, GAME_CONFIG, SIM_CONFIG
from utils.plotting import plot_joint_trajectories, plot_position_error


def main():
    print("=== Prueba del Controlador por Juego Cooperativo ===\n")
    
    arm = ThreeDOFArm(**ARM_CONFIG)
    
    q0 = np.array([0.2, 0.5, 0.1])
    p_target = np.array([0.9, 0.15, 0.6])  # mismo objetivo que en test MPC
    
    print(f"Configuración inicial: {q0}")
    p0, _ = arm.forward_kinematics(q0)
    print(f"Posición inicial: {p0}")
    print(f"Objetivo: {p_target}")
    print(f"Distancia inicial: {np.linalg.norm(p0 - p_target):.4f} m\n")
    
    # Crear solver de juego
    game_solver = CooperativeGameSolver(
        arm,
        alpha=GAME_CONFIG['alpha'],
        beta=GAME_CONFIG['beta'],
        max_iters=GAME_CONFIG['max_game_iters'],
        tol=GAME_CONFIG['game_tol'],
        dt=SIM_CONFIG['dt']
    )
    
    print("Ejecutando control en lazo cerrado con juego cooperativo...")
    result = game_solver.run_closed_loop(
        q0, p_target,
        max_steps=SIM_CONFIG['max_iter'],
        stall_threshold=SIM_CONFIG['stall_threshold'],
        stall_steps=SIM_CONFIG['stall_steps']
    )
    
    print(f"\nResultados:")
    print(f"  - Éxito: {result['success']}")
    print(f"  - Pasos: {result['steps']}")
    print(f"  - Error final: {result['error'][-1]:.5f} m")
    print(f"  - Posición final: {result['p'][-1]}")
    
    # Gráficas
    plot_joint_trajectories(result['q'], SIM_CONFIG['dt'], title="Trayectorias articulares - Juego Cooperativo")
    plot_position_error([result['error']], ['Juego Cooperativo'], 
                        title="Evolución del error de posición - Juego Cooperativo")
    
    # Visualización 3D
    viz = ArmVisualizer(arm, figsize=(10, 8))
    viz.animate_trajectory(
        [q for q in result['q']], 
        target=p_target, 
        save_path="game_trajectory.gif"
    )
    viz.close()
    
    viz2 = ArmVisualizer(arm, figsize=(10, 8))
    viz2.plot_arm(result['q'][-1], show_target=p_target)
    viz2.plot_trajectory_path(result['q'], color='orange', label='Camino Juego')
    viz2.show()
    
    print("\n¡Prueba completada!")


if __name__ == "__main__":
    main()