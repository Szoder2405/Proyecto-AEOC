"""
Prueba del solver MPC centralizado con CasADi.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arm_model.three_dof_arm import ThreeDOFArm
from arm_model.visualizer import ArmVisualizer
from solvers.mpc_centralized import CentralizedMPC
from utils.config import ARM_CONFIG, MPC_CONFIG, SIM_CONFIG
from utils.plotting import plot_joint_trajectories, plot_position_error


def main():
    print("=== Prueba del MPC Centralizado ===\n")
    
    # 1. Crear brazo
    arm = ThreeDOFArm(**ARM_CONFIG)
    
    # 2. Configuración inicial y objetivo
    q0 = np.array([0.2, 0.5, 0.1])  # rad
    p_target = np.array([0.9, 0.15, 0.6])  # metros
    
    print(f"Configuración inicial q0: {q0}")
    p0, _ = arm.forward_kinematics(q0)
    print(f"Posición inicial: {p0}")
    print(f"Objetivo: {p_target}")
    print(f"Distancia inicial: {np.linalg.norm(p0 - p_target):.4f} m\n")
    
    # 3. Crear MPC
    mpc = CentralizedMPC(
        arm,
        horizon=SIM_CONFIG['horizon'],
        dt=SIM_CONFIG['dt'],
        Q=MPC_CONFIG['Q'],
        R=MPC_CONFIG['R'],
        v_max=MPC_CONFIG['v_max']
    )
    
    # 4. Ejecutar lazo cerrado
    print("Ejecutando MPC en lazo cerrado...")
    result = mpc.run_closed_loop(
        q0, p_target,
        max_steps=SIM_CONFIG['max_iter'],
        tol=SIM_CONFIG['tolerance'],
        stall_threshold=SIM_CONFIG['stall_threshold'],
        stall_steps=SIM_CONFIG['stall_steps']
    )
    
    # 5. Resultados
    print(f"\nResultados:")
    print(f"  - Éxito: {result['success']}")
    print(f"  - Pasos: {result['steps']}")
    print(f"  - Error final: {result['error'][-1]:.5f} m")
    print(f"  - Posición final: {result['p'][-1]}")
    
    # 6. Visualización de trayectorias articulares
    plot_joint_trajectories(result['q'], mpc.dt, title="Trayectorias articulares - MPC")
    
    # 7. Visualización del error
    plot_position_error([result['error']], ['MPC'], 
                        title="Evolución del error de posición - MPC")
    
    # 8. Visualización 3D de la trayectoria
    viz = ArmVisualizer(arm, figsize=(10, 8))
    viz.animate_trajectory(
        [q for q in result['q']], 
        target=p_target, 
        save_path="mpc_trajectory.gif"
    )
    viz.close()
    
    # También mostrar el camino del efector final
    viz2 = ArmVisualizer(arm, figsize=(10, 8))
    viz2.plot_arm(result['q'][-1], show_target=p_target)
    viz2.plot_trajectory_path(result['q'], color='orange', label='Camino MPC')
    viz2.show()
    
    print("\n¡Prueba completada!")


if __name__ == "__main__":
    main()