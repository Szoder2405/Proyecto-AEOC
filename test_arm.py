"""
Script de prueba para verificar el modelo del brazo y el visualizador.
"""

import numpy as np
import sys
import os

# Agregar el directorio raíz al path para imports relativos
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arm_model.three_dof_arm import ThreeDOFArm
from arm_model.visualizer import ArmVisualizer
from utils.config import ARM_CONFIG, EXP_CONFIG


def main():
    print("=== Prueba del modelo y visualizador del brazo 3-DOF ===\n")
    
    # 1. Crear el brazo con longitudes de la configuración
    arm = ThreeDOFArm(
        l1=ARM_CONFIG['l1'],
        l2=ARM_CONFIG['l2'],
        l3=ARM_CONFIG['l3']
    )
    print(f"Brazo creado: L1={arm.l1}, L2={arm.l2}, L3={arm.l3}")
    print(f"Límites articulares: {arm.q_min} a {arm.q_max} rad\n")
    
    # 2. Probar cinemática directa con una configuración
    q_test = np.array([0.5, 0.8, 0.3])  # ángulos de prueba
    print(f"Probando cinemática directa con q = {q_test}")
    pos, joint_positions = arm.forward_kinematics(q_test)
    print(f"Posición del efector final: {pos}")
    print(f"Posiciones de articulaciones: {len(joint_positions)} puntos\n")
    
    # 3. Probar jacobiano
    J = arm.jacobian(q_test)
    print("Jacobiano en q_test:")
    print(J)
    print()
    
    # 4. Verificar límites
    q_invalid = np.array([4.0, 0.5, 0.2])  # fuera de límite en hombro
    print(f"¿Configuración {q_invalid} dentro de límites? {arm.check_joint_limits(q_invalid)}")
    q_clipped = arm.clip_joints(q_invalid)
    print(f"Recortada a: {q_clipped}\n")
    
    # 5. Visualizar el brazo en 3D
    print("Mostrando visualización 3D del brazo...")
    viz = ArmVisualizer(arm, figsize=(10, 8))
    
    # Punto objetivo de ejemplo (dentro del espacio de trabajo)
    target = np.array([0.6, 0.3, 0.5])
    print(f"Objetivo: {target}")
    
    viz.plot_arm(q_test, color='blue', show_target=target)
    viz.show()
    
    # 6. Probar múltiples configuraciones en subplots
    configs = [
        (np.array([0.0, 0.5, 0.0]), "q1=0, q2=0.5, q3=0", "blue"),
        (np.array([0.5, 0.2, 0.8]), "q1=0.5, q2=0.2, q3=0.8", "green"),
        (np.array([-0.3, 0.9, -0.5]), "q1=-0.3, q2=0.9, q3=-0.5", "red"),
    ]
    
    print("\nComparando configuraciones en subplots...")
    viz.compare_configurations(configs, target=target)
    
    # 7. Generar GIF de una trayectoria simple
    print("\nGenerando GIF de trayectoria simple...")
    trajectory = []
    steps = 30
    for i in range(steps):
        t = i / (steps - 1)
        q = np.array([
            0.3 * np.sin(2 * np.pi * t),
            0.5 + 0.3 * np.cos(2 * np.pi * t),
            0.2 * np.sin(4 * np.pi * t)
        ])
        trajectory.append(q)

    # Guardar GIF en la carpeta actual
    viz.animate_trajectory(trajectory, target=None, save_path="trayectoria_brazo.gif")

    print("\n¡Prueba completada exitosamente!")


if __name__ == "__main__":
    main()