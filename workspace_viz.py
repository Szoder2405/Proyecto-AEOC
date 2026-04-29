"""Script rápido para visualizar el espacio de trabajo del brazo 3-DOF."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arm_model.three_dof_arm import ThreeDOFArm
from utils.plotting import plot_workspace_analysis

def main():
    # Crear brazo con límites predeterminados (q1: ±180°, q2, q3: ±90°)
    arm = ThreeDOFArm()

    # Generar configuraciones aleatorias dentro de los límites articulares
    n_samples = 5000
    q_random = np.random.uniform(
        low=arm.q_min,
        high=arm.q_max,
        size=(n_samples, 3)
    )

    # Calcular posiciones del efector final
    points = []
    for q in q_random:
        p, _ = arm.forward_kinematics(q)
        points.append(p)
    points = np.array(points)

    # Visualizar espacio de trabajo
    plot_workspace_analysis(points, title="Espacio de trabajo - Brazo Antropomórfico 3-DOF")

if __name__ == "__main__":
    main()