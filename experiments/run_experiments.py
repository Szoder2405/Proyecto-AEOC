"""
Experimento comparativo: seguimiento de trayectoria con los tres solvers.
Se genera un GIF por solver mostrando trayectoria deseada vs real.
"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arm_model.three_dof_arm import ThreeDOFArm
from arm_model.visualizer import ArmVisualizer
from solvers.mpc_centralized import CentralizedMPC
from solvers.game_cooperative import CooperativeGameSolver
from solvers.hybrid import DistributedMPC
from utils.config import (ARM_CONFIG, SIM_CONFIG, MPC_CONFIG,
                          GAME_CONFIG, HYBRID_CONFIG)
from utils.plotting import plot_position_error


def generate_trajectory(center, radius, num_points):
    """Genera un círculo en 3D (plano YZ) centrado en `center`."""
    theta = np.linspace(0, 2*np.pi, num_points)
    x = center[0] * np.ones_like(theta) # mantiene X constante
    y = center[1] + radius * np.cos(theta)   
    z = center[2] + radius * np.sin(theta)
    return np.column_stack((x, y, z))


def run_solver_on_trajectory(solver, q0, waypoints, solver_name):
    """
    Recorre una lista de waypoints con el solver dado.
    Retorna diccionario con toda la historia del efector.
    """
    q = q0.copy()
    all_q = [q]
    all_p = [solver.arm.forward_kinematics(q)[0]]
    total_steps = 0
    success = True

    for idx, point in enumerate(waypoints):
        # Cada punto se persigue hasta acercarse o agotar pasos
        res = solver.run_closed_loop(
            q, point,
            max_steps=SIM_CONFIG['max_iter'],
            tol=SIM_CONFIG['tolerance'],
            stall_threshold=SIM_CONFIG['stall_threshold'],
            stall_steps=SIM_CONFIG['stall_steps']
        )
        # Agregar toda la historia (omitimos el primer elemento de res['q'] porque ya lo tenemos)
        q_history = res['q'][1:]  # omitir el inicial que es q
        p_history = [solver.arm.forward_kinematics(qi)[0] for qi in q_history]
        all_q.extend(q_history)
        all_p.extend(p_history)
        q = q_history[-1]
        total_steps += res['steps']
        if not res['success']:
            success = False
            print(f"{solver_name}: no alcanzó el waypoint {idx} (error final={res['error'][-1]:.4f})")
        else:
            print(f"{solver_name}: waypoint {idx} alcanzado en {res['steps']} pasos.")

    return {
        'q': np.array(all_q),
        'p': np.array(all_p),
        'total_steps': total_steps,
        'success_all': success
    }


def main():
    # Configuración
    arm = ThreeDOFArm(**ARM_CONFIG)
    q0 = np.array([0.2, 0.5, 0.1])

    # Trayectoria deseada: círculo en XZ desplazado
    center = np.array([0.9, 0.15, 0.3])
    radius = 0.3
    num_waypoints = 100
    waypoints = generate_trajectory(center, radius, num_waypoints)

    # --- Vista previa de la trayectoria deseada ---
    print("Mostrando trayectoria deseada (orbita con el ratón, cierra para continuar)...")
    viz_prev = ArmVisualizer(arm, figsize=(10, 8))
    viz_prev.plot_arm(q0, show_target=None)
    viz_prev.plot_desired_path(waypoints, color='magenta', label='Deseada')
    plt.show(block=True)  # <-- bloquea hasta cerrar, pero permite interacción
    print("Previsualización cerrada. Iniciando experimentos...\n")

    # Solvers
    solvers = {
        'MPC': CentralizedMPC(
            arm,
            horizon=SIM_CONFIG['horizon'],
            dt=SIM_CONFIG['dt'],
            Q=MPC_CONFIG['Q'],
            R=MPC_CONFIG['R'],
            Q_term=MPC_CONFIG['Q_term'],
            w_int=MPC_CONFIG['w_int'],
            v_max=MPC_CONFIG['v_max']
        ),
        #'Juego': CooperativeGameSolver(arm, alpha=GAME_CONFIG['alpha'], beta=GAME_CONFIG['beta'],
                                       #max_iters=GAME_CONFIG['max_game_iters'], tol=GAME_CONFIG['game_tol'],
                                       #dt=SIM_CONFIG['dt']),
        #'Híbrido': DistributedMPC(arm, horizon=HYBRID_CONFIG['horizon'], dt=SIM_CONFIG['dt'],
                                  #Q_pos=HYBRID_CONFIG['Q_pos'], r_vel=HYBRID_CONFIG['r_vel'],
                                  #w_acc=HYBRID_CONFIG['w_acc'], w_int=HYBRID_CONFIG['w_int'],
                                  #Q_term=HYBRID_CONFIG['Q_term'], game_iters=HYBRID_CONFIG['game_iters'],
                                  #v_max=HYBRID_CONFIG['v_max'])
    }

    results = {}
    for name, solver in solvers.items():
        print(f"\n--- Probando {name} ---")
        results[name] = run_solver_on_trajectory(solver, q0, waypoints, name)

    # Graficar error de posición respecto a la trayectoria deseada
    plt.figure(figsize=(10,6))
    for name, res in results.items():
        # Calcular distancia mínima a la trayectoria deseada (aproximada como distancia al waypoint más cercano en cada paso)
        # Simplificación: comparar con waypoints linealmente interpolados en el tiempo.
        # Para ser exactos, deberíamos calcular distancia punto a curva, pero asumiremos que la referencia es constante entre waypoints.
        errors = []
        t = np.linspace(0, 1, len(res['p']))
        # Interpolar waypoints en el tiempo
        desired_at_t = np.array([waypoints[int(min(i, num_waypoints-1))] for i in np.linspace(0, num_waypoints-1, len(res['p']))])
        for pp, dd in zip(res['p'], desired_at_t):
            errors.append(np.linalg.norm(pp - dd))
        plt.plot(errors, label=name)
    plt.xlabel('Paso')
    plt.ylabel('Error [m]')
    plt.title('Error de seguimiento')
    plt.legend()
    plt.grid(True)
    plt.savefig('tracking_error_comparison.png')
    plt.show()

    # Generar GIFs con trayectoria deseada
    for name, res in results.items():
        viz = ArmVisualizer(arm, figsize=(10,8))
        #viz.plot_arm(res['q'][0], show_target=waypoints[0])
        viz.animate_trajectory(
            [q for q in res['q']],
            target=None,                # no mostramos punto fijo, ya que la trayectoria es la deseada
            save_path=f"tracking_{name}.gif",
            desired_path=waypoints      # <-- aquí pasas la trayectoria deseada
        )
        viz.close()

    # Resumen numérico
    print("\nResumen:")
    for name, res in results.items():
        err_final = np.linalg.norm(res['p'][-1] - waypoints[-1])
        print(f"{name}: pasos totales={res['total_steps']}, éxito={res['success_all']}, error final={err_final:.4f}")


if __name__ == "__main__":
    main()