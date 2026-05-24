"""
Experimento comparativo: seguimiento de trayectoria con los tres solvers.
Se genera un MP4 por solver mostrando trayectoria deseada vs real.

"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import io
import cv2
from pathlib import Path

# Configuración: modo no interactivo para los frames del video
matplotlib.use('Agg')

# --- CONFIGURACIÓN CRÍTICA DE DLLS ABSOLUTAS PARA WINDOWS ---
os.environ['OMP_CANCELLATION'] = 'TRUE'

if sys.platform == 'win32':
    # 1. Obtener la ruta raíz del proyecto de forma totalmente absoluta
    base_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 2. Construir rutas absolutas explícitas de CasADi en el entorno virtual
    casadi_absolute_path = os.path.abspath(os.path.join(base_project_dir, 'venv', 'Lib', 'site-packages', 'casadi'))
    
    # 3. Forzar el registro en el kernel de Windows usando rutas validadas
    if os.path.exists(casadi_absolute_path):
        os.add_dll_directory(casadi_absolute_path)
        
    # 4. Intentar también con la carpeta ipopt_bin por si acaso
    ipopt_bin_absolute = os.path.abspath(os.path.join(base_project_dir, 'ipopt_bin'))
    if os.path.exists(ipopt_bin_absolute):
        os.add_dll_directory(ipopt_bin_absolute)
        os.environ['PATH'] = ipopt_bin_absolute + os.pathsep + os.environ.get('PATH', '')

# -------------------------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arm_model.three_dof_arm import ThreeDOFArm
from arm_model.visualizer import ArmVisualizer
from solvers.mpc_centralized import CentralizedMPC
from solvers.game_cooperative import CooperativeGameSolver
from solvers.hybrid import DistributedMPC
from utils.config import (ARM_CONFIG, SIM_CONFIG, MPC_CONFIG,
                          GAME_CONFIG, HYBRID_CONFIG, VIZ_CONFIG)


# -----------------------------------------------------------------------------
# Generación de trayectoria
# -----------------------------------------------------------------------------
def generate_trajectory(shape, num_points: int = 200):
    """
    Genera trayectorias 3D elaboradas en el plano YZ del brazo.
    """
    theta = np.linspace(0, 2 * np.pi, num_points)
    cx, cy, cz = 0.75, 0.0, 0.40

    if shape == 'flor':
        r = 0.2 * (0.5 + 0.5 * np.cos(5 * theta))
        x = cx * np.ones(num_points)
        y = cy + r * np.cos(theta)
        z = cz + r * np.sin(theta)

    elif shape == 'lissajous':
        x = cx * np.ones(num_points)
        y = cy + 0.27 * np.sin(3 * theta)
        z = cz + 0.2 * np.sin(2 * theta + np.pi / 4)

    elif shape == 'estrella':
        R_out = 0.20
        R_in  = 0.09
        n_puntas = 5
        angulos_puntas = np.linspace(0, 2*np.pi, n_puntas, endpoint=False)
        angulos_valles = angulos_puntas + np.pi / n_puntas
        angulos = []
        radios = []
        for i in range(n_puntas):
            angulos.append(angulos_puntas[i])
            radios.append(R_out)
            angulos.append(angulos_valles[i])
            radios.append(R_in)
        angulos.append(angulos[0])
        radios.append(radios[0])
        cy, cz = 0.0, 0.40
        y_vert = cy + np.array(radios) * np.cos(angulos)
        z_vert = cz + np.array(radios) * np.sin(angulos)
        diffs = np.diff(np.column_stack((y_vert, z_vert)), axis=0)
        seg_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        cum_dist = np.concatenate(([0], np.cumsum(seg_lengths)))
        total_dist = cum_dist[-1]
        t_new = np.linspace(0, total_dist, num_points)
        y_interp = np.interp(t_new, cum_dist, y_vert)
        z_interp = np.interp(t_new, cum_dist, z_vert)
        x = cx * np.ones(num_points)
        y = y_interp
        z = z_interp

    elif shape == 'trebol':
        r = 0.15 * np.abs(np.cos(2 * theta))
        x = cx * np.ones(num_points)
        y = cy + r * np.cos(theta)
        z = cz + r * np.sin(theta)

    else:  # 'circulo'
        x = cx * np.ones(num_points)
        y = cy + 0.28 * np.cos(theta)
        z = cz + 0.28 * np.sin(theta)

    return np.column_stack((x, y, z))


# -----------------------------------------------------------------------------
# Ejecución del solver sobre la trayectoria completa
# -----------------------------------------------------------------------------
def run_solver_on_trajectory(solver, q0, waypoints, solver_name):
    """Recorre una lista de waypoints con el solver dado."""
    q = q0.copy()
    all_q = [q]
    all_p = [solver.arm.forward_kinematics(q)[0]]
    total_steps = 0
    success = True

    for idx, point in enumerate(waypoints):
        res = solver.run_closed_loop(
            q, point,
            max_steps=SIM_CONFIG['max_iter'],
            tol=SIM_CONFIG['tolerance'],
            stall_threshold=SIM_CONFIG['stall_threshold'],
            stall_steps=SIM_CONFIG['stall_steps']
        )
        q_history = res['q'][1:]
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


# -----------------------------------------------------------------------------
# Exportación de video MP4 (con título dinámico)
# -----------------------------------------------------------------------------
def _render_frame(arm, q, desired_path, actual_path_so_far, step, total,
                  figsize, dpi, solver_name, shape_name):
    """Dibuja un frame matplotlib con título que incluye solver y figura."""
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(desired_path[:, 0], desired_path[:, 1], desired_path[:, 2],
            'b--', lw=1.0, alpha=0.5, label='Deseada')

    if len(actual_path_so_far) > 1:
        ap = np.array(actual_path_so_far)
        ax.plot(ap[:, 0], ap[:, 1], ap[:, 2], 'r-', lw=1.5, alpha=0.85, label='Real')
        ax.scatter(*ap[-1], color='red', s=25, zorder=5)

    _, joints = arm.forward_kinematics(q)
    joints = np.array(joints)
    jx, jy, jz = joints[:, 0], joints[:, 1], joints[:, 2]
    ax.plot(jx, jy, jz, 'k-o', lw=2.5, markersize=5, label='Brazo')
    ax.scatter(*joints[-1], color='green', s=60, zorder=6, label='EE')

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title(f'{solver_name} - Figura: {shape_name.capitalize()}   |   Paso {step+1}/{total}', fontsize=10)
    lim = arm.l1 + arm.l2 + arm.l3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0, lim)
    ax.legend(loc='upper left', fontsize=7)
    ax.view_init(elev=25, azim=45)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return bgr


def save_mp4(arm, q_history, p_history, desired_path, save_path,
             solver_name, shape_name,
             fps=20, skip_frames=3, figsize=(10, 8), dpi=100):
    """Genera un video MP4 con título específico para cada combinación solver+figura."""
    indices = list(range(0, len(q_history), skip_frames))
    total = len(indices)
    print(f"[MP4] Generando {total} frames para {solver_name} ({shape_name}) -> '{save_path}' (fps={fps})")

    first = _render_frame(arm, q_history[0], desired_path,
                          p_history[:1], 0, len(q_history)-1,
                          figsize, dpi, solver_name, shape_name)
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    writer.write(first)

    for i, idx in enumerate(indices[1:], start=1):
        if i % 50 == 0:
            print(f"  frame {i}/{total}")
        frame = _render_frame(arm, q_history[idx], desired_path,
                              p_history[:idx+1], idx, len(q_history)-1,
                              figsize, dpi, solver_name, shape_name)
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)

    writer.release()
    print(f"[MP4] Guardado: {save_path}")


# -----------------------------------------------------------------------------
# Función para guardar gráfica de error (sin mostrarla)
# -----------------------------------------------------------------------------
def save_error_plot(results, waypoints, num_waypoints, shape_name, save_dir):
    """Guarda la gráfica de error de seguimiento para una figura."""
    plt.figure(figsize=(10, 6))
    for name, res in results.items():
        desired_at_t = np.array([
            waypoints[int(min(i, num_waypoints - 1))]
            for i in np.linspace(0, num_waypoints - 1, len(res['p']))
        ])
        errors = [np.linalg.norm(pp - dd) for pp, dd in zip(res['p'], desired_at_t)]
        plt.plot(errors, label=name)
    plt.xlabel('Paso')
    plt.ylabel('Error [m]')
    plt.title(f'Error de seguimiento - {shape_name.capitalize()}')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(save_dir, f"error_{shape_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grafica de error guardada: {save_path}")


# -----------------------------------------------------------------------------
# Interacción con el usuario y ejecución principal
# -----------------------------------------------------------------------------
def main():
    # Configuración inicial
    arm = ThreeDOFArm(**ARM_CONFIG)
    q0 = np.array([0.2, 0.5, 0.1])

    # Menú de figuras
    print("\n" + "="*60)
    print(" EXPERIMENTO DE SEGUIMIENTO DE TRAYECTORIA ")
    print("="*60)
    print("El brazo robotico de 3 grados de libertad realizara una figura")
    print("elegida por ti, utilizando tres controladores diferentes:")
    print("  - MPC centralizado")
    print("  - Juego cooperativo")
    print("  - Control hibrido distribuido")
    print("\nAl final se generaran tres videos MP4 (uno por controlador)")
    print("y una grafica comparativa del error de seguimiento.\n")

    print("Figuras disponibles:")
    print("   1. flor (5 petalos)")
    print("   2. lissajous (figura de Lissajous 3:2)")
    print("   3. estrella (estrella de 5 puntas)")
    print("   4. trebol (trebol de 4 hojas)")
    print("   5. circulo (circulo simple)")

    opciones_validas = {'1': 'flor', '2': 'lissajous', '3': 'estrella',
                        '4': 'trebol', '5': 'circulo'}

    while True:
        seleccion = input("\n¿Que figura quieres que realice el brazo? (1-5): ").strip()
        if seleccion in opciones_validas:
            shape = opciones_validas[seleccion]
            break
        else:
            print("Opcion no valida. Por favor elige un numero entre 1 y 5.")

    print(f"\n[OK] Figura seleccionada: {shape}")
    print("A continuacion se mostrara la trayectoria deseada. Cierra la ventana para continuar.\n")

    # Generar trayectoria
    num_waypoints = 200
    waypoints = generate_trajectory(shape=shape, num_points=num_waypoints)

    # ---- Previsualización de la trayectoria (con ventana interactiva) ----
    # Cambiamos temporalmente el backend a uno interactivo para mostrar la figura
    backend_actual = matplotlib.get_backend()
    matplotlib.use('TkAgg')  # o 'Qt5Agg' si lo prefieres
    viz_prev = ArmVisualizer(arm, figsize=(10, 8))
    viz_prev.plot_arm(q0, show_target=None)
    viz_prev.plot_desired_path(waypoints, color='magenta', label='Deseada')
    plt.title(f"Trayectoria deseada: {shape.capitalize()}")
    plt.show(block=True)
    # Restaurar el backend a 'Agg' para la generación de videos
    matplotlib.use(backend_actual)
    print("Previsualizacion cerrada. Iniciando experimentos...\n")

    # Crear solvers (se instancian nuevos para esta figura)
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
        'Juego': CooperativeGameSolver(arm, alpha=GAME_CONFIG['alpha'], beta=GAME_CONFIG['beta'],
                                       max_iters=GAME_CONFIG['max_game_iters'], tol=GAME_CONFIG['game_tol'],
                                       dt=SIM_CONFIG['dt']),
        'Híbrido': DistributedMPC(
            arm,
            horizon    = HYBRID_CONFIG['horizon'],
            dt         = HYBRID_CONFIG['dt'],
            Q_pos      = HYBRID_CONFIG['Q_pos'],
            r_vel      = HYBRID_CONFIG['r_vel'],
            w_acc      = HYBRID_CONFIG['w_acc'],
            w_int      = HYBRID_CONFIG['w_int'],
            Q_term     = HYBRID_CONFIG['Q_term'],
            game_iters = HYBRID_CONFIG['game_iters'],
            nash_tol   = HYBRID_CONFIG['nash_tol'],
            v_max      = HYBRID_CONFIG['v_max']
        ),
    }

    # Directorios de salida
    videos_dir = Path.home() / "Downloads" / "videos_brazo"
    errors_dir = Path.home() / "Downloads" / "error_seguimiento"
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(errors_dir, exist_ok=True)

    # Configuración de video desde VIZ_CONFIG
    fps = VIZ_CONFIG.get('video_fps', 20)
    skip_frames = VIZ_CONFIG.get('video_skip_frames', 3)
    figsize = VIZ_CONFIG.get('figsize', (10, 8))
    dpi = VIZ_CONFIG.get('dpi', 100)

    # Ejecutar los solvers
    results = {}
    for name, solver in solvers.items():
        print(f"\n--- Probando {name} con figura {shape} ---")
        results[name] = run_solver_on_trajectory(solver, q0, waypoints, name)

    # Guardar gráfica de error
    save_error_plot(results, waypoints, num_waypoints, shape, errors_dir)

    # Generar videos MP4
    for name, res in results.items():
        save_path = videos_dir / f"tracking_{name}_{shape}.mp4"
        print(f"Generando video para {name} (figura: {shape})...")
        save_mp4(
            arm,
            q_history    = res['q'],
            p_history    = res['p'],
            desired_path = waypoints,
            save_path    = str(save_path),
            solver_name  = name,
            shape_name   = shape,
            fps          = fps,
            skip_frames  = skip_frames,
            figsize      = figsize,
            dpi          = dpi
        )

    # Resumen numérico
    print(f"\nResumen para {shape}:")
    for name, res in results.items():
        err_final = np.linalg.norm(res['p'][-1] - waypoints[-1])
        print(f"  {name}: pasos={res['total_steps']}, exito={res['success_all']}, error final={err_final:.4f}")

    print("\n" + "="*60)
    print("PROCESO COMPLETADO.")
    print(f"Videos guardados en: {videos_dir}")
    print(f"Grafica de error guardada en: {errors_dir}")
    print("="*60)


if __name__ == "__main__":
    main()