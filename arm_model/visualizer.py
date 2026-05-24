"""
Visualizador 3D para el brazo robótico de 3 DOF.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional, Tuple, TYPE_CHECKING
import time

if TYPE_CHECKING:
    from .three_dof_arm import ThreeDOFArm

class ArmVisualizer:
    """
    Clase para visualizar el brazo robótico en 3D.
    """

    def __init__(self, arm: 'ThreeDOFArm', figsize: Tuple[float, float] = (8, 8)):
        """
        Inicializa el visualizador.

        Args:
            arm: Instancia del modelo del brazo.
            figsize: Tamaño de la figura en pulgadas.
        """
        self.arm = arm
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self._setup_axes()

        # Elementos gráficos que se actualizarán
        self.lines = []
        self.joint_points = []
        self.target_point = None
        self.trajectory_line = None

    def _setup_axes(self):
        """Configura los límites y etiquetas de los ejes."""
        total_length = self.arm.l1 + self.arm.l2 + self.arm.l3
        limit = total_length * 1.2
        self.ax.set_xlim([-limit, limit])
        self.ax.set_ylim([-limit, limit])
        self.ax.set_zlim([-limit, limit])
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_zlabel('Z [m]')
        self.ax.set_title('Brazo Robótico 3-DOF')
        # Configurar vista inicial
        self.ax.view_init(elev=20, azim=-60)

    def plot_arm(self, q: np.ndarray, color: str = 'blue', alpha: float = 1.0,
                 show_target: Optional[np.ndarray] = None):
        """
        Dibuja el brazo en la configuración dada.

        Args:
            q: Vector de ángulos articulares [theta1, theta2, theta3].
            color: Color de los eslabones.
            alpha: Transparencia.
            show_target: Punto objetivo opcional para mostrar.
        """
        self.ax.clear()
        self._setup_axes()

        positions = self.arm.get_joint_positions(q)
        x = [p[0] for p in positions]
        y = [p[1] for p in positions]
        z = [p[2] for p in positions]

        # Dibujar eslabones
        self.ax.plot(x, y, z, 'o-', color=color, linewidth=4, markersize=8, alpha=alpha,
                     markerfacecolor='red', markeredgecolor='black')

        # Resaltar efector final
        self.ax.scatter(x[-1], y[-1], z[-1], color='green', s=100, label='Efector final')

        # Mostrar punto objetivo si se proporciona
        if show_target is not None:
            self.ax.scatter(show_target[0], show_target[1], show_target[2],
                            color='red', s=100, marker='*', label='Objetivo')

        # Mostrar base
        self.ax.scatter(0, 0, 0, color='black', s=150, marker='s', label='Base')

        self.ax.legend()
        plt.draw()
        plt.pause(0.01)

    def animate_trajectory(self, q_trajectory: List[np.ndarray], target: Optional[np.ndarray] = None,
                        interval: float = 0.05, save_path: str = "animation.gif",
                        desired_path: Optional[np.ndarray] = None):
        """
        Genera un GIF de la trayectoria del brazo mostrando también
        la trayectoria deseada y la real del efector.

        Args:
            q_trajectory: Lista de vectores de ángulos articulares.
            target: Punto objetivo fijo (opcional).
            interval: Tiempo entre frames en segundos (no usado en esta versión).
            save_path: Ruta donde guardar el GIF.
            desired_path: Array (N,3) con la trayectoria deseada (opcional).
        """
        print(f"Iniciando generación de GIF con {len(q_trajectory)} frames...")
        
        # Precalcular todas las posiciones reales del efector final
        real_positions = []
        for q in q_trajectory:
            pos, _ = self.arm.forward_kinematics(q)
            real_positions.append(pos)
        real_positions = np.array(real_positions)
        
        frames = []
        
        for i, q in enumerate(q_trajectory):
            self.ax.clear()
            self._setup_axes()
            
            # Dibujar el brazo
            positions = self.arm.get_joint_positions(q)
            x = [p[0] for p in positions]
            y = [p[1] for p in positions]
            z = [p[2] for p in positions]
            self.ax.plot(x, y, z, 'o-', color='blue', linewidth=4, markersize=8,
                        alpha=0.8, markerfacecolor='red', markeredgecolor='black')
            
            # Efector final
            self.ax.scatter(x[-1], y[-1], z[-1], color='green', s=100, label='Efector final')
            
            # Objetivo fijo (si se dio)
            if target is not None:
                self.ax.scatter(target[0], target[1], target[2],
                                color='red', s=100, marker='*', label='Objetivo')
            
            # Trayectoria deseada (si se proporciona)
            if desired_path is not None:
                self.ax.plot(desired_path[:, 0], desired_path[:, 1], desired_path[:, 2],
                            color='magenta', linewidth=2, linestyle='--', label='Deseada')
            
            # Trayectoria real hasta el frame actual
            if i > 0:  # al menos dos puntos para una línea
                self.ax.plot(real_positions[:i+1, 0], real_positions[:i+1, 1], real_positions[:i+1, 2],
                            color='orange', linewidth=2, linestyle='-', label='Real')
            
            # Base
            self.ax.scatter(0, 0, 0, color='black', s=150, marker='s', label='Base')
            
            self.ax.legend(loc='upper left')
            self.ax.set_title(f'Frame {i+1}/{len(q_trajectory)}')
            
            # Renderizar y capturar
            self.fig.canvas.draw()
            frame = np.array(self.fig.canvas.renderer.buffer_rgba())
            frames.append(frame)
            
            if (i + 1) % 10 == 0:
                print(f"  Procesado frame {i+1}/{len(q_trajectory)}")
        
        # Guardar GIF
        try:
            import imageio
            imageio.mimsave(save_path, frames, fps=10, loop=0)
            print(f"GIF guardado exitosamente en: {save_path}")
        except ImportError:
            print("imageio no está instalado. Instálalo con: pip install imageio")
        
        print(f"Dimensiones del GIF: {frames[0].shape[1]}x{frames[0].shape[0]} píxeles")

    def plot_trajectory_path(self, q_trajectory: List[np.ndarray], color: str = 'cyan',
                             label: str = 'Trayectoria'):
        """
        Dibuja el camino recorrido por el efector final.

        Args:
            q_trajectory: Lista de configuraciones articulares.
            color: Color de la línea.
            label: Etiqueta para la leyenda.
        """
        points = []
        for q in q_trajectory:
            pos, _ = self.arm.forward_kinematics(q)
            points.append(pos)
        points = np.array(points)
        self.ax.plot(points[:, 0], points[:, 1], points[:, 2],
                     color=color, linewidth=2, linestyle='--', label=label)
        self.ax.legend()

    def plot_desired_path(self, desired_points: np.ndarray, color: str = 'magenta',
                      label: str = 'Deseada'):
        """
        Dibuja la trayectoria deseada como una línea discontinua.

        Args:
            desired_points: Array (N, 3) con las coordenadas de la trayectoria.
            color: Color de la línea.
            label: Etiqueta para la leyenda.
        """
        self.ax.plot(desired_points[:, 0], desired_points[:, 1], desired_points[:, 2],
                    color=color, linewidth=2, linestyle='--', label=label)
        self.ax.legend()

    def compare_configurations(self, configs: List[Tuple[np.ndarray, str, str]],
                               target: Optional[np.ndarray] = None):
        """
        Compara múltiples configuraciones en subplots.

        Args:
            configs: Lista de tuplas (q, título, color).
            target: Punto objetivo opcional.
        """
        n = len(configs)
        fig, axes = plt.subplots(1, n, subplot_kw={'projection': '3d'}, figsize=(5*n, 5))
        if n == 1:
            axes = [axes]

        total_length = self.arm.l1 + self.arm.l2 + self.arm.l3
        limit = total_length * 1.2

        for ax, (q, title, color) in zip(axes, configs):
            positions = self.arm.get_joint_positions(q)
            x = [p[0] for p in positions]
            y = [p[1] for p in positions]
            z = [p[2] for p in positions]

            ax.plot(x, y, z, 'o-', color=color, linewidth=4, markersize=6)
            ax.scatter(x[-1], y[-1], z[-1], color='green', s=80)
            if target is not None:
                ax.scatter(target[0], target[1], target[2], color='red', s=80, marker='*')
            ax.scatter(0, 0, 0, color='black', s=100, marker='s')

            ax.set_xlim([-limit, limit])
            ax.set_ylim([-limit, limit])
            ax.set_zlim([-limit, limit])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(title)
            ax.view_init(elev=20, azim=-60)

        plt.tight_layout()
        plt.show()

    def show(self):
        """Muestra la figura actual sin bloquear."""
        plt.show(block=False)

    def close(self):
        """Cierra la figura."""
        plt.close(self.fig)