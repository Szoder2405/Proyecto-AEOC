"""
Funciones de graficación para análisis de resultados.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Tuple


def set_plot_style():
    """Configura estilo consistente para todas las gráficas."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.figsize'] = (10, 6)


def plot_position_error(errors: List[np.ndarray], labels: List[str],
                        title: str = "Error de posición vs iteraciones",
                        xlabel: str = "Iteración", ylabel: str = "Error [m]",
                        log_scale: bool = True, save_path: Optional[str] = None):
    """
    Grafica la evolución del error de posición para diferentes métodos.

    Args:
        errors: Lista de arrays con errores por iteración.
        labels: Etiquetas para cada curva.
        title: Título de la gráfica.
        xlabel, ylabel: Etiquetas de ejes.
        log_scale: Si True, usa escala logarítmica en y.
        save_path: Ruta para guardar la figura.
    """
    set_plot_style()
    plt.figure()

    for err, label in zip(errors, labels):
        plt.plot(err, linewidth=2, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if log_scale:
        plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_joint_trajectories(q_history: List[np.ndarray], dt: float,
                            title: str = "Trayectorias articulares",
                            save_path: Optional[str] = None):
    """
    Grafica la evolución de los ángulos articulares en el tiempo.

    Args:
        q_history: Lista de vectores de ángulos articulares.
        dt: Paso de tiempo.
        title: Título de la gráfica.
        save_path: Ruta para guardar la figura.
    """
    set_plot_style()
    q_array = np.array(q_history)
    time = np.arange(len(q_history)) * dt

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    joint_names = ['Hombro', 'Codo', 'Muñeca']
    for i, ax in enumerate(axes):
        ax.plot(time, q_array[:, i], linewidth=2, color=f'C{i}')
        ax.set_ylabel(f'{joint_names[i]} [rad]')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    axes[-1].set_xlabel('Tiempo [s]')
    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_comparison_bar(metrics: Dict[str, List[float]], metric_name: str,
                        title: Optional[str] = None, ylabel: Optional[str] = None,
                        save_path: Optional[str] = None):
    """
    Crea un gráfico de barras comparando métricas entre métodos.

    Args:
        metrics: Diccionario {método: [valores]}.
        metric_name: Nombre de la métrica para el título.
        title: Título personalizado.
        ylabel: Etiqueta del eje y.
        save_path: Ruta para guardar.
    """
    set_plot_style()
    methods = list(metrics.keys())
    means = [np.mean(metrics[m]) for m in methods]
    stds = [np.std(metrics[m]) for m in methods]

    x = np.arange(len(methods))
    width = 0.6

    plt.figure()
    bars = plt.bar(x, means, width, yerr=stds, capsize=5,
                   color=['steelblue', 'darkorange', 'forestgreen'][:len(methods)])

    plt.xticks(x, methods)
    plt.ylabel(ylabel or metric_name)
    plt.title(title or f"Comparación de {metric_name}")
    plt.grid(True, axis='y', alpha=0.3)

    # Añadir valores encima de las barras
    for bar, mean, std in zip(bars, means, stds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std,
                 f'{mean:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_convergence_comparison(histories: Dict[str, List[float]],
                                title: str = "Comparación de convergencia",
                                xlabel: str = "Iteración", ylabel: str = "Costo",
                                save_path: Optional[str] = None):
    """
    Compara curvas de convergencia de diferentes algoritmos.

    Args:
        histories: Diccionario {método: [valores_por_iteración]}.
        title: Título.
        xlabel, ylabel: Etiquetas.
        save_path: Ruta para guardar.
    """
    set_plot_style()
    plt.figure()

    colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson']
    for (method, values), color in zip(histories.items(), colors):
        plt.plot(values, linewidth=2, label=method, color=color)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_workspace_analysis(reachable_points: np.ndarray,
                            title: str = "Análisis del espacio de trabajo",
                            save_path: Optional[str] = None):
    """
    Visualiza el espacio de trabajo alcanzable (nube de puntos).

    Args:
        reachable_points: Array de shape (N, 3) con posiciones alcanzables.
        title: Título.
        save_path: Ruta para guardar.
    """
    fig = plt.figure(figsize=(12, 5))

    # Proyecciones 2D
    ax1 = fig.add_subplot(131)
    ax1.scatter(reachable_points[:, 0], reachable_points[:, 1], s=1, alpha=0.3, c='blue')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_title('Plano XY')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(132)
    ax2.scatter(reachable_points[:, 0], reachable_points[:, 2], s=1, alpha=0.3, c='green')
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Z [m]')
    ax2.set_title('Plano XZ')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(reachable_points[:, 0], reachable_points[:, 1], reachable_points[:, 2],
                s=1, alpha=0.3, c=reachable_points[:, 2], cmap='viridis')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Vista 3D')

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()