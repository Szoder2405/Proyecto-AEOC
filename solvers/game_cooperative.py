"""
Controlador distribuido basado en teoría de juegos cooperativos para el brazo 3-DOF.
Cada articulación es un agente que optimiza su propio costo, pero todas comparten
el objetivo global de posicionar el efector final en el punto deseado.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.optimize import minimize
from arm_model.three_dof_arm import ThreeDOFArm


class CooperativeGameSolver:
    """
    Solver de juego cooperativo para cinemática inversa redundante.
    
    Cada articulación i minimiza:
        J_i(θ_i, θ_{-i}) = α * (θ_i - θ_i^0)^2 + β * ||p(θ) - p_target||^2
    
    donde θ_i^0 es el ángulo inicial de la articulación i.
    """

    def __init__(self, arm: ThreeDOFArm, alpha: float = 1.0, beta: float = 10.0,
                 max_iters: int = 50, tol: float = 1e-5, dt: float = 0.05):
        """
        Args:
            arm: Modelo del brazo.
            alpha: Peso del costo de movimiento individual.
            beta: Peso del error global de posición.
            max_iters: Máximo de iteraciones del juego.
            tol: Tolerancia para convergencia del equilibrio de Nash.
            dt: Paso de tiempo (para simulación continua, no usado en el solver estático).
        """
        self.arm = arm
        self.alpha = alpha
        self.beta = beta
        self.max_iters = max_iters
        self.tol = tol
        self.dt = dt

    def joint_cost(self, theta_i: float, i: int, q: np.ndarray,
                   q0: np.ndarray, p_target: np.ndarray) -> float:
        """
        Calcula el costo para la articulación i dado su ángulo y el vector completo q.
        """
        q_i = q.copy()
        q_i[i] = theta_i
        p = self.arm.forward_kinematics(q_i)[0]
        position_error = np.linalg.norm(p - p_target) ** 2
        movement_cost = (theta_i - q0[i]) ** 2
        return self.alpha * movement_cost + self.beta * position_error

    def best_response(self, i: int, q: np.ndarray, q0: np.ndarray,
                      p_target: np.ndarray) -> float:
        """
        Calcula la mejor respuesta de la articulación i dadas las demás fijas.
        Resuelve un problema de optimización 1D con bounds.
        """
        bounds = [(self.arm.q_min[i], self.arm.q_max[i])]
        # Adivinanza inicial: valor actual
        x0 = np.array([q[i]])

        result = minimize(
            fun=lambda theta: self.joint_cost(theta[0], i, q, q0, p_target),
            x0=x0,
            bounds=bounds,
            method='L-BFGS-B',
            options={'ftol': 1e-8, 'maxiter': 20}
        )
        return result.x[0]

    def solve_game(self, q_start: np.ndarray, p_target: np.ndarray,
                   q_initial_ref: Optional[np.ndarray] = None) -> dict:
        """
        Encuentra el equilibrio de Nash mediante best response dinámico (Gauss-Seidel).

        Args:
            q_start: Configuración inicial para comenzar las iteraciones.
            p_target: Posición deseada del efector final.
            q_initial_ref: Referencia para el costo de movimiento (si None, se usa q_start).

        Returns:
            Diccionario con la configuración final, historial y métricas.
        """
        q = q_start.copy()
        if q_initial_ref is None:
            q0_ref = q_start.copy()
        else:
            q0_ref = q_initial_ref.copy()

        q_history = [q.copy()]
        error_history = [np.linalg.norm(self.arm.forward_kinematics(q)[0] - p_target)]
        cost_history = []

        converged = False
        for iteration in range(self.max_iters):
            q_old = q.copy()
            # Actualizar cada articulación secuencialmente
            for i in range(3):
                q[i] = self.best_response(i, q, q0_ref, p_target)

            q_history.append(q.copy())
            p = self.arm.forward_kinematics(q)[0]
            error = np.linalg.norm(p - p_target)
            error_history.append(error)

            # Calcular costo total (suma de costos individuales)
            total_cost = 0.0
            for i in range(3):
                total_cost += self.joint_cost(q[i], i, q, q0_ref, p_target)
            cost_history.append(total_cost)

            # Verificar convergencia (cambio en ángulos)
            if np.max(np.abs(q - q_old)) < self.tol:
                converged = True
                print(f"Juego: Convergió en {iteration+1} iteraciones. Error final: {error:.5f}")
                break
        else:
            print(f"Juego: Máximo de iteraciones alcanzado ({self.max_iters}). Error final: {error:.5f}")

        return {
            'q': np.array(q_history),
            'error': np.array(error_history),
            'cost': np.array(cost_history),
            'success': error_history[-1] < 0.01,  # criterio arbitrario
            'iterations': iteration + 1 if 'iteration' in locals() else self.max_iters,
            'converged': converged,
            'q_final': q
        }

    def run_closed_loop(self, q0: np.ndarray, p_target: np.ndarray,
                        max_steps: int = 100, tol: float = 1e-3, 
                        stall_threshold: float = 1e-4, stall_steps: int = 5) -> dict:
        """
        Ejecuta el control en lazo cerrado: en cada paso, se resuelve el juego completo
        desde la configuración actual para obtener el siguiente estado.
        (Alternativa: usar solo una iteración del juego por paso, pero aquí resolvemos el juego completo cada vez).

        Esta implementación simula un controlador que en cada instante resuelve el juego
        de forma instantánea (cinemática inversa), y luego da un paso pequeño hacia la solución.
        """
        q = q0.copy()
        q_history = [q.copy()]
        p_history = [self.arm.forward_kinematics(q)[0]]
        error_history = [np.linalg.norm(p_history[-1] - p_target)]

        stall_counter = 0
        step = 0

        for step in range(max_steps):
            # Resolver el juego para obtener la configuración objetivo instantánea
            game_result = self.solve_game(q, p_target, q_initial_ref=q0)
            q_target = game_result['q_final']

            # Moverse un paso hacia la solución (evita cambios bruscos, simula dinámica)
            delta_q = q_target - q
            # Limitar velocidad
            max_delta = self.dt * 1.0  # velocidad máxima 1 rad/s (ajustable)
            delta_q = np.clip(delta_q, -max_delta, max_delta)

            q_new = q + delta_q
            q_new = self.arm.clip_joints(q_new)

            max_delta_q = np.max(np.abs(q_new - q))
            q = q_new

            q_history.append(q.copy())
            p = self.arm.forward_kinematics(q)[0]
            p_history.append(p)
            error = np.linalg.norm(p - p_target)
            error_history.append(error)

            # Verificar éxito
            if error < tol:
                print(f"Juego (lazo cerrado): Objetivo alcanzado en {step+1} pasos. Error: {error:.5f}")
                break

            # Detección de estancamiento
            if max_delta_q < stall_threshold:
                stall_counter += 1
                if stall_counter >= stall_steps:
                    print(f"Juego (lazo cerrado): Movimiento detenido después de {stall_steps} pasos.")
                    break
            else:
                stall_counter = 0

        else:
            print(f"Juego (lazo cerrado): Máximo de pasos alcanzado ({max_steps}). Error final: {error:.5f}")

        return {
            'q': np.array(q_history),
            'p': np.array(p_history),
            'error': np.array(error_history),
            'steps': step + 1,
            'success': error_history[-1] < tol
        }