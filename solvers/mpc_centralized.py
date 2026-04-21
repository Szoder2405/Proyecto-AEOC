"""
Control Predictivo Basado en Modelo (MPC) centralizado para el brazo 3-DOF.
Utiliza CasADi para formulación y optimización no lineal.
"""

import casadi as ca
import numpy as np
from typing import Tuple, List, Optional
from arm_model.three_dof_arm import ThreeDOFArm


class CentralizedMPC:
    """
    MPC centralizado que resuelve el problema de control óptimo en cada paso.
    
    Minimiza:
        J = Σ (p - p_ref)ᵀ Q (p - p_ref) + Δqᵀ R Δq
    
    Sujeto a:
        q_{k+1} = q_k + Δq_k * dt
        p_k = FK(q_k)
        q_min ≤ q_k ≤ q_max
        Δq_min ≤ Δq_k ≤ Δq_max
    """

    def __init__(self, arm: ThreeDOFArm, horizon: int = 10, dt: float = 0.05,
                 Q: np.ndarray = None, R: np.ndarray = None,
                 q_min: np.ndarray = None, q_max: np.ndarray = None,
                 v_max: np.ndarray = None):
        """
        Inicializa el MPC.

        Args:
            arm: Modelo del brazo robótico.
            horizon: Horizonte de predicción (N).
            dt: Paso de tiempo [s].
            Q: Matriz de peso 3x3 para error de posición (default: identidad).
            R: Matriz de peso 3x3 para cambios de ángulo (default: 0.1*I).
            q_min, q_max: Límites articulares (si None, se usan los del brazo).
            v_max: Velocidad máxima articular [rad/s] (simétrica).
        """
        self.arm = arm
        self.N = horizon
        self.dt = dt

        # Pesos
        self.Q = Q if Q is not None else np.eye(3)
        self.R = R if R is not None else 0.1 * np.eye(3)

        # Límites articulares
        self.q_min = q_min if q_min is not None else arm.q_min
        self.q_max = q_max if q_max is not None else arm.q_max

        # Límites de velocidad
        if v_max is not None:
            self.v_max = v_max
            self.v_min = -v_max
        else:
            self.v_max = np.array([1.0, 1.0, 1.0])
            self.v_min = -self.v_max

        # Construir el solver de optimización con CasADi
        self._build_solver()

    def _forward_kinematics_casadi(self, q: ca.SX) -> ca.SX:
        """
        Cinemática directa simbólica usando transformaciones DH.
        Coincide exactamente con ThreeDOFArm.forward_kinematics.
        """
        l1, l2, l3 = self.arm.l1, self.arm.l2, self.arm.l3
        q1, q2, q3 = q[0], q[1], q[2]

        # Función para matriz DH simbólica
        def dh_transform(theta, d, a, alpha):
            ct = ca.cos(theta)
            st = ca.sin(theta)
            ca_ = ca.cos(alpha)
            sa_ = ca.sin(alpha)
            # Matriz 4x4
            T = ca.SX.zeros(4, 4)
            T[0, 0] = ct
            T[0, 1] = -st * ca_
            T[0, 2] = st * sa_
            T[0, 3] = a * ct
            T[1, 0] = st
            T[1, 1] = ct * ca_
            T[1, 2] = -ct * sa_
            T[1, 3] = a * st
            T[2, 0] = 0
            T[2, 1] = sa_
            T[2, 2] = ca_
            T[2, 3] = d
            T[3, 0] = 0
            T[3, 1] = 0
            T[3, 2] = 0
            T[3, 3] = 1
            return T

        # Transformaciones DH (mismos parámetros que en ThreeDOFArm)
        T01 = dh_transform(q1, 0, 0, np.pi/2)
        T12 = dh_transform(q2, 0, l1, 0)
        T23 = dh_transform(q3, 0, l2, 0)
        T34 = dh_transform(0, 0, l3, 0)

        T04 = T01 @ T12 @ T23 @ T34
        p = T04[:3, 3]  # extraer posición (x,y,z)

        return p

    def _build_solver(self):
        """
        Construye el problema de optimización MPC usando CasADi.
        """
        # Variables de decisión: secuencia de ángulos q_k para k=0..N (cada uno 3x1)
        # y acciones de control Δq_k para k=0..N-1
        q = ca.SX.sym('q', 3, self.N + 1)
        dq = ca.SX.sym('dq', 3, self.N)

        # Parámetros: estado inicial q0 y referencia p_ref
        q0 = ca.SX.sym('q0', 3)
        p_ref = ca.SX.sym('p_ref', 3)

        # Función de costo y restricciones
        cost = 0
        g = []      # restricciones de igualdad
        lbg = []
        ubg = []

        # Restricción inicial: q0 debe ser igual al parámetro q0
        g.append(q[:, 0] - q0)
        lbg.extend([0, 0, 0])
        ubg.extend([0, 0, 0])

        # Variables de decisión empaquetadas en orden:
        # q0, dq0, q1, dq1, q2, ..., dq_{N-1}, q_N
        vars_opt = [q[:, 0]]
        for k in range(self.N):
            vars_opt.append(dq[:, k])
            vars_opt.append(q[:, k + 1])

        # Límites de variables
        lbx = []
        ubx = []

        # q0 (se fija mediante restricción, límites libres)
        lbx.extend([-ca.inf] * 3)
        ubx.extend([ca.inf] * 3)

        for k in range(self.N):
            # Límites para Δq_k
            lbx.extend(self.v_min.tolist())
            ubx.extend(self.v_max.tolist())
            # Límites para q_{k+1}
            lbx.extend(self.q_min.tolist())
            ubx.extend(self.q_max.tolist())

            # Dinámica: q_{k+1} = q_k + dq_k * dt
            g.append(q[:, k + 1] - (q[:, k] + dq[:, k] * self.dt))
            lbg.extend([0, 0, 0])
            ubg.extend([0, 0, 0])

            # Posición del efector final en el paso k+1
            p_k = self._forward_kinematics_casadi(q[:, k + 1])

            # Costo acumulado
            cost += ca.mtimes([(1*p_k - 1*p_ref).T, self.Q, (1*p_k - 1*p_ref)]) \
                    + ca.mtimes([dq[:, k].T, self.R, dq[:, k]])

        # Empaquetar todo
        X = ca.vertcat(*vars_opt)
        G = ca.vertcat(*g)

        # Crear el problema NLP
        nlp = {
            'x': X,
            'f': cost,
            'g': G,
            'p': ca.vertcat(q0, p_ref)
        }

        # Opciones del solver

        opts = {
            'ipopt.print_level': 0,  #5      # mostrar iteraciones
            'print_time': 0,   #1
            'ipopt.sb': 'yes',
            'ipopt.max_iter': 100
        }

        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # Guardar límites para uso posterior
        self.lbx = lbx
        self.ubx = ubx
        self.lbg = lbg
        self.ubg = ubg

        # Guardar dimensiones para desempaquetar
        self.n_vars = X.size1()
        self.n_g = G.size1()

    def solve(self, q_current: np.ndarray, p_target: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Resuelve el problema MPC para un estado actual y objetivo dados.

        Returns:
            dq_opt: primera acción de control [3].
            info: diccionario con detalles de la solución.
        """
        # Parámetros
        p = ca.vertcat(q_current, p_target)

        # Solución inicial (adivinanza simple)
        x0 = np.zeros(self.n_vars)
        # Rellenar q0
        x0[0:3] = q_current
        idx = 3
        for _ in range(self.N):
            # Δq = 0
            x0[idx:idx+3] = 0.0
            idx += 3
            # q siguiente = q_current (estimación)
            x0[idx:idx+3] = q_current
            idx += 3

        # Resolver (los límites de restricciones ya están fijos en el solver, no hace falta pasarlos)
        sol = self.solver(x0=x0, p=p, lbx=self.lbx, ubx=self.ubx)
        x_opt = sol['x'].full().flatten()

        # Desempaquetar: el orden es q0, dq0, q1, dq1, q2, ..., qN
        q_traj = [q_current]
        dq_opt = None
        idx = 3  # saltar q0
        for k in range(self.N):
            dq_k = x_opt[idx:idx+3]
            idx += 3
            q_next = x_opt[idx:idx+3]
            idx += 3
            if k == 0:
                dq_opt = dq_k
            q_traj.append(q_next)

        p_final = self.arm.forward_kinematics(q_traj[-1])[0]

        info = {
            'cost': sol['f'].full().flatten()[0],
            'q_trajectory': q_traj,
            'p_final': p_final,
            'solver_status': self.solver.stats()['return_status']
        }

        return dq_opt, info

    def run_closed_loop(self, q0: np.ndarray, p_target: np.ndarray,
                        max_steps: int = 100, tol: float = 1e-3,
                        stall_threshold: float = 1e-4, stall_steps: int = 5) -> dict:
        """
        Ejecuta el MPC en lazo cerrado hasta alcanzar el objetivo o detectar estancamiento.

        Args:
            q0: Configuración inicial [rad].
            p_target: Posición objetivo [m].
            max_steps: Número máximo de pasos de control.
            tol: Tolerancia de error de posición para considerar éxito.
            stall_threshold: Cambio mínimo en ángulos articulares para considerar movimiento.
            stall_steps: Número de pasos consecutivos sin movimiento significativo para detenerse.

        Returns:
            Diccionario con historial de estados, acciones, errores, etc.
        """
        q = q0.copy()
        q_history = [q]
        dq_history = []
        p_history = [self.arm.forward_kinematics(q)[0]]
        error_history = [np.linalg.norm(p_history[-1] - p_target)]
        cost_history = []

        stall_counter = 0
        step = 0

        for step in range(max_steps):
            dq, info = self.solve(q, p_target)
            q_new = q + dq * self.dt
            q_new = self.arm.clip_joints(q_new)

            # Calcular cambio máximo en ángulos
            max_delta_q = np.max(np.abs(q_new - q))

            q = q_new
            q_history.append(q)
            dq_history.append(dq)
            p = self.arm.forward_kinematics(q)[0]
            p_history.append(p)
            error = np.linalg.norm(p - p_target)
            error_history.append(error)
            cost_history.append(info['cost'])

            # Verificar éxito por error
            if error < tol:
                print(f"MPC: Objetivo alcanzado en {step+1} pasos. Error: {error:.5f}")
                break

            # Verificar estancamiento
            if max_delta_q < stall_threshold:
                stall_counter += 1
                if stall_counter >= stall_steps:
                    print(f"MPC: Movimiento detenido después de {stall_steps} pasos con cambio < {stall_threshold} rad.")
                    break
            else:
                stall_counter = 0

        else:
            print(f"MPC: Máximo de pasos alcanzado ({max_steps}). Error final: {error_history[-1]:.5f}")

        return {
            'q': np.array(q_history),
            'dq': np.array(dq_history),
            'p': np.array(p_history),
            'error': np.array(error_history),
            'cost': np.array(cost_history),
            'steps': step + 1,
            'success': error_history[-1] < tol
        }