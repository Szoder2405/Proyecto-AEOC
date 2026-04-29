"""
Control Predictivo Basado en Modelo (MPC) centralizado mejorado.
Incluye costo terminal e integral para convergencia rápida.
"""

import casadi as ca
import numpy as np
from typing import Tuple, Optional
from arm_model.three_dof_arm import ThreeDOFArm


class CentralizedMPC:
    def __init__(self, arm: ThreeDOFArm, horizon: int = 10, dt: float = 0.05,
                 Q: np.ndarray = None, R: np.ndarray = None,
                 Q_term: float = 0.0,           # nuevo: peso terminal
                 w_int: float = 0.0,            # nuevo: peso integral
                 q_min: np.ndarray = None, q_max: np.ndarray = None,
                 v_max: np.ndarray = None):
        self.arm = arm
        self.N = horizon
        self.dt = dt

        self.Q = Q if Q is not None else np.eye(3)
        self.R = R if R is not None else 0.1 * np.eye(3)
        self.Q_term = Q_term
        self.w_int = w_int

        self.q_min = q_min if q_min is not None else arm.q_min
        self.q_max = q_max if q_max is not None else arm.q_max
        if v_max is not None:
            self.v_max = v_max
            self.v_min = -v_max
        else:
            self.v_max = np.array([1.0, 1.0, 1.0])
            self.v_min = -self.v_max

        self._build_solver()

    def _forward_kinematics_casadi(self, q: ca.SX) -> ca.SX:
        """Cinemática directa simbólica (misma que antes)."""
        l1, l2, l3 = self.arm.l1, self.arm.l2, self.arm.l3
        q1, q2, q3 = q[0], q[1], q[2]

        def dh_transform(theta, d, a, alpha):
            ct = ca.cos(theta)
            st = ca.sin(theta)
            ca_ = ca.cos(alpha)
            sa_ = ca.sin(alpha)
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

        T01 = dh_transform(q1, 0, 0, np.pi/2)
        T12 = dh_transform(q2, 0, l1, 0)
        T23 = dh_transform(q3, 0, l2, 0)
        T34 = dh_transform(0, 0, l3, 0)

        T04 = T01 @ T12 @ T23 @ T34
        return T04[:3, 3]

    def _build_solver(self):
        q = ca.SX.sym('q', 3, self.N + 1)
        dq = ca.SX.sym('dq', 3, self.N)

        q0 = ca.SX.sym('q0', 3)
        p_ref = ca.SX.sym('p_ref', 3)

        cost = 0
        g = []
        lbg = []
        ubg = []

        g.append(q[:, 0] - q0)
        lbg.extend([0, 0, 0])
        ubg.extend([0, 0, 0])

        vars_opt = [q[:, 0]]
        for k in range(self.N):
            vars_opt.append(dq[:, k])
            vars_opt.append(q[:, k + 1])

        lbx = []
        ubx = []
        lbx.extend([-ca.inf] * 3)
        ubx.extend([ca.inf] * 3)

        integral = 0  # acumulador del error integral

        for k in range(self.N):
            lbx.extend(self.v_min.tolist())
            ubx.extend(self.v_max.tolist())
            lbx.extend(self.q_min.tolist())
            ubx.extend(self.q_max.tolist())

            g.append(q[:, k + 1] - (q[:, k] + dq[:, k] * self.dt))
            lbg.extend([0, 0, 0])
            ubg.extend([0, 0, 0])

            p_k = self._forward_kinematics_casadi(q[:, k + 1])
            err_k = p_k - p_ref

            # Costo proporcional
            cost += ca.mtimes([err_k.T, self.Q, err_k])
            # Esfuerzo de control
            cost += ca.mtimes([dq[:, k].T, self.R, dq[:, k]])

            # Término integral
            integral = integral + err_k * self.dt
            cost += self.w_int * ca.mtimes([integral.T, integral])

        # Costo terminal
        p_term = self._forward_kinematics_casadi(q[:, self.N])
        cost += self.Q_term * ca.mtimes([(p_term - p_ref).T, (p_term - p_ref)])

        X = ca.vertcat(*vars_opt)
        G = ca.vertcat(*g)

        nlp = {'x': X, 'f': cost, 'g': G, 'p': ca.vertcat(q0, p_ref)}
        opts = {'ipopt.print_level': 0, 'print_time': 0,
                'ipopt.sb': 'yes', 'ipopt.max_iter': 100}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        self.lbx = lbx
        self.ubx = ubx
        self.lbg = lbg
        self.ubg = ubg
        self.n_vars = X.size1()
        self.n_g = G.size1()

    def solve(self, q_current: np.ndarray, p_target: np.ndarray) -> Tuple[np.ndarray, dict]:
        p = ca.vertcat(q_current, p_target)

        x0 = np.zeros(self.n_vars)
        x0[0:3] = q_current
        idx = 3
        for _ in range(self.N):
            x0[idx:idx+3] = 0.0
            idx += 3
            x0[idx:idx+3] = q_current
            idx += 3

        sol = self.solver(x0=x0, p=p, lbx=self.lbx, ubx=self.ubx)
        x_opt = sol['x'].full().flatten()

        q_traj = [q_current]
        dq_opt = None
        idx = 3
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

            max_delta_q = np.max(np.abs(q_new - q))
            q = q_new
            q_history.append(q)
            dq_history.append(dq)
            p = self.arm.forward_kinematics(q)[0]
            p_history.append(p)
            error = np.linalg.norm(p - p_target)
            error_history.append(error)
            cost_history.append(info['cost'])

            if error < tol:
                print(f"MPC: Objetivo alcanzado en {step+1} pasos. Error: {error:.5f}")
                break

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