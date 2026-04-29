"""
Distributed Model Predictive Control (DMPC) basado en teoría de juegos para brazo 3-DOF.
Cada articulación planifica sobre un horizonte usando los planes de las otras.
Se itera Gauss-Seidel para alcanzar un equilibrio de Nash del juego dinámico.
"""

import casadi as ca
import numpy as np
from typing import Tuple, Optional
from arm_model.three_dof_arm import ThreeDOFArm


class DistributedMPC:
    """
    DMPC con negociación secuencial de planes articulares.
    Cada articulación i minimiza:
        J_i = Σ_k [ ||p(q_k) - p_ref||_Q^2 + r * dq_i(k)^2 + w * (ddq_i(k))^2
                   + w_int * ||integral_k||^2 ]
              + ||p(q_N) - p_ref||_Qterm^2
    sujeto a su dinámica y límites.
    """

    def __init__(self, arm: ThreeDOFArm,
                 horizon: int = 15,
                 dt: float = 0.05,
                 Q_pos: np.ndarray = None,
                 r_vel: float = 0.1,
                 w_acc: float = 0.01,
                 w_int: float = 0.0,          
                 Q_term: float = 100.0,
                 game_iters: int = 3,
                 v_max: np.ndarray = None,
                 q_min: np.ndarray = None,
                 q_max: np.ndarray = None):
        self.arm = arm
        self.N = horizon
        self.dt = dt
        self.r_vel = r_vel
        self.w_acc = w_acc
        self.w_int = w_int                  # <-- guardar
        self.Q_term = Q_term
        self.max_game_iters = game_iters

        self.Q = Q_pos if Q_pos is not None else 10.0 * np.eye(3)

        self.q_min = q_min if q_min is not None else arm.q_min
        self.q_max = q_max if q_max is not None else arm.q_max
        if v_max is not None:
            self.v_max = v_max
            self.v_min = -v_max
        else:
            self.v_max = np.array([1.0, 1.0, 1.0])
            self.v_min = -self.v_max

        # Construir solvers por articulación
        self._joint_solvers = {}
        for i in range(3):
            self._joint_solvers[i] = self._create_joint_solver(i)

    def _fk_casadi(self, q0, q1, q2):
        """Cinemática directa simbólica dados los tres ángulos."""
        l1, l2, l3 = self.arm.l1, self.arm.l2, self.arm.l3
        c0 = ca.cos(q0); s0 = ca.sin(q0)
        c1 = ca.cos(q1); s1 = ca.sin(q1)
        c12 = ca.cos(q1+q2); s12 = ca.sin(q1+q2)

        x = c0 * (l1*c1 + l2*c12 + l3*c12)
        y = s0 * (l1*c1 + l2*c12 + l3*c12)
        z = l1*s1 + l2*s12 + l3*s12
        return ca.vertcat(x, y, z)

    def _create_joint_solver(self, i: int):
        """Crea solver CasADi para la articulación i."""
        dq_i = ca.SX.sym('dq_i', self.N)
        q0_i = ca.SX.sym('q0_i')
        plan_other1 = ca.SX.sym('plan_o1', self.N+1)
        plan_other2 = ca.SX.sym('plan_o2', self.N+1)
        p_ref = ca.SX.sym('p_ref', 3)

        q_i = q0_i
        cost = 0
        integral = 0   # acumulador del error integral

        for k in range(self.N):
            q_i_next = q_i + dq_i[k] * self.dt
            # Vector q completo
            if i == 0:
                q_vec = ca.vertcat(q_i_next, plan_other1[k+1], plan_other2[k+1])
            elif i == 1:
                q_vec = ca.vertcat(plan_other1[k+1], q_i_next, plan_other2[k+1])
            else:
                q_vec = ca.vertcat(plan_other1[k+1], plan_other2[k+1], q_i_next)

            p_k = self._fk_casadi(q_vec[0], q_vec[1], q_vec[2])
            err_k = p_k - p_ref

            # Costo de etapa
            cost += ca.mtimes([err_k.T, self.Q, err_k])
            cost += self.r_vel * dq_i[k]**2
            if k > 0:
                cost += self.w_acc * ((dq_i[k] - dq_i[k-1])/self.dt)**2

            # Término integral
            integral = integral + err_k * self.dt
            cost += self.w_int * ca.mtimes([integral.T, integral])

            q_i = q_i_next

        # Costo terminal
        if i == 0:
            q_vec_term = ca.vertcat(q_i, plan_other1[self.N], plan_other2[self.N])
        elif i == 1:
            q_vec_term = ca.vertcat(plan_other1[self.N], q_i, plan_other2[self.N])
        else:
            q_vec_term = ca.vertcat(plan_other1[self.N], plan_other2[self.N], q_i)
        p_term = self._fk_casadi(q_vec_term[0], q_vec_term[1], q_vec_term[2])
        cost += self.Q_term * ca.mtimes([(p_term - p_ref).T, (p_term - p_ref)])

        X = dq_i
        lbx = [self.v_min[i]] * self.N
        ubx = [self.v_max[i]] * self.N
        P = ca.vertcat(q0_i, plan_other1, plan_other2, p_ref)

        nlp = {'x': X, 'f': cost, 'p': P}
        opts = {'ipopt.print_level': 0, 'print_time': 0,
                'ipopt.sb': 'yes', 'ipopt.max_iter': 100}
        solver = ca.nlpsol(f'solver_{i}', 'ipopt', nlp, opts)
        return solver, lbx, ubx

    def solve(self, q_current: np.ndarray, p_target: np.ndarray) -> Tuple[np.ndarray, dict]:
        n_joints = 3
        plans = [np.full(self.N+1, q_current[i]) for i in range(n_joints)]

        for _ in range(self.max_game_iters):
            for i in range(n_joints):
                other_plans = [plans[j] for j in range(n_joints) if j != i]
                plan_o1, plan_o2 = other_plans[0], other_plans[1]

                solver_i, lbx_i, ubx_i = self._joint_solvers[i]
                p = ca.vertcat(q_current[i], plan_o1, plan_o2, p_target)

                x0 = np.zeros(self.N)
                sol = solver_i(x0=x0, p=p, lbx=lbx_i, ubx=ubx_i)
                dq_opt = sol['x'].full().flatten()

                q_i = q_current[i]
                new_plan = [q_i]
                for k in range(self.N):
                    q_i += dq_opt[k] * self.dt
                    q_i = np.clip(q_i, self.q_min[i], self.q_max[i])
                    new_plan.append(q_i)
                plans[i] = np.array(new_plan)

        dq_first = np.array([(plans[i][1] - q_current[i]) / self.dt for i in range(n_joints)])
        dq_first = np.clip(dq_first, self.v_min, self.v_max)
        return dq_first, {'plans': plans}

    def run_closed_loop(self, q0, p_target, max_steps=100, tol=1e-3,
                        stall_threshold=1e-4, stall_steps=5):
        q = q0.copy()
        q_hist = [q.copy()]
        p_hist = [self.arm.forward_kinematics(q)[0]]
        err_hist = [np.linalg.norm(p_hist[-1] - p_target)]
        stall_counter = 0
        step = 0

        for step in range(max_steps):
            dq, info = self.solve(q, p_target)
            q_new = q + dq * self.dt
            q_new = self.arm.clip_joints(q_new)
            max_delta = np.max(np.abs(q_new - q))
            q = q_new
            q_hist.append(q)
            p = self.arm.forward_kinematics(q)[0]
            p_hist.append(p)
            err = np.linalg.norm(p - p_target)
            err_hist.append(err)

            if err < tol:
                print(f"DMPC: Objetivo alcanzado en {step+1} pasos. Error: {err:.5f}")
                break
            if max_delta < stall_threshold:
                stall_counter += 1
                if stall_counter >= stall_steps:
                    print(f"DMPC: Movimiento detenido tras {stall_steps} pasos.")
                    break
            else:
                stall_counter = 0
        else:
            print(f"DMPC: Máximo de pasos alcanzado ({max_steps}). Error final: {err_hist[-1]:.5f}")

        return {
            'q': np.array(q_hist),
            'p': np.array(p_hist),
            'error': np.array(err_hist),
            'steps': step+1,
            'success': err_hist[-1] < tol
        }