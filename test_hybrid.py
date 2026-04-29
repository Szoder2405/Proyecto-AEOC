"""Prueba del Game-Theoretic MPC (híbrido real)."""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arm_model.three_dof_arm import ThreeDOFArm
from arm_model.visualizer import ArmVisualizer
from solvers.hybrid import DistributedMPC
from utils.config import ARM_CONFIG, SIM_CONFIG, HYBRID_CONFIG
from utils.plotting import plot_joint_trajectories, plot_position_error

def main():
    arm = ThreeDOFArm(**ARM_CONFIG)
    q0 = np.array([0.2, 0.5, 0.1])
    p_target = np.array([0.9, 0.15, 0.6])

    gt_mpc = DistributedMPC(
        arm,
        horizon=HYBRID_CONFIG['horizon'],
        dt=SIM_CONFIG['dt'],
        Q_pos=HYBRID_CONFIG['Q_pos'],
        r_vel=HYBRID_CONFIG['r_vel'],
        w_acc=HYBRID_CONFIG['w_acc'],
        w_int=HYBRID_CONFIG['w_int'],   
        Q_term=HYBRID_CONFIG['Q_term'],
        game_iters=HYBRID_CONFIG['game_iters'],
        v_max=HYBRID_CONFIG['v_max'],
    )

    result = gt_mpc.run_closed_loop(q0, p_target,
                                    max_steps=SIM_CONFIG['max_iter'],
                                    tol=SIM_CONFIG['tolerance'],
                                    stall_threshold=SIM_CONFIG['stall_threshold'],
                                    stall_steps=SIM_CONFIG['stall_steps'])

    print(f"\nÉxito: {result['success']}, pasos: {result['steps']}, error final: {result['error'][-1]:.5f}")

    plot_joint_trajectories(result['q'], SIM_CONFIG['dt'], title="GT‑MPC - Trayectorias articulares")
    plot_position_error([result['error']], ['GT‑MPC'], title="Error de posición - GT‑MPC")

    viz = ArmVisualizer(arm, figsize=(10,8))
    viz.animate_trajectory([q for q in result['q']], target=p_target, save_path="gt_mpc_trajectory.gif")
    viz.close()

if __name__ == "__main__":
    main()