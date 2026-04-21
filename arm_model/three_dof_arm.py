"""
Modelo de brazo robótico antropomórfico de 3 grados de libertad.
Articulaciones: Hombro (base), Codo, Muñeca.
Todas son rotacionales.
"""

import numpy as np
from typing import Tuple, List


class ThreeDOFArm:
    """
    Brazo robótico de 3 DOF con configuración similar a un brazo humano.
    - Eslabón 1: del hombro al codo (longitud L1)
    - Eslabón 2: del codo a la muñeca (longitud L2)
    - Eslabón 3: de la muñeca a la mano/efector final (longitud L3)
    """

    def __init__(self, l1: float = 0.5, l2: float = 0.4, l3: float = 0.2,
             q_min: np.ndarray = None, q_max: np.ndarray = None):
        """
        Inicializa el brazo con longitudes de eslabones.

        Args:
            l1: Longitud del primer eslabón (hombro a codo)
            l2: Longitud del segundo eslabón (codo a muñeca)
            l3: Longitud del tercer eslabón (muñeca a efector final)
        """
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

        if q_min is not None:
            self.q_min = q_min
        else:
            self.q_min = np.array([-np.pi, -np.pi/2, -np.pi/2])
        
        if q_max is not None:
            self.q_max = q_max
        else:
            self.q_max = np.array([np.pi, np.pi/2, np.pi/2])

    def dh_transform(self, theta: float, d: float, a: float, alpha: float) -> np.ndarray:
        """
        Calcula la matriz de transformación homogénea para un eslabón según DH.

        Args:
            theta: Ángulo de la articulación (variable)
            d: Desplazamiento a lo largo de z_{i-1}
            a: Longitud del eslabón a lo largo de x_i
            alpha: Ángulo de torsión alrededor de x_i

        Returns:
            Matriz 4x4 de transformación homogénea.
        """
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)

        T = np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0,        sa,       ca,      d],
            [0,         0,        0,      1]
        ])
        return T

    def forward_kinematics(self, q: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Calcula la cinemática directa del brazo.

        Args:
            q: Vector de ángulos articulares [theta1, theta2, theta3] en radianes.

        Returns:
            - Posición (x, y, z) del efector final.
            - Lista de posiciones de cada articulación (hombro, codo, muñeca, mano)
              para visualización.
        """
        # Definición de parámetros DH para un brazo antropomórfico 3-DOF
        # Eslabón 1: rotación alrededor de z (theta1), eslabón perpendicular (l1)
        T01 = self.dh_transform(q[0], 0, 0, np.pi/2)
        # Eslabón 2: rotación alrededor de z (theta2), eslabón l2
        T12 = self.dh_transform(q[1], 0, self.l1, 0)
        # Eslabón 3: rotación alrededor de z (theta3), eslabón l3
        T23 = self.dh_transform(q[2], 0, self.l2, 0)
        # Eslabón 4: efector final (traslación l3)
        T34 = self.dh_transform(0, 0, self.l3, 0)

        # Matrices acumuladas
        T02 = T01 @ T12
        T03 = T02 @ T23
        T04 = T03 @ T34

        # Posiciones de las articulaciones (origen de cada sistema de coordenadas)
        p0 = np.array([0, 0, 0])
        p1 = T01[:3, 3]
        p2 = T02[:3, 3]
        p3 = T03[:3, 3]
        p_hand = T04[:3, 3]

        return p_hand, [p0, p1, p2, p3, p_hand]

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Calcula el jacobiano geométrico del brazo (3x3).

        El jacobiano relaciona velocidades articulares con velocidad lineal del efector final:
            v = J(q) * q_dot

        Args:
            q: Vector de ángulos articulares [theta1, theta2, theta3] en radianes.

        Returns:
            Matriz jacobiana 3x3.
        """
        # Para un brazo antropomórfico, el jacobiano se puede calcular analíticamente
        s1 = np.sin(q[0])
        c1 = np.cos(q[0])
        s2 = np.sin(q[1])
        c2 = np.cos(q[1])
        s23 = np.sin(q[1] + q[2])
        c23 = np.cos(q[1] + q[2])

        # Posición del efector final relativa a cada articulación
        # Articulación 1 (hombro): contribución de los eslabones 2 y 3
        J1_x = -self.l2 * s1 * c2 - self.l3 * s1 * c23
        J1_y = self.l2 * c1 * c2 + self.l3 * c1 * c23
        J1_z = 0.0

        # Articulación 2 (codo)
        J2_x = -self.l2 * c1 * s2 - self.l3 * c1 * s23
        J2_y = -self.l2 * s1 * s2 - self.l3 * s1 * s23
        J2_z = self.l2 * c2 + self.l3 * c23

        # Articulación 3 (muñeca)
        J3_x = -self.l3 * c1 * s23
        J3_y = -self.l3 * s1 * s23
        J3_z = self.l3 * c23

        J = np.array([
            [J1_x, J2_x, J3_x],
            [J1_y, J2_y, J3_y],
            [J1_z, J2_z, J3_z]
        ])
        return J

    def get_joint_positions(self, q: np.ndarray) -> List[np.ndarray]:
        """
        Obtiene únicamente las posiciones de las articulaciones (para visualización).

        Args:
            q: Vector de ángulos articulares.

        Returns:
            Lista de posiciones 3D de cada articulación.
        """
        _, positions = self.forward_kinematics(q)
        return positions

    def check_joint_limits(self, q: np.ndarray) -> bool:
        """
        Verifica si los ángulos articulares están dentro de los límites.

        Args:
            q: Vector de ángulos articulares.

        Returns:
            True si todos los ángulos están dentro de límites.
        """
        return np.all(q >= self.q_min) and np.all(q <= self.q_max)

    def clip_joints(self, q: np.ndarray) -> np.ndarray:
        """
        Recorta los ángulos a los límites permitidos.

        Args:
            q: Vector de ángulos articulares.

        Returns:
            Vector recortado.
        """
        return np.clip(q, self.q_min, self.q_max)