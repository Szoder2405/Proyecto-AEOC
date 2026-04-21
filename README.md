# Control de Brazo Robótico 3-DOF con Teoría de Juegos y MPC

Proyecto final de control distribuido para un brazo robótico redundante de 3 grados de libertad, comparando:
- **MPC centralizado** con CasADi/IPOPT
- **Control basado en juegos cooperativos** (equilibrio de Nash)

## Estructura del proyecto
codigo/
├── arm_model/ # Modelo cinemático y visualización
│ ├── three_dof_arm.py
│ └── visualizer.py
├── solvers/ # Algoritmos de control
│ ├── mpc_centralized.py
│ └── game_cooperative.py
├── utils/ # Utilidades
│ ├── config.py
│ └── plotting.py
├── experiments/ # Scripts de experimentos
├── test_mpc.py # Prueba individual MPC
├── test_game.py # Prueba individual juego cooperativo
├── requirements.txt # Dependencias
└── README.md
# Uso
## Probar MPC centralizado
~~~bash
python test_mpc.py
~~~

## Probar juego cooperativo
~~~bash
python test_game.py
~~~

## Configuración
Los parámetros de simulación y control se encuentran en *utils/config.py*.

## Dependencias
- Python 3.8+

- NumPy

- Matplotlib

- CasADi (con IPOPT)

- SciPy

- ImageIO (para GIFs)

# Autores
[Santiago Suaza] - Universidad de los Andes
[Daniel Niño] - Universidad de los Andes
