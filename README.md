# Control de Brazo Robótico 3‑DOF con Teoría de Juegos y MPC

  

Proyecto final de control distribuido para un brazo robótico redundante de 3 grados de libertad.

Compara tres estrategias de coordinación:

  

- **MPC centralizado** – optimización no lineal con CasADi / IPOPT.

- **Juego cooperativo** – equilibrio de Nash mediante mejor respuesta (Gauss‑Seidel).

- **Híbrido distribuido (DMPC‑Nash)** – control predictivo con negociación entre articulaciones.

  

El objetivo es que el efector final siga trayectorias complejas en el espacio cartesiano

(flor, Lissajous, estrella, trébol, círculo) y evaluar las compensaciones entre

error de seguimiento, suavidad y costo computacional.

  

---

  

## Requisitos previos

  

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (o Anaconda) instalado.

- Git (para clonar el repositorio).

  

---

  

## Instalación y configuración

  

1. **Clonar el repositorio**

  

   ```

   git clone <URL-del-repo>

   cd Proyecto-AEOC/codigo

   ```

  

2. **Crear el entorno conda** (solo la primera vez)

  

   ```

   conda env create -f environment.yml

   ```

  

   El archivo `environment.yml` contiene Python 3.12, CasADi, IPOPT y todas las dependencias necesarias.

  

3. **Activar el entorno**

  

   ```

   conda activate brazo_robotico

   ```

  

---

  

## Estructura del proyecto

  

```

codigo/

├── arm_model/               # Modelo cinemático y visualización 3D

│   ├── three_dof_arm.py

│   └── visualizer.py

├── solvers/                 # Algoritmos de control

│   ├── mpc_centralized.py   # MPC centralizado con CasADi/IPOPT

│   ├── game_cooperative.py  # Juego cooperativo (equilibrio de Nash)

│   └── hybrid.py            # Control híbrido distribuido (DMPC‑Nash)

├── utils/                   # Configuración y herramientas de gráficos

│   ├── config.py

│   └── plotting.py

├── experiments/             # Script principal de experimentos

│   └── run_experiments.py

├── test_mpc.py              # Prueba individual del MPC

├── test_game.py             # Prueba individual del juego cooperativo

├── test_hybrid.py           # Prueba individual del híbrido

├── environment.yml          # Definición del entorno conda

└── README.md

```

  

---

  

## Ejecutar los experimentos

  

El script `experiments/run_experiments.py` permite elegir una figura, previsualizarla y

luego ejecutar los tres controladores, generando videos comparativos (MP4) y una gráfica

de error de seguimiento.

  

```bash

python experiments/run_experiments.py

```

  

El programa pedirá que elijas una trayectoria (1‑5) y mostrará la trayectoria deseada

antes de comenzar. Los videos se guardan en la misma carpeta con nombres descriptivos.

  

---

  

## Probar un controlador individual

  

Para depurar o ajustar parámetros de un solo solver:

  

```bash

python test_mpc.py        # solo MPC centralizado

python test_game.py       # solo juego cooperativo

python test_hybrid.py     # solo híbrido distribuido

```

  

Cada script genera gráficas de evolución articular, error y un GIF de la trayectoria.

  

---

  

## Configuración

  

Todos los parámetros de simulación y pesos de los controladores se encuentran en

`utils/config.py`. Allí puedes modificar:

  

- Límites articulares y de velocidad.

- Horizonte de predicción, pesos `Q`, `R`, términos integrales, etc.

- Lista de puntos objetivo para experimentos personalizados.

  

---

  

## Dependencias

  

El entorno conda incluye:

| Paquete      | Uso principal                     |
|--------------|-----------------------------------|
| Python 3.12  | Lenguaje base                     |
| CasADi       | Modelado y optimización no lineal |
| IPOPT        | Solver de programación no lineal  |
| NumPy        | Cálculos numéricos                |
| SciPy        | Optimización (juego cooperativo)  |
| Matplotlib   | Gráficos y visualización 3D       |
| OpenCV       | Generación de videos MP4          |
| ImageIO      | Generación de GIFs                |

  

---

  

## Referencias

  

1. J. Zhang, L. Jin, Y. Wang, and C. Yang, “A collaboration scheme for controlling multi‑manipulator system: A game‑theoretic perspective,” *IEEE/ASME Trans. Mechatronics*, vol. 28, no. 2, pp. 149–158, 2023.

2. Y. Zhou, W. Jin, and X. Wang, “Distributed differentiable dynamic game for multi‑robot coordination,” *IEEE Robot. Autom. Lett.*, vol. 8, no. 7, pp. 4139–4146, 2023.

3. K. Hu, Z. Ma, S. Zou, J. Li, and J. Zhang, “Game‑based social learning particle swarm optimizer for inverse kinematics of robotic arms,” *IEEE Robot. Autom. Lett.*, vol. 9, no. 8, pp. 7262–7269, 2024.

4. H. Xu, C. Xue, Q. Chen, J. Yang, and B. Liang, “Continuous multi‑target approaching control of hyper‑redundant manipulators based on reinforcement learning,” *Mathematics*, vol. 12, no. 23, Art. no. 3822, 2024.

  

---

  

## Autores

  

- Santiago Suaza – Universidad de los Andes

- Daniel Niño – Universidad de los Andes
