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

  

 ```bash
 git clone <URL-del-repo>

 cd Proyecto-AEOC/codigo
 ```

  

2. **Crear el entorno conda** (solo la primera vez)

  

 ```bash
 conda env create -f environment.yml
 ```

  

   El archivo `environment.yml` contiene Python 3.12, CasADi, IPOPT y todas las dependencias necesarias.

  

3. **Activar el entorno**

  

 ```bash
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

---

# Resultados

Video: [Video Proyecto AEOC - Santiago Suaza & Daniel Niño](https://youtu.be/UMxWIIUtgmQ)

Informe: [Proyecto_final_AEOC - Santiago Suaza - Daniel Niño.pdf](https://github.com/user-attachments/files/29830929/Proyecto_final_AEOC.-.Santiago.Suaza.-.Daniel.Nino.pdf)


https://github.com/user-attachments/assets/87c3c4a6-a697-4449-99ea-0475cd3caf7f



https://github.com/user-attachments/assets/dde534d7-272e-41a5-9c2b-d0294b1d71bd



https://github.com/user-attachments/assets/df7127fd-205e-4c70-854b-fa26a2bf98b4



https://github.com/user-attachments/assets/e4d6ce74-1bba-44a3-b51e-35e3b8d56c3f



https://github.com/user-attachments/assets/4c3eafa1-80b1-46ac-ba19-0c2f3d2cd93a



https://github.com/user-attachments/assets/38d846f1-f5a4-461c-8f2c-f115e4b5bcc9



https://github.com/user-attachments/assets/111be9c4-2549-4ab0-a578-8e25f4f516de



https://github.com/user-attachments/assets/ebabf7de-5ba3-4de9-9ca4-ef2ba65becc3



https://github.com/user-attachments/assets/0fe06014-cad0-4cae-878d-48ade193a706



https://github.com/user-attachments/assets/816ea9fa-c645-4480-8be3-13f637f7d30f



https://github.com/user-attachments/assets/d51ca4bc-8cd0-40b3-bfba-7a216cd19ac7


