#  YOLOv8 Object Detection & Orientation Estimation with LibTorch

Acest proiect implementează un sistem avansat de viziune artificială capabil să detecteze vehicule în imagini și să le estimeze orientarea (unghiul) folosind un pipeline secvențial bazat pe **YOLOv8**, **TensorRT** și **LibTorch (C++)**.



---

##  1. Scopul Proiectului
Obiectivul principal este determinarea orientării obiectelor detectate fără a utiliza senzori LiDAR. Sistemul funcționează în doi pași:
1.  **Detecție:** Localizarea vehiculelor folosind YOLOv8.
2.  **Clasificare Unghi:** Estimarea orientării printr-o rețea neuronală secundară antrenată să recunoască 8 sectoare de unghi (0°, 45°, 90°, etc.).

---

##  2. Arhitectura Tehnică & Etape

### Etapa 1: Detecția Obiectelor
- Utilizăm **YOLOv8** (pre-antrenat pe dataset-ul COCO).
- Modelul a fost convertit din `.pt` în formatul **TensorRT (.engine)** pentru a maximiza performanța pe GPU (NVIDIA GTX/Jetson).

### Etapa 2: Determinarea Orientării
- Am antrenat o rețea neuronală de clasificare în **PyTorch** folosind dataset-ul **KITTI**.
- Rețeaua a fost exportată via **TorchScript** pentru a fi încărcată nativ în C++.
- Folosim **Distribution Focal Loss (DFL)** pentru a rafina predicția unghiulară.

### Etapa 3: Integrare C++ & Afișare
- Pipeline-ul este scris integral în **C++17**.
- Rezultatele sunt procesate cu **OpenCV** pentru desenarea dreptunghiurilor orientate și a etichetelor de clasă.



---

##  3. Rezultate și Performanță
Sistemul a fost testat comparativ pe un set de 1598 de imagini (NVIDIA GTX 1650):

| Configurație | Timp Mediu (ms) | FPS Mediu |
| :--- | :--- | :--- |
| Python (PyTorch Native) | 40.65 | 24.60 |
| **LibTorch (C++)** | **51.21** | **19.52** |
| TensorRT (.engine) | 65.58 | 15.25 |

---

##  4. Instalare și Compilare

### Cerințe de sistem
- **OS:** Windows 10/11 sau Linux (Jetson Orin Nano)
- **Compiler:** MSVC (Visual Studio 2019/2022) sau GCC
- **Biblioteci:**
  - OpenCV 4.x
  - LibTorch (Varianta CUDA pentru GPU)
  - TensorRT (pentru modelele .engine)

### Pași pentru Build (CMake)
1. Configurează căile către LibTorch și OpenCV în `CMakeLists.txt`.
2. Creează folderul de build:
   ```bash
   mkdir build && cd build
   cmake ..
