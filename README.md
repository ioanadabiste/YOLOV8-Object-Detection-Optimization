# Proiect: Detectia si orientarea obiectelor utilizand YOLO si Libtorch

## 1. Scopul proiectului
Scopul acestui proiect este implementarea unui sistem care detecteaza obiecte intr-o imagine si determina orientarea fiecarui obiect, utilizand modele YOLO pre-antrenate si retele neuronale realizate in PyTorch, ulterior exportate in Libtorch pentru integrare in C.

---

## 2. Etapele proiectului

### Etapa 1: Detectia obiectelor
- Se utilizeaza un model **YOLO** (ex. YOLOv5 sau YOLOv12 disponibil pe GitHub) pentru detectarea obiectelor din imagini.  
- Nu este necesara antrenarea modelului, deoarece YOLO ofera deja modele pre-antrenate pentru detectia generala.  
- Rularea modelului se face in **Python**, iar rezultatele (bounding boxes) se exporta prin **Libtorch**.

### Etapa 2: Determinarea orientarii obiectelor
- Pentru fiecare obiect detectat, se determina **orientarea** acestuia.  
- Orientarea se obtine printr-o retea neuronala simpla antrenata in **PyTorch** pe orientari discrete.  
- Modelul se exporta ulterior in **Libtorch** pentru utilizare in C.

### Etapa 3: Afisarea rezultatelor
- Se afiseaza o imagine alba pe care sunt desenate **dreptunghiurile orientate** corespunzator obiectelor detectate.  
- Nu se utilizeaza LiDAR, proiectii 3D sau plasarea obiectelor in scena.  
- Implementarea finala se face in **C**, folosind biblioteca **Libtorch**.

---

## 3. Rezumat functional
Proiectul contine doua componente principale:
1. Detectia obiectelor – realizata cu YOLO.  
2. Estimarea orientarii – realizata printr-o retea neuronala PyTorch.  

Rezultatul final este o imagine cu obiectele detectate si orientate, reprezentate prin dreptunghiuri.
