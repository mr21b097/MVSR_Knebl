# 6D Pose Estimation - Morobot



## Nutzung
python train_yolo.py für YOLO-Training starten

python detect_and_icp.py

Ergebnisse unter `out_vis/` prüfen




## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 


## Projektübersicht
In diesem Projekt wurde ein Objekterkennungs- und Visualisierungssystem für den Morobot entwickelt.  
Grundlage sind **RGB-Bilder**, **Tiefenbilder** und **3D-Modelle** im `.obj`-Format.
Die genauere Beschreibung der Implementierung und der verwendeten Methoden befindet sich in der Dokumentation.

---

## System in meiner Implementierung, keine Anforderungen bis auf Versionen
- Windows 11 Home 64-bit (getestet)
- CPU: Intel Core i5-12450H
- GPU: NVIDIA GeForce RTX 3050
- CUDA-Treiber: 12.5
- Python: 3.10.18 (Conda environment `happypose`)

---

## Installation

### Conda-Umgebung erstellen
```bash
conda env create -f environment.yml
conda activate happypose
```

### Alternativ mit pip
```bash
pip install -r requirements.txt
```
---




## Ablauf
1. **YOLO Training**
   - Annotation mit [CVAT](https://www.cvat.ai/)
   - Training mit `yolov8m.pt` / `yolov8l.pt`
   - Synthetische Daten durch Rendering aus CAD-Modellen (pyrender)

2. **Pose-Schätzung**
   - Bounding Boxes als Initialisierung
   - Registrierung mit **RANSAC** + FPFH-Features
   - Verfeinerung mit **ICP** (optional: Colored ICP)
   - Prüfung verschiedener 180°-Rotationen (Orientierungsfehler)

3. **Ergebnisse**
   - Visualisierte Posen (`out_vis/`)
   - Ausgabe: 4×4-Transformationsmatrix

---

## Bekannte Einschränkungen
- Ergebnisse sind teilweise instabil (springende Orientierungen)
- ICP allgemein sehr anfällig
- Unterschiede in CAD-Modellen (Skalierung, Orientierung)
- Ungenaue Tiefendaten erschweren Initialisierung
- Synthetische Daten nur eingeschränkt auf reale Aufnahmen übertragbar

---

## Zusammenfassung
Die Pipeline kombiniert YOLO-Detektion mit RANSAC/ICP-Registrierung und ermöglicht prinzipiell eine 6D-Pose-Schätzung.  
Die Methodik ist funktionsfähig, benötigt jedoch Optimierungen bei Datenqualität, ICP-Parametern und Modell-Normalisierung, um funktionstüchtig zu werden.

---

