from ultralytics import YOLO

# Pfad zur YAML-Datei
data_yaml = r"C:\Users\fabia\Desktop\knebl_poseestimation\data.yaml"

# Modell laden (für Robustheit größere Version)
model = YOLO("yolov8l.pt")  

# Training starten
model.train(
    data=data_yaml,
    epochs=150,  # Anzahl der Epochen
    imgsz=640,  # Größe der Bilder
    batch=2,   # Batchgröße
    workers=0,  # Anzahl der CPU-Kerne
    device=0,   # 0 für die erste GPU (falls vorhanden)
    name="morobot_detect",  # Name für das Training
)
