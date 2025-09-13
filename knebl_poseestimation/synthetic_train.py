from ultralytics import YOLO

# Pfad zur YAML-Datei, die wir erstellt haben
data_yaml = r"C:\Users\fabia\Desktop\knebl_poseestimation\synthetic_data.yaml"

# Modell laden (für Robustheit, hier nehmen wir eine größere Version)
model = YOLO("yolov8m.pt")  # Oder die Version, die du bevorzugst, z.B. yolov8n.pt

# Training starten
model.train(
    data=data_yaml,
    epochs=50,  # Anzahl der Epochen, kannst du anpassen
    imgsz=640,  # Größe der Bilder
    batch=16,   # Batchgröße
    workers=0,  # Anzahl der CPU-Kerne
    device=0,   # 0 für die erste GPU (falls vorhanden)
    name="morobot_detect",  # Name für das Training
)
