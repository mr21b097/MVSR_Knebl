import os
import cv2
from ultralytics import YOLO

# Pfad zum synthetisch trainierten Modell
model_path = r"C:\Users\fabia\Desktop\dsfsdf\runs\detect\morobot_detect96\weights\best.pt"
model = YOLO(model_path)

# Verzeichnisse mit echten Testbildern
test_images_dir = r"C:\Users\fabia\Desktop\dsfsdf\images\train"
output_dir = r"C:\Users\fabia\Desktop\dsfsdf\predictions"
os.makedirs(output_dir, exist_ok=True)

# Testbilder durchgehen
for image_name in os.listdir(test_images_dir):
    if image_name.endswith('.png'):
        image_path = os.path.join(test_images_dir, image_name)
        img = cv2.imread(image_path)

        # Vorhersagen
        results = model(image_path)

        # Bounding Boxes einzeichnen
        for result in results[0].boxes.data.tolist():
            xmin, ymin, xmax, ymax, conf, cls = result
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)

        cv2.imwrite(os.path.join(output_dir, f"pred_{image_name}"), img)

print("[DONE] Visualisierungen auf Testbildern abgeschlossen.")
