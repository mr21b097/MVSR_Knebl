import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
import os
import json
from PIL import Image

# Kamera-Intrinsics
K = {
    "fx": 616.741455078125,
    "fy": 616.919677734375,
    "cx": 324.817626953125,
    "cy": 238.0455780029297
}

# Pfade relativ zu deinem Skriptordner
model_dir = "models"
rgb_base = os.path.join("data", "rgb")
ann_base = os.path.join("data", "annotations")

os.makedirs(rgb_base, exist_ok=True)
os.makedirs(ann_base, exist_ok=True)

# Alle .obj-Dateien im models-Ordner
model_files = [f for f in os.listdir(model_dir) if f.endswith(".obj")]

# Pyrender Initialisierung
r = pyrender.OffscreenRenderer(viewport_width=400, viewport_height=400)
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
camera_pose = np.array([
    [1.0,  0.0,  0.0,  0.0],
    [0.0,  1.0,  0.0,  0.0],
    [0.0,  0.0,  1.0,  1.5],  # Kamera-Z-Abstand
    [0.0,  0.0,  0.0,  1.0],
])

# Pro Modell ein Bild erzeugen
for model_file in model_files:
    model_path = os.path.join(model_dir, model_file)
    class_name = os.path.splitext(model_file)[0]
    print(f"▶️  Rendere: {class_name}")

    # Ausgabeordner für dieses Objekt
    rgb_outdir = os.path.join(rgb_base, class_name)
    ann_outdir = os.path.join(ann_base, class_name)
    os.makedirs(rgb_outdir, exist_ok=True)
    os.makedirs(ann_outdir, exist_ok=True)

    # CAD-Modell laden & vorbereiten
    mesh_trimesh = trimesh.load(model_path)
    mesh_trimesh.apply_scale(1.0 / np.max(mesh_trimesh.extents))
    mesh_trimesh.apply_translation(-mesh_trimesh.centroid)
    rot = trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0])
    mesh_trimesh.apply_transform(rot)
    mesh = pyrender.Mesh.from_trimesh(mesh_trimesh)

    # Szene aufbauen
    scene = pyrender.Scene()
    node = scene.add(mesh)
    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                               innerConeAngle=np.pi/16.0,
                               outerConeAngle=np.pi/6.0)
    scene.add(light, pose=camera_pose)

    # Rendern
    color, _ = r.render(scene)
    if color is None or np.sum(color) == 0:
        print(f"  Kein Bild gerendert für {class_name}, überspringe...")
        continue

    # Speichern
    rgb_path = os.path.join(rgb_outdir, "0000.png")
    Image.fromarray(color).save(rgb_path)

    # JSON-Annotation speichern
    annotation = {
        "class": class_name,
        "image_path": rgb_path.replace("\\", "/"),
        "K": K,
        "camera_pose": camera_pose.tolist()
    }
    ann_path = os.path.join(ann_outdir, "0000.json")
    with open(ann_path, "w") as f:
        json.dump(annotation, f, indent=4)

print(" Alle Modelle erfolgreich gerendert.")
