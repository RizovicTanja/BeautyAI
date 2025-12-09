import os
import cv2
import json
import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_mesh

# Učitaj JSON sa oblicima lica
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
json_path = os.path.join(BASE_DIR, "data", "face_shapes.json")

if os.path.exists(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        FACE_SHAPES_DATA = json.load(f)
else:
    FACE_SHAPES_DATA = {}


def detect_face_shape(image_path):
    """
    Analizira lice i vraća tip oblika lica: oval, round, square, heart, oblong.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        # Koordinate ključnih tačaka
        jaw_left = np.array([landmarks[234].x * w, landmarks[234].y * h])
        jaw_right = np.array([landmarks[454].x * w, landmarks[454].y * h])
        chin = np.array([landmarks[152].x * w, landmarks[152].y * h])
        forehead = np.array([landmarks[10].x * w, landmarks[10].y * h])
        cheek_left = np.array([landmarks[50].x * w, landmarks[50].y * h])
        cheek_right = np.array([landmarks[280].x * w, landmarks[280].y * h])

        # Mere
        face_width = np.linalg.norm(jaw_right - jaw_left)
        face_height = np.linalg.norm(forehead - chin)
        ratio = face_height / face_width

        # Udaljenost jagodica
        cheek_width = np.linalg.norm(cheek_right - cheek_left)

        # Logika za klasifikaciju
        if ratio > 1.5:
            face_shape = "oblong"
        elif ratio > 1.35:
            face_shape = "oval"
        elif ratio < 1.1 and cheek_width / face_width > 0.9:
            face_shape = "round"
        elif abs(face_width - cheek_width) < 30:
            face_shape = "square"
        else:
            face_shape = "heart"

        return face_shape


def analyze_all_style_images(base_dir):
    """
    Analizira sve slike u data/images/{natural, glam, evening} i pravi JSON mapu:
    {
        "path/to/image.jpg": "oval",
        "path/to/image2.jpg": "square"
    }
    """
    shape_data = {}
    style_dirs = ["natural", "glam", "evening"]

    for style in style_dirs:
        folder = os.path.join(base_dir, "data", "images", style)
        for filename in os.listdir(folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(folder, filename)
                shape = detect_face_shape(img_path)
                if shape:
                    shape_data[img_path] = shape
                    print(f"[OK] {filename} → {shape}")
                else:
                    print(f"[X] {filename} → nije detektovano lice")

    # Sačuvaj rezultate
    json_path = os.path.join(base_dir, "data", "face_shapes.json")
    with open(json_path, "w") as f:
        json.dump(shape_data, f, indent=4)

    print(f"\ Sačuvano {len(shape_data)} zapisa u {json_path}")
    return shape_data

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(" Tražim slike u:", os.path.join(BASE_DIR, "data", "images"))
    analyze_all_style_images(BASE_DIR)

def get_face_shape(img_path):
    """
    Vraća oblik lica za datu sliku koristeći JSON.
    Ako nema podatka u JSON-u, pali se detekcija uživo.
    """
    shape = FACE_SHAPES_DATA.get(img_path)
    if shape is None:
        # fallback na detekciju ako JSON nema podatak
        shape = detect_face_shape(img_path)
    return shape

