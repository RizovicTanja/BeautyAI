
import cv2
import numpy as np
import mediapipe as mp
import os
import uuid

def analyze_face(image_path: str):
    # Učitaj sliku
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Image not found"}

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --- Detekcija lica ---
    mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
    results = mp_face.process(image_rgb)

    if not results.multi_face_landmarks:
        return {"error": "No face detected"}

    landmarks = results.multi_face_landmarks[0]

    # --- Obeležavanje landmarka na slici ---
    for lm in landmarks.landmark:
        h, w, _ = image.shape
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

    # --- Heuristički podton ---
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 20, 70), (50, 255, 255))    #!!!!!!!!!!!!! zasto value (mora da bi odbacili sum)i izvori 
    avg_color = cv2.mean(hsv, mask=mask)
    hue = avg_color[0]

    if hue < 20:
        undertone = "topao"
    elif hue < 35:
        undertone = "neutralan"
    else:
        undertone = "hladan"

    # --- Oblik lica ---
    pts = [(lm.x, lm.y) for lm in landmarks.landmark]
    face_ratio = (pts[10][1] - pts[152][1]) / (pts[234][0] - pts[454][0])
    if face_ratio < 1.1:
        face_shape = "okruglo"
    elif face_ratio < 1.3:
        face_shape = "ovalno"
    else:
        face_shape = "izduženo"

    # --- Sačuvaj i prikaži sliku sa landmarkima ---
    save_dir = os.path.join(os.path.dirname(image_path), "faces_detected")
    os.makedirs(save_dir, exist_ok=True)
    output_filename = f"{uuid.uuid4()}.png"
    output_path = os.path.join(save_dir, output_filename)
    cv2.imwrite(output_path, image)

    # Otvori sliku na backendu 
    # try:
    #     if os.name == "nt":  # Windows
    #         os.startfile(output_path)
    #     else:  # Linux / macOS
    #         import subprocess
    #         subprocess.run(["xdg-open", output_path])
    # except Exception as e:
    #     print(f"Could not open image: {e}")

    print(f"Detected Face Shape: {face_shape}")
    print(f"Detected Undertone: {undertone}")
    print(f"Landmark image saved at: {output_path}")

    return {
        "undertone": undertone,
        "face_shape": face_shape,
        "image_path": output_path
    }
