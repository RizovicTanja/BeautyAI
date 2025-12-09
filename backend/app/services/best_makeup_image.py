import os
import cv2
import mediapipe as mp
import numpy as np
from app.services.face_shape_detector import detect_face_shape

def face_region_mask(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
    results = mp_face.process(image_rgb)

    if not results.multi_face_landmarks:
        return None, None

    landmarks = results.multi_face_landmarks[0]

    h, w, _ = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # koristimo obraze (landmark 234-454) i čelo (landmark 10-338)
    points = []
    for i in list(range(234, 454)) + list(range(10, 338)):
        lm = landmarks.landmark[i]
        x, y = int(lm.x * w), int(lm.y * h)
        points.append((x, y))

    points = np.array(points, np.int32)
    cv2.fillConvexPoly(mask, points, 255)

    return image, mask

def avg_hsv_region(image, mask):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(hsv, mask=mask)
    return np.array(mean_hsv[:3])

def select_best_style_image_region(user_image_path, style_folder, user_face_shape):
    user_image, user_mask = face_region_mask(user_image_path)
    if user_image is None:
        return None

    user_hsv = avg_hsv_region(user_image, user_mask)
    scored_images = []

    for filename in os.listdir(style_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(style_folder, filename)
            example_image, example_mask = face_region_mask(img_path)
            if example_image is None:
                continue

            example_hsv = avg_hsv_region(example_image, example_mask)
            color_dist = np.linalg.norm(user_hsv - example_hsv)
            example_face_shape = detect_face_shape(img_path)
            shape_penalty = 0 if example_face_shape == user_face_shape else 1

            score = color_dist + 50 * shape_penalty #zasto 50 me je pitala 
            scored_images.append((score, img_path))

    if not scored_images:
        return None

    # sort i uzmi top 3
    scored_images.sort(key=lambda x: x[0])
    top_images = [img for _, img in scored_images[:3]]

    # random odabir
    return np.random.choice(top_images)
