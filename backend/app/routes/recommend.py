import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, Form, Query
from app.services.face_analysis import analyze_face
from app.services.fuzzy_system import fuzzy_makeup_recommendation
from app.services.best_makeup import select_best_products
from app.services.best_makeup_image import select_best_style_image_region

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "images", "obrada")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAKEUP_PATH = os.path.join(BASE_DIR, "data", "makeup_data.json")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# POST /api/recommend
@router.post("/")
async def recommend_image(file: UploadFile = File(...), style: str = Form("natural")):
    if not allowed_file(file.filename):
        return {"error": "Unsupported file type. Use png, jpg, or jpeg."}

    # --- snimi sliku korisnika ---
    file_ext = file.filename.split(".")[-1]
    file_name = f"{uuid.uuid4()}.{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, file_name)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # --- analiza lica ---
    analysis = analyze_face(file_path)
    undertone = analysis.get("undertone", "neutral")
    face_shape = analysis.get("face_shape", "oval")
    skin_rgb = analysis.get("skin_rgb", (230, 200, 180))  # prosečna boja kože

    # --- fuzzy sistem za stil ---
    fuzzy_result = fuzzy_makeup_recommendation(undertone, face_shape)

    # --- pronalazak najboljih proizvoda ---
    best_products = select_best_products(MAKEUP_PATH, undertone)

    # --- default vrednosti ---
    DEFAULT_IMAGE = "assets/default.png"
    DEFAULT_SHADE = "N/A"

    BACKEND_URL = "http://127.0.0.1:8000"  # FastAPI port


# --- pronalazak najbolje slike make-upa ---
# Slika iz foldera po izboru korisnika
    user_style_folder = os.path.join(BASE_DIR, f"data/images/{style}")
    user_chosen_best_image = select_best_style_image_region(file_path, user_style_folder, face_shape)

# Slika iz foldera po fuzzy preporuci
    fuzzy_style_folder = os.path.join(BASE_DIR, f"data/images/{fuzzy_result['recommended_style']}")
    fuzzy_best_image = select_best_style_image_region(file_path, fuzzy_style_folder, face_shape)

# --- Kreiranje URL-a za frontend ---
    def make_image_url(image_path):
        if image_path:
            rel_path = os.path.relpath(image_path, os.path.join(BASE_DIR, "data", "images"))
            return f"{BACKEND_URL}/static/{rel_path.replace(os.sep, '/')}"
        return f"{BACKEND_URL}/static/default.png"

    user_chosen_best_image_url = make_image_url(user_chosen_best_image)
    fuzzy_best_image_url = make_image_url(fuzzy_best_image)


    response = {
        "undertone": undertone,
        "face_shape": face_shape,
        "recommended_style": fuzzy_result["recommended_style"],
        "user_chosen_style_image": user_chosen_best_image_url,  # slika po izboru korisnika
        "fuzzy_recommended_style_image": fuzzy_best_image_url,  # slika po fuzzy preporuci
        "products": {}
    }



    for cat in ["foundation", "blush", "lipstick", "bronzer", "eyeshadow"]:
        p = best_products.get(cat, {})
        response["products"][cat] = {
            "brand": p.get("brand", ""),
            "name": p.get("name", ""),
            "shade": p.get("shade", DEFAULT_SHADE),
            "image": p.get("image", DEFAULT_IMAGE),
            "hex": p.get("hex", "")
        }

    return response
