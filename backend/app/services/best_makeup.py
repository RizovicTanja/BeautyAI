import json
from app.services.evolutionary_algo import USER_TONES, genetic_recommendation, rgb_to_hue

# --- Helper funkcija ---
def hex_to_rgb(hex_value):
    hex_value = hex_value.strip("#")
    return tuple(int(hex_value[i:i+2], 16) for i in (0, 2, 4))

# --- Glavna funkcija ---
def select_best_products(makeup_json_path, undertone):
    with open(makeup_json_path, 'r', encoding='utf-8') as f:
        products = json.load(f)

    categories = ["foundation", "blush", "lipstick", "bronzer", "eyeshadow"]
    best_matches = {}

    # Dobijamo najbolju kombinaciju proizvoda pomoću GA
    best_individual = genetic_recommendation(undertone)

    for cat in categories:
        prod = best_individual.get(cat)
        if not prod:
            continue

        # Prolazimo kroz sve boje proizvoda i biramo onu koja najviše odgovara podtonu
        best_color = None
        best_tone_score = -1
        for color in prod.get("product_colors", []):
            hex_value = color.get("hex_value")
            if not hex_value:
                continue  # preskoči boje bez hex vrednosti

            rgb = hex_to_rgb(hex_value)
            hue_min, hue_max = USER_TONES.get(undertone, (0, 360))
            h = rgb_to_hue(rgb)
            tone_score = 1.0 if hue_min <= h <= hue_max else 0.0

            if tone_score > best_tone_score:
                best_tone_score = tone_score
                best_color = color

        best_matches[cat] = {
            "category": cat,
            "brand": prod.get("brand", ""),
            "name": prod.get("name", ""),
            "shade": best_color.get("colour_name", "N/A").strip() if best_color else "N/A",
            "hex": best_color.get("hex_value", "") if best_color else "",
            "image": prod.get("api_featured_image", "")
        }

    return best_matches
