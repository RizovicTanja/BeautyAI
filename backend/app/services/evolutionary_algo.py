import random
import colorsys
from app.utils.data_loader import load_products

# --- Undertone korisnika u HSV hue ---
USER_TONES = {
    "topao": (0, 50),
    "neutralan": (50, 100),
    "hladan": (180, 270)
}

# --- Konverzije ---
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hue(rgb):
    r, g, b = [x/255.0 for x in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360

# Procenat boja proizvoda koje se uklapaju u undertone korisnikan (2 od 3 boje 2/3)
def ctone_score(product_colors, user_tone):
    hue_min, hue_max = USER_TONES[user_tone]
    matches = 0
    for color in product_colors:
        rgb = hex_to_rgb(color["hex_value"])
        h = rgb_to_hue(rgb)
        if hue_min <= h <= hue_max:
            matches += 1
    return matches / len(product_colors) if product_colors else 0

# --- Score funkcija za jedan proizvod ---
def evaluate_score(product, user_tone, max_price, alpha=0.5, beta=0.3, gamma=0.2):
    tone_score = ctone_score(product.get("product_colors", []), user_tone)
    price = float(product.get("price") or 0)
    price_score = 1 - price / max_price if max_price > 0 else 0
    rating_score = float(product.get("rating") or 4) / 5
    return alpha*tone_score + beta*price_score + gamma*rating_score

# --- GA za kombinaciju proizvoda po kategorijama ---
def genetic_recommendation(user_tone, product_types=["foundation", "blush", "lipstick", "bronzer", "eyeshadow"], 
                           population_size=10, generations=5, alpha=0.5, beta=0.3, gamma=0.2,
                           mutation_rate=0.1):

    products = load_products()
    max_price = max(float(p.get("price") or 0) for p in products)

    # Kreiranje random jedinke (kombinacija proizvoda po kategorijama, bira slucajan proizvod)
    def random_individual():
        individual = {}
        for cat in product_types:
            cat_products = [p for p in products if p.get("product_type")==cat]
            individual[cat] = random.choice(cat_products) if cat_products else None
        return individual

    # Suma score-ova svih proizvoda u jedinki, koliko je kombinacija dobra, prosecna vrednost
    def fitness(individual):
        total_score = 0
        for cat, prod in individual.items():
            if prod:
                total_score += evaluate_score(prod, user_tone, max_price, alpha, beta, gamma)
        return total_score / len(product_types)

    # Populacija
    population = [random_individual() for _ in range(population_size)]

    for _ in range(generations):
        # Sortiraj populaciju po fitness-u
        population.sort(key=fitness, reverse=True)
        parents = population[:population_size//2]  # Top 50%

        # Ukrštanje i mutacija
        offspring = []
        while len(offspring) < population_size//2:
            p1, p2 = random.sample(parents, 2)
            child = {}
            for cat in product_types:
                # Ukrštanje: bira proizvod jednog od roditelja
                chosen = random.choice([p1[cat], p2[cat]])
                # Mutacija: mala šansa da se zameni random proizvodom iz te kategorije
                if random.random() < mutation_rate:
                    cat_products = [p for p in products if p.get("product_type")==cat]
                    if cat_products:
                        chosen = random.choice(cat_products)
                child[cat] = chosen
            offspring.append(child)

        population = parents + offspring

    # Najbolja jedinka
    best = max(population, key=fitness)
    return best
