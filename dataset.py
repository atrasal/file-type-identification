import os
import requests
from PIL import Image
from io import BytesIO

API_KEY = "SfrpPhJB3I8a6fMqS9Lcbh2KvTI9Vh9p0yhmvctxEC4HZyjiDEdmy5kf"
OUTPUT_DIR = "downloaded_pngs"
QUERY_LIST = ["cat", "dog", "car", "mountain", "city", "technology", "space", "robot", "beach", "desert",
    "waterfall", "forest", "tree", "flower", "bird", "sky", "sun", "moon", "star", "cloud",
    "train", "airplane", "ship", "bicycle", "bus", "subway", "bridge", "castle", "tower", "street",
    "market", "people", "portrait", "child", "family", "wedding", "love", "music", "dance", "festival",
    "fireworks", "snow", "rain", "storm", "lightning", "sunset", "sunrise", "ocean", "river", "lake",
    "island", "jungle", "zebra", "lion", "tiger", "elephant", "giraffe", "kangaroo", "bear", "panda",
    "monkey", "penguin", "dolphin", "whale", "shark", "fish", "coral", "reef", "shell", "crab",
    "butterfly", "bee", "spider", "snake", "frog", "lizard", "horse", "cow", "sheep", "goat",
    "chicken", "duck", "turkey", "camel", "antelope", "deer", "fox", "wolf", "bat", "rat",
    "mouse", "hedgehog", "owl", "eagle", "hawk", "falcon", "parrot", "peacock", "swan", "goose",
    "pigeon", "sparrow", "woodpecker", "flamingo", "toucan", "octopus", "lobster", "jellyfish", "starfish", "seaweed",
    "volcano", "canyon", "valley", "hill", "cliff", "cave", "glacier", "iceberg", "aurora", "galaxy",
    "nebula", "planet", "asteroid", "comet", "spaceship", "rocket", "satellite", "astronaut", "alien", "ufo",
    "computer", "laptop", "phone", "tablet", "keyboard", "mouse", "monitor", "server", "robotics", "ai",
    "cyberpunk", "coding", "python", "javascript", "html", "css", "linux", "windows", "mac", "terminal",
    "circuit", "chip", "drone", "vr", "ar", "3d", "printer", "camera", "lens", "tripod",
    "microscope", "telescope", "calculator", "book", "notebook", "pen", "pencil", "paper", "chalk", "board",
    "school", "college", "university", "classroom", "teacher", "student", "exam", "graduation", "library", "science",
    "physics", "chemistry", "biology", "math", "engineering", "robot", "experiment", "lab", "formula", "dna",
    "cell", "bacteria", "virus", "vaccine", "doctor", "nurse", "hospital", "clinic", "surgery", "medicine",
    "pill", "capsule", "stethoscope", "xray", "mri", "ct", "ambulance", "emergency", "fire", "police",
    "army", "navy", "airforce", "gun", "tank", "helicopter", "war", "peace", "flag", "nation",
    "india", "usa", "china", "japan", "korea", "germany", "france", "brazil", "russia", "australia",
    "food", "fruit", "apple", "banana", "orange", "grape", "mango", "pineapple", "strawberry", "blueberry",
    "cherry", "watermelon", "kiwi", "peach", "plum", "pear", "lemon", "lime", "coconut", "avocado",
    "carrot", "potato", "tomato", "onion", "garlic", "pepper", "chili", "cabbage", "broccoli", "spinach",
    "bread", "rice", "pasta", "pizza", "burger", "sandwich", "cake", "cookie", "icecream", "chocolate",
    "coffee", "tea", "milk", "juice", "soda", "water", "wine", "beer", "whiskey", "cocktail",
    "kitchen", "restaurant", "chef", "table", "plate", "fork", "knife", "spoon", "bowl", "cup",
    "glass", "oven", "stove", "fridge", "microwave", "toaster", "blender", "recipe", "cooking", "baking",
    "home", "house", "apartment", "building", "architecture", "interior", "design", "furniture", "chair", "sofa",
    "table", "bed", "wardrobe", "shelf", "lamp", "curtain", "carpet", "wallpaper", "painting", "mirror",
    "garden", "yard", "balcony", "terrace", "patio", "fence", "gate", "door", "window", "roof",
    "garage", "driveway", "pathway", "pool", "pond", "fountain", "treehouse", "swing", "slide", "sandbox",
    "playground", "park", "bench", "picnic", "barbecue", "camping", "tent", "firepit", "hiking", "biking",
    "fishing", "hunting", "sports", "football", "basketball", "cricket", "tennis", "golf", "swimming",
    "running", "cycling", "yoga", "gym", "fitness", "exercise", "workout", "health", "wellness", "meditation",
    "relaxation", "spa", "massage", "therapy", "mentalhealth", "selfcare", "lifestyle", "fashion", "style",
    "clothing", "shoes", "accessories", "jewelry", "watch", "bag", "hat", "sunglasses", "makeup", "skincare",
    "hair", "nails", "beauty", "cosmetics", "perfume", "fragrance", "shopping", "mall", "store", "boutique",
    "sale", "discount", "brand", "logo", "advertisement", "marketing", "business", "finance", "investment",
    "economy", "stock", "trade", "currency", "banking", "credit", "debit", "loan", "mortgage",
    "insurance", "tax", "budget", "savings", "retirement", "wealth", "rich", "poor", "middleclass"]
PER_QUERY =1 # images per keyword

os.makedirs(OUTPUT_DIR, exist_ok=True)

headers = {
    "Authorization": API_KEY
}

import time
from requests.exceptions import RequestException

def safe_get(url, headers, retries=3, timeout=10):
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except RequestException as e:
            print(f"⚠️ Attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    return None

def download_images():
    count = 1
    for query in QUERY_LIST:
        url = f"https://api.pexels.com/v1/search?query={query}&per_page={PER_QUERY}"
        response = safe_get(url, headers=headers)
        if not response:
            print(f"❌ Skipping '{query}' after multiple failures.")
            continue

        try:
            data = response.json()
            photos = data.get("photos", [])
            if not photos:
                print(f"❌ No image found for: {query}")
                continue

            photo = photos[0]  # 1 image per keyword
            img_url = photo["src"]["large"]
            img_response = safe_get(img_url, headers={}, timeout=10)
            if not img_response:
                print(f"❌ Failed to download image for: {query}")
                continue

            img = Image.open(BytesIO(img_response.content)).convert("RGBA")
            img.save(os.path.join(OUTPUT_DIR, f"{count}.png"))
            print(f"✅ Saved {count}.png from query: {query}")
            count += 1

        except Exception as e:
            print(f"❌ Error processing '{query}': {e}")

download_images()