import os

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

# ---- AYARLAR ----
MODEL_PATH = "vgg16_flowers102.keras"
IMG_SIZE = 224

app = Flask(__name__, static_folder=".", static_url_path="")

# ---- Modeli yükle ----
print("Loading model:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

# ---- Flowers102 class names (TFDS) ----
CLASS_NAMES = None
# 0..101 (TFDS order) — Oxford Flowers 102 class names
CLASS_NAMES = [
    "pink primrose",
    "hard-leaved pocket orchid",
    "canterbury bells",
    "sweet pea",
    "english marigold",
    "tiger lily",
    "moon orchid",
    "bird of paradise",
    "monkshood",
    "globe thistle",
    "snapdragon",
    "colt's foot",
    "king protea",
    "spear thistle",
    "yellow iris",
    "globe-flower",
    "purple coneflower",
    "peruvian lily",
    "balloon flower",
    "giant white arum lily",
    "fire lily",
    "pincushion flower",
    "fritillary",
    "red ginger",
    "grape hyacinth",
    "corn poppy",
    "prince of wales feathers",
    "stemless gentian",
    "artichoke",
    "sweet william",
    "carnation",
    "garden phlox",
    "love in the mist",
    "mexican aster",
    "alpine sea holly",
    "ruby-lipped cattleya",
    "cape flower",
    "great masterwort",
    "siam tulip",
    "lenten rose",
    "barbeton daisy",
    "daffodil",
    "sword lily",
    "poinsettia",
    "bolero deep blue",
    "wallflower",
    "marigold",
    "buttercup",
    "oxeye daisy",
    "common dandelion",
    "petunia",
    "wild pansy",
    "primula",
    "sunflower",
    "pelargonium",
    "bishop of llandaff",
    "gaura",
    "geranium",
    "orange dahlia",
    "pink-yellow dahlia?",
    "cautleya spicata",
    "japanese anemone",
    "black-eyed susan",
    "silverbush",
    "californian poppy",
    "osteospermum",
    "spring crocus",
    "bearded iris",
    "windflower",
    "tree poppy",
    "gazania",
    "azalea",
    "water lily",
    "rose",
    "thorn apple",
    "morning glory",
    "passion flower",
    "lotus",
    "toad lily",
    "anthurium",
    "frangipani",
    "clematis",
    "hibiscus",
    "columbine",
    "desert-rose",
    "tree mallow",
    "magnolia",
    "cyclamen",
    "watercress",
    "canna lily",
    "hippeastrum",
    "bee balm",
    "ball moss",
    "foxglove",
    "bougainvillea",
    "camellia",
    "mallow",
    "mexican petunia",
    "bromelia",
    "blanket flower",
    "trumpet creeper",
    "blackberry lily",
]

def prepare_image(file_storage):
    img = Image.open(file_storage.stream).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.vgg16.preprocess_input(arr)
    return arr

@app.get("/")
def index():
    return send_from_directory(".", "index.html")

@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file field named 'image'"}), 400

    f = request.files["image"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    x = prepare_image(f)
    probs = model.predict(x, verbose=0)[0]  # (102,)
    class_id = int(np.argmax(probs))
    confidence = float(probs[class_id])

    result = {
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id] if CLASS_NAMES else None,
        "confidence": confidence,
        "top5": []
    }

    top5_idx = np.argsort(probs)[::-1][:5]
    for idx in top5_idx:
        idx = int(idx)
        item = {
            "class_id": idx,
            "class_name": CLASS_NAMES[idx] if CLASS_NAMES else None,
            "prob": float(probs[idx])
        }
        result["top5"].append(item)

    return jsonify(result)

if __name__ == "__main__":
    # Local:
    # http://127.0.0.1:5001
    app.run(host="0.0.0.0", port=5001, debug=True)
