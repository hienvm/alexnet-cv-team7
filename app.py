# Import necessary modules from Flask and Keras libraries
import torch
import torchvision
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from improved.data_transforms import prepreprocess
from improved.eval import predict
from improved.model import AlexNetImproved

app = Flask("Medical Diagnosis")

# classes = ["classA", "classB", "healthy"]

classes = [
    "actinic keratosis",
    "basal cell carcinoma",
    "dermatofibroma",
    "healthy",
    "melanoma",
    "nevus",
    "pigmented benign keratosis",
    "seborrheic keratosis",
    "squamous cell carcinoma",
    "vascular lesion",
]

model = AlexNetImproved(num_classes=len(classes))
model.load_state_dict(
    torch.load("models/skin.model.pt", map_location=torch.device("cpu"))
)
preprocess = torch.load("models/skin.model.pt", map_location=torch.device("cpu"))


# home page
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# API for image upload and prediction, accepting POST requests
@app.route("/", methods=["POST"])
def diagnose():
    """
    Returns:
    {
        'label': 'Most probable class',
        'probs': {
            'classA': 0.w2,
            'classB': 0.3,
            ...
        }
    }
    """
    # convert to PIL image
    img_file = request.files["img"]

    # img_url = "tmp/" + img_file.filename

    img_url = "static/images/" + secure_filename(img_file.filename)

    img_file.save(img_url)

    img = torchvision.io.read_image(img_url).unsqueeze_(0)

    # Probabilities
    prediction = predict(model, prepreprocess(img)).flatten()

    # Most probable class
    label = classes[prediction.argmax()]

    # class -> probability
    class_probs = {classes[i]: p * 100 for i, p in enumerate(prediction.tolist())}

    class_probs = dict(
        sorted(class_probs.items(), reverse=True, key=lambda item: item[1])
    )

    return render_template(
        "index.html",
        result={
            "image_path": img_url,
            "label": label,
            "class_probs": class_probs,
        },
    )
    # return jsonify({'label': label, 'probs': probs})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
