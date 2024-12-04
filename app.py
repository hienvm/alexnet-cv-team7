# Import necessary modules from Flask and Keras libraries
import torch
import torchvision
import os
import uuid
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
preprocess = torch.load("models/skin.preprocess.pt", map_location=torch.device("cpu"))


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
    img_files = request.files.getlist("images[]")

    # img_url = "tmp/" + img_file.filename

    img_urls = []

    for file in img_files:
        url = "static/images/" + secure_filename(file.filename)
        img_urls.append(url)
        file.save(url)

    prediction = torch.zeros(len(classes))
    img_batch = []
    for url in img_urls:
        # take only 3 RGB channels
        img = torchvision.io.read_image(url)[:3]
        img_batch.append(preprocess(img))
    img_batch = torch.stack(img_batch, 0)
    # take average
    prediction += predict(model, img_batch).mean(0)

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
            "image_path": img_urls,
            "label": label,
            "class_probs": class_probs,
        },
    )
    # return jsonify({'label': label, 'probs': probs})

# lưu các file ảnh ở 2 trường vào folder có đường dẫn /collected_samples/{tên_class}/{id}
@app.route("/add_samples", methods=["POST"])
def add_samples():
    # Lấy dữ liệu từ form
    symptom_images = request.files.getlist("symptom_images[]")  # Ảnh triệu chứng
    proof_images = request.files.getlist("proof_images[]")  # Ảnh minh chứng
    diagnosis_class = request.form.get("class")  # Tên class (disease)

    # Tạo ID duy nhất cho mẫu này
    unique_id = str(uuid.uuid4())

    # Tạo các thư mục lưu trữ
    base_path = os.path.join("collected_samples", diagnosis_class, unique_id)
    symptoms_path = os.path.join(base_path, "symptoms")  # Thư mục ảnh triệu chứng
    proofs_path = os.path.join(base_path, "proof")  # Thư mục ảnh minh chứng

    os.makedirs(symptoms_path, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
    os.makedirs(proofs_path, exist_ok=True)

    # Lưu ảnh triệu chứng vào thư mục "symptoms"
    for file in symptom_images:
        file_path = os.path.join(symptoms_path, secure_filename(file.filename))
        file.save(file_path)

    # Lưu ảnh minh chứng vào thư mục "proof"
    for file in proof_images:
        file_path = os.path.join(proofs_path, secure_filename(file.filename))
        file.save(file_path)

    # Phản hồi JSON xác nhận lưu trữ thành công
    return jsonify(
        {
            "message": "Data added successfully",
            "class": diagnosis_class,
            "id": unique_id,
        }
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
