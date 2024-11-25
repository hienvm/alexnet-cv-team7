# Import necessary modules from Flask and Keras libraries
from improved.model import AlexNetImproved
from improved.eval import predict
from flask import Flask, request, render_template, jsonify
from improved.data_transforms import prepreprocess
import torchvision

app = Flask('Medical Diagnosis')

classes = ['classA', 'classB', 'healthy']

model = AlexNetImproved(num_classes=len(classes))
# model.load_state_dict(torch.load('models/'))

# home page
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# API for image upload and prediction, accepting POST requests
@app.route("/", methods=["POST"])
def diagnose():
    '''
    Returns:
    {
        'label': 'Most probable class',
        'probs': {
            'classA': 0.2,
            'classB': 0.3,
            ...
        }
    }
    '''
    # convert to PIL image
    img_file = request.files["img"]
    
    img_url = 'tmp/' + img_file.filename
    
    img_file.save(img_url)
    
    img = torchvision.io.read_image(img_url)[:-1].unsqueeze_(0)
    
    # Probabilities
    prediction = predict(
        model, 
        prepreprocess(img)
    ).flatten()

    # Most probable class
    label = classes[prediction.argmax()]
    
    # class -> probability
    class_probs = { classes[i]: p*100 for i, p in enumerate(prediction.tolist()) }

    return render_template('index.html', result={'label': label, 'class_probs': class_probs})
    # return jsonify({'label': label, 'probs': probs})


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)