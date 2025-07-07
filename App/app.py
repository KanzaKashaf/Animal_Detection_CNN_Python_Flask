# from flask import Flask, request, render_template, redirect, url_for
# from werkzeug.utils import secure_filename
# import os
# import keras
# from tensorflow.keras.models import load_model
# # import tensorflow as tf
# import numpy as np
# from PIL import Image

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# # Load your trained model
# model = load_model('model/animalDetection.h5')
# # model=keras.models.load_model('model/AnimalModel.h5')
# # load_model('')
# # model = keras.saving.load_model("model/my_model.keras")
# def prepare_image(image, target_size):
#     image = image.resize(target_size)
#     image = np.array(image)
#     if image.shape[2] == 4:  # Remove alpha channel if present
#         image = image[:, :, :3]
#     image = np.expand_dims(image, axis=0)  # Add batch dimension
#     return image

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return redirect(request.url)
#     file = request.files['file']
#     if file.filename == '':
#         return redirect(request.url)
#     if file:
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
        
#         # image = Image.open(filepath)
#         # prepared_image = prepare_image(image, target_size=(224, 224))  # Adjust target size based on your model

#         # prediction = model.predict(prepared_image)
#         # Process prediction as needed, e.g., draw bounding boxes on the image

#         # For now, let's just return the raw prediction
#         return render_template('index.html', prediction=1, image_url=url_for('static', filename='uploads/' + filename))

# if __name__ == '__main__':
#     app.run(debug=True)





# from flask import Flask, request, jsonify
# import tensorflow as tf
# # from tensorflow.keras.models import load_model
# import numpy as np
# from PIL import Image
# import os

# app = Flask(__name__)

# # Load your model
# # model_path = 'model\AnimalModel.h5'
# model_path = 'model/animalff.keras'
# if os.path.isfile(model_path):
#     # Load the model
#     model = tf.keras.models.load_model(model_path)
# else:
#     print(f"Model file does not exist at {model_path}")

# # Ensure the directory exists
# UPLOAD_FOLDER = 'static/uploads'
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Load and preprocess the image
#     file = request.files['file']
#     filename = file.filename
#     filepath = os.path.join(UPLOAD_FOLDER, filename)
    
#     file.save(filepath)  # Save the file

#     img = Image.open(filepath)
#     img = img.resize((64, 64))  # Ensure the image is resized to the same size as the training images
#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
    
#     # Predict
#     predictions = model.predict(img_array)
#     predicted_class = np.argmax(predictions[0])
    
#     return jsonify({'predicted_class': int(predicted_class)})
    
# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
