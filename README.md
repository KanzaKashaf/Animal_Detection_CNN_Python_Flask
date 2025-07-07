
# Animal Detection System

A deep learning-based image classification system that detects and classifies animals into **cats**, **dogs**, or **squirrels**. The system uses Convolutional Neural Networks (CNN) for training on image datasets and is deployed using a Flask web interface.

---

## Project Overview

This project was developed as part of the **Artificial Intelligence** coursework at **National Textile University, Faisalabad**.

The model is trained on over 6000+ images and tested on 1200+ images using the following datasets:

- Animals-10 Dataset https://www.kaggle.com/datasets/alessiocorrado99/animals10

---

## Group Members

- **Kanza Kashaf**  
- **Muhammad Hassaan Raza**

---

## Model Performance

| Dataset        | Accuracy | Loss  |
|----------------|----------|-------|
| Training Set   | 93%      | 0.19  |
| Testing Set    | 75%      | 0.88  |

---

## Technologies Used

- Python
- TensorFlow & Keras
- NumPy
- Flask (for Web Interface)
- ImageDataGenerator (for data augmentation)

---

## Features

- Image classification of cats, dogs, and squirrels.
- Web interface using Flask for real-time image uploads and prediction.
- Data preprocessing with augmentation for improved generalization.
- CNN model with dropout to reduce overfitting.
- Class prediction with probability scores.

---

## Model Architecture

- **Conv2D** layers for feature extraction
- **MaxPooling2D** for dimensionality reduction
- **Dropout** for regularization
- **Dense** layers for classification
- **Softmax** activation for multi-class output

---

## Training Process

```python
cnn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.fit(x=training_set, validation_data=test_set, epochs=120)
```

---

## Project Structure

```
Animal_Detection_CNN_Python_Flask
├── App/
│   └── app.py                 # Flask app
│   └── model.h5               # Trained model
│   └── static/
│       └──  uploads/
│   └── templates/
│       └── index.html
├── Prediction/
├── Documentation
│   └── Documentation
│   └── PPT
├── Animal_Detection_ModelTraining.ipynb
├── Animal_Detection_Trained_Model.keras
├── Dataset.txt
├── README.md
```

---

## Predicting New Images

```python
from keras.preprocessing import image
test_image = image.load_img('Prediction/ccc.jpeg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)

# Prediction logic
if result[0][0] == 1:
    print('cat')
elif result[0][1] == 1:
    print('dog')
elif result[0][2] == 1:
    print('squirrel')
```

---

## Web Interface

The system is integrated with a user-friendly web interface using Flask. It allows users to upload images and receive predictions directly in the browser.

To run:

```bash
python app.py
```

Visit: `http://localhost:5000`

---

## Acknowledgments

- Kaggle for the image datasets  
- TensorFlow/Keras documentation  
- Flask official documentation  

---
