# Real-Time Hand Gesture Recognition Using Computer Vision

## 1. Project Overview

This project is a real-time hand gesture recognition system developed using Computer Vision and Machine Learning.

The system captures video from a webcam, detects hand landmarks using MediaPipe, extracts landmark coordinates as features, and uses a Random Forest classifier to recognize predefined hand gestures.

### Supported Gestures

* FIST
* ONE
* TWO
* THREE
* OPEN_HAND

## 2. Technologies Used

* Python 3.11
* OpenCV
* MediaPipe
* NumPy
* Pandas
* Scikit-learn
* Random Forest

## 3. Project Pipeline

```text
Webcam
   ↓
OpenCV
   ↓
MediaPipe Hand Landmarker
   ↓
21 Hand Landmarks
   ↓
Feature Extraction
   ↓
Random Forest Classifier
   ↓
Gesture Prediction
```

## 4. Project Structure

```text
HandGestureRecognition/
│
├── README.md
├── requirements.txt
├── main.py
├── collect_data.py
├── train_model.py
├── predict.py
├── gesture_data.csv
├── gesture_model.pkl
├── hand_landmarker.task
└── .gitignore
```

## 5. Requirements

* Python 3.11 or compatible Python version
* Working webcam
* Internet connection for installing Python dependencies

## 6. Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/Harsh-Yadav-E/HandGestureRecognition.git
```

Move into the project directory:

```bash
cd HandGestureRecognition
```

### Step 2: Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

## 7. Running the Project

The trained model is included in the repository.

Run the real-time recognition system:

```bash
python predict.py
```

A webcam window will open.

Show your hand to the camera and the system will display the predicted gesture and prediction confidence.

Press `Q` to exit.

## 8. Training the Model

The project also includes the scripts required to create and train the gesture recognition model.

### Collect Data

Run:

```bash
python collect_data.py
```

Enter a gesture name when prompted.

Press `SPACE` to save a hand landmark sample.

Press `Q` to stop collecting data.

The collected data is stored in:

```text
gesture_data.csv
```

### Train the Model

After collecting the required gesture samples, run:

```bash
python train_model.py
```

The script trains a Random Forest classifier and saves the trained model as:

```text
gesture_model.pkl
```

## 9. Gesture Recognition

The trained model uses the 21 hand landmarks detected by MediaPipe.

Each landmark provides an X and Y coordinate, resulting in:

```text
21 landmarks × 2 coordinates = 42 features
```

These features are provided to the Random Forest classifier, which predicts the gesture.

## 10. Machine Learning Model

The project uses a Random Forest classifier.

The dataset is divided into training and testing subsets using an 80/20 split.

The model's accuracy is calculated on the test dataset using Scikit-learn.

## 11. Usage

1. Start the application using `predict.py`.
2. Position your hand in front of the webcam.
3. Make one of the supported gestures.
4. The predicted gesture is displayed on the screen.
5. The model confidence is also displayed.
6. Press `Q` to exit.

## 12. Future Improvements

Possible future improvements include:

* Adding more gestures
* Supporting two-hand gestures
* Improving feature normalization
* Increasing the training dataset
* Using additional machine learning models
* Adding gesture-controlled applications
* Improving prediction stability
* Adding a graphical user interface

