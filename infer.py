import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf

# Initialize label and time step settings
label = "Warmup...."
n_time_steps = 50
lm_list = []

# Initialize MediaPipe Pose model
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Load the trained multi-class LSTM model
model = tf.keras.models.load_model("exercise_lstm_model.h5")

# Define labels for multiple exercises (6 exercises in this case)
exercise_labels = ["1", "2", "3", "4", "5", "6"]

# Start video capture
cap = cv2.VideoCapture(0)


# Function to create landmarks for each frame (to be used as input for the LSTM model)
def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


# Function to draw pose landmarks on the image
def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return img


# Function to draw predicted class on the image
def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img


# Modified detect function for multi-class classification
def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    print(lm_list.shape)

    # Predict the exercise
    results = model.predict(lm_list)
    print(results)

    # Get the class with the highest probability using argmax
    predicted_class = np.argmax(results)

    # Update label based on the predicted class
    label = exercise_labels[predicted_class]
    return label


# Initialize variables for detection
i = 0
warmup_frames = 60

# Start real-time video feed and pose detection
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    i += 1

    if i > warmup_frames:
        print("Start detect....")

        if results.pose_landmarks:
            c_lm = make_landmark_timestep(results)
            lm_list.append(c_lm)

            # Check if we have collected enough frames (n_time_steps) to make a prediction
            if len(lm_list) == n_time_steps:
                # Perform prediction in a separate thread
                t1 = threading.Thread(target=detect, args=(model, lm_list,))
                t1.start()
                lm_list = []  # Reset landmark list for next sequence

            # Draw pose landmarks on the image
            img = draw_landmark_on_image(mpDraw, results, img)

    # Draw the predicted class label on the image
    img = draw_class_on_image(label, img)

    # Show the result in a window
    cv2.imshow("Image", img)

    # Exit if the user presses 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
