import cv2
import mediapipe as mp
import pandas as pd
import os
import time

# Read from webcam
cap = cv2.VideoCapture(0)

# Initialize mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Parameters
label = "JUMP"  # Modify this label for each exercise
n_repetitions = 60  # Number of repetitions
n_time_steps_per_rep = 50  # Number of frames per repetition (increased for longer capture)
save_folder = f"./exercise_data/{label}/"  # Folder to save repetitions
lm_list = []  # To store landmarks for one repetition
rep_count = 0  # Track the repetition count
frame_count = 0  # Track the frame count per repetition
break_time = 0  # Break time between repetitions in seconds

# Create folder if it doesn't exist
if not os.path.exists(save_folder):
    os.makedirs(save_folder)


def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


def draw_landmark_on_image(mpDraw, results, img):
    # Draw landmarks and connections
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    # Draw nodes
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
    return img


while rep_count < n_repetitions:
    # Prompt indicating start of capturing a repetition
    print(f"Starting repetition {rep_count + 1} of {n_repetitions}...")
    time.sleep(1)  # Brief pause before starting

    while frame_count < n_time_steps_per_rep:
        ret, frame = cap.read()
        if ret:
            # Process frame for pose detection
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frameRGB)

            if results.pose_landmarks:
                lm = make_landmark_timestep(results)
                lm_list.append(lm)
                frame_count += 1

                # Draw the pose landmarks on the image
                frame = draw_landmark_on_image(mpDraw, results, frame)

            # Display the frame with pose landmarks
            cv2.imshow("image", frame)

            if cv2.waitKey(1) == ord('q'):
                break

    # Save the repetition as a CSV file
    df = pd.DataFrame(lm_list)
    df.to_csv(f"{save_folder}/{label}_rep{rep_count + 1}.csv", index=False)
    print(f"Saved repetition {rep_count + 1} to {save_folder}")

    # Reset for the next repetition
    lm_list = []
    frame_count = 0
    rep_count += 1  # Increment the repetition count

    # Prompt for a break between repetitions
    if rep_count < n_repetitions:
        print(f"Take a {break_time} second break before starting the next repetition.")
        time.sleep(break_time)  # Wait for the specified break time

# Final prompt indicating end of all repetitions
print(f"All {n_repetitions} repetitions completed.")

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
