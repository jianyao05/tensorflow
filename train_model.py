import numpy as np
import pandas as pd

from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Read data for all exercises
exercise1_df = pd.read_csv("EXERCISE1.txt")
exercise2_df = pd.read_csv("EXERCISE2.txt")
exercise3_df = pd.read_csv("EXERCISE3.txt")
exercise4_df = pd.read_csv("EXERCISE4.txt")
exercise5_df = pd.read_csv("EXERCISE5.txt")
exercise6_df = pd.read_csv("EXERCISE6.txt")
exercise7_df = pd.read_csv("EXERCISE7.txt")
exercise8_df = pd.read_csv("EXERCISE8.txt")
exercise9_df = pd.read_csv("EXERCISE9.txt")
exercise10_df = pd.read_csv("EXERCISE10.txt")
exercise11_df = pd.read_csv("EXERCISE11.txt")
exercise12_df = pd.read_csv("EXERCISE12.txt")




X = []
y = []
no_of_timesteps = 10

# Process data for each exercise
datasets = [
    (exercise1_df, 0),  # Label 0 for exercise 1
    (exercise2_df, 1),  # Label 1 for exercise 2 right lateral pull
    (exercise3_df, 2),  # Label 2 for exercise 3
    (exercise4_df, 3),  # Label 3 for exercise 4 right arm circle
    (exercise5_df, 4),  # Label 4 for exercise 5
    (exercise6_df, 5),   # Label 5 for exercise 6 right pendulum
    (exercise7_df, 6),   # Label 5 for exercise 7
    (exercise8_df, 7),  # Label 4 for exercise 8 right towel strewtch
    (exercise9_df, 8),   # Label 5 for exercise 9
    (exercise10_df, 9),  # Label 4 for exercise 10 right cross body stretch
    (exercise11_df, 10),   # Label 5 for exercise 11 armpiut squat
    (exercise12_df, 11)  # Label 5 for exercise 11 armpiut squat
]

for df, label in datasets:
    dataset = df.iloc[:, 1:].values
    n_sample = len(dataset)
    for i in range(no_of_timesteps, n_sample):
        X.append(dataset[i-no_of_timesteps:i, :])
        y.append(label)

# Convert to numpy arrays
X, y = np.array(X), np.array(y)

# One-hot encode the labels (for multi-class classification)
y = to_categorical(y, num_classes=12)

print(X.shape, y.shape)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=12, activation="softmax"))  # 6 units for 6 classes, softmax activation for multi-class

# Compile the model
model.compile(optimizer="adam", metrics=['accuracy'], loss="categorical_crossentropy")

# Train the model
model.fit(X_train, y_train, epochs=16, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save("exercise_model.h5")
