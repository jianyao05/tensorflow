import numpy as np
import pandas as pd

from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical  # For one-hot encoding the labels

# Load data (Assuming more datasets for additional exercises are available)
bodyswing_df = pd.read_csv("EXERCISE1.txt")
handswing_df = pd.read_csv("EXERCISE3.txt")
pushup_df = pd.read_csv("EXERCISE5.txt")  # New exercise data
squat_df = pd.read_csv("EXERCISE7.txt")    # New exercise data
jumpjack_df = pd.read_csv("EXERCISE9.txt")  # New exercise data
lunge_df = pd.read_csv("EXERCISE11.txt")    # New exercise data


X = []
y = []
no_of_timesteps = 10

# Function to append dataset with a label
def append_data(dataset, label):
    n_sample = len(dataset)
    for i in range(no_of_timesteps, n_sample):
        X.append(dataset[i-no_of_timesteps:i, :])
        y.append(label)

# Processing datasets for each exercise and adding corresponding labels
append_data(bodyswing_df.iloc[:, 1:].values, 0)  # SWING HAND label = 0
append_data(handswing_df.iloc[:, 1:].values, 1)  # SWING BODY label = 1
append_data(pushup_df.iloc[:, 1:].values, 2)     # PUSH UP label = 2
append_data(squat_df.iloc[:, 1:].values, 3)      # SQUAT label = 3
append_data(jumpjack_df.iloc[:, 1:].values, 4)   # JUMPING JACK label = 4
append_data(lunge_df.iloc[:, 1:].values, 5)      # LUNGE label = 5

# Convert X and y to numpy arrays
X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

# One-hot encoding of labels for categorical classification
y = to_categorical(y, num_classes=6)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Building the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=6, activation="softmax"))  # Output 6 units for 6 classes (softmax)

# Compiling the model with categorical cross-entropy for multi-class classification
model.compile(optimizer="adam", metrics=['accuracy'], loss="categorical_crossentropy")

# Training the model
model.fit(X_train, y_train, epochs=16, batch_size=32, validation_data=(X_test, y_test))

# Saving the trained model
model.save("model_multi_exercises.h5")
