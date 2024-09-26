import os
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential

# Path to your data directory
data_dir = 'C:\\Users\\223162D\\PycharmProjects\\Tensor\\exercise_data'
n_time_steps = 50  # Number of timesteps for each sequence
n_features = 132  # Assuming 133 features (x, y, z, and visibility for multiple landmarks)
n_classes = 6  # Number of exercise classes


# Function to load and preprocess data
def load_exercise_data(folder):
    X = []
    y = []

    # Iterate over each exercise folder (each exercise is a class)
    for label, exercise in enumerate(os.listdir(folder)):
        exercise_folder = os.path.join(folder, exercise)

        for file in os.listdir(exercise_folder):
            file_path = os.path.join(exercise_folder, file)
            data = pd.read_csv(file_path)
            sequence = data.values  # Convert DataFrame to numpy array

            # Ensure that the sequence is padded/truncated to the required time steps
            if len(sequence) >= n_time_steps:
                X.append(sequence[:n_time_steps])  # Truncate if longer
            else:
                X.append(pad_sequences([sequence], maxlen=n_time_steps, padding='post')[0])  # Pad if shorter

            y.append(label)  # Label corresponds to the exercise type (0 to n_classes-1)

    return np.array(X), np.array(y)


# Load the data
X, y = load_exercise_data(data_dir)

# One-hot encode labels for categorical classification
y = to_categorical(y, num_classes=n_classes)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Building the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(n_time_steps, n_features)))  # Input shape based on X_train
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=n_classes, activation="softmax"))  # Output for multi-class classification

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_test, y_test))

# Save the model
model.save("exercise_lstm_model.h5")
