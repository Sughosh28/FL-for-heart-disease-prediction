import flwr as fl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Function to load your custom CSV data
def load_custom_csv_data():
    # Load your CSV data
    # Replace 'path/to/your/data.csv' with the actual path to your CSV file
    data = pd.read_csv('C:\\Users\\HP\\Desktop\\Trial-css\\heart_disease_data.csv')

    # Split the data into features (X) and labels (y)
    X = data.drop(columns=['target']).to_numpy()
    y = data['target'].to_numpy()

    return X, y

# Define a simple model creation function
def create_your_model(input_dim):
    # Replace this with your own model creation code
    # This is just a placeholder example
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(2, activation='softmax'))  # Assuming binary classification

    # Compile the model with an appropriate loss and optimizer
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Define a class that inherits from fl.client.NumPyClient
from typing import Tuple, Dict

class YourCustomClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test, address):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.address = address

    def get_parameters(self, config):
        return self.model.get_weights()
    
    def predict(self, input_data):
        """Make predictions using the trained model."""
    # Ensure the input_data is in the same format as during training
        input_data = np.asarray(input_data).astype(np.float32)

    # Make predictions using the trained model
        predictions = self.model.predict(input_data)

    # Assuming binary classification with a threshold of 0.5
        threshold = 0.5
        binary_prediction = (predictions > threshold).astype(int)
        return binary_prediction      

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        print(f"Client {self.address} - Length of X_train: {len(self.X_train)}, Length of y_train: {len(self.y_train)}")
        print(f"Client {self.address} - Setting parameters...")
        self.set_parameters(parameters)
        # Replace this with your actual training logic
        print(f"Client {self.address} - Training on local data...")
        history = self.model.fit(self.X_train, self.y_train, epochs=4)
        # Get the number of examples processed during training
        num_examples_processed = len(self.X_train)
        # Return the updated model parameters, the number of examples processed, and additional metrics
        updated_params = self.get_parameters(config=config)
        metrics = {"accuracy": history.history.get("accuracy", [0])[-1]}  # Add your metrics here
        print(f"Client {self.address} - Training complete. Updated model parameters:")
        print(updated_params)
        print(f"Number of examples processed: {num_examples_processed}")
        print(f"Metrics: {metrics}")
        # Make predictions on test data
        test_predictions = self.predict(self.X_test)
        threshold = 0.5
        binary_test_prediction = (test_predictions[:, 1] > threshold).astype(int)
        # Print or use the binary test prediction as needed
        print(f"Client {self.address} - Binary Prediction on test data:")
        print(binary_test_prediction)
        return updated_params, num_examples_processed, metrics
    
    def evaluate(self, parameters, config):
        print(f"Client {self.address} - Length of X_test: {len(self.X_test)}, Length of y_test: {len(self.y_test)}")
        print(f"Client {self.address} - Setting parameters for evaluation...")
        self.set_parameters(parameters)
        # Replace this with your actual evaluation logic
        print(f"Client {self.address} - Evaluating on test data...")
        evaluation_metrics = self.model.evaluate(self.X_test, self.y_test)
        # Get the number of examples processed during evaluation
        num_examples_processed = len(self.X_test)
        # Return the evaluation metrics, the number of examples processed, and additional metrics (if any)
        accuracy_position = 1  # Adjust accordingly based on your model's evaluate output
        additional_metrics = {"accuracy": evaluation_metrics[accuracy_position]}
        return evaluation_metrics[0], num_examples_processed, additional_metrics



# Specify the server address
server_address = "localhost:5000"

# Load your custom CSV data
# Load your custom CSV data
X, y = load_custom_csv_data()

# Print the shape of the entire dataset
print("X shape:", X.shape)
print("y shape:", y.shape)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shape of the training data
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


# Create your model
model = create_your_model(input_dim=X_train.shape[1])

# Create an instance of YourCustomClient
# Create an instance of YourCustomClient
client = YourCustomClient(model, X_train, y_train, X_test, y_test, address="localhost:5000")

# Print the length of the training data in the client
# print(f"Client {client.address} - Length of X_train: {len(client.X)}, Length of y_train: {len(client.y)}")

print(f"Connecting to Flower server at {server_address}...")
print("Sending updated model to Flower server...")
fl.client.start_client(
    server_address=server_address,
    client=client.to_client(),
)
print("Model sent successfully.")

print("Connection successful. Exiting.")
