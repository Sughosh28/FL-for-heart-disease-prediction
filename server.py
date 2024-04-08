import traceback
import flwr as fl
import numpy as np
from flask import Flask, jsonify, render_template, request
import subprocess
import threading
from flask_cors import CORS
import tensorflow as tf  # Import TensorFlow if not imported already
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

app = Flask(__name__)
CORS(app)

# Status variable to track federated learning completion
fl_status = {'completed': False}
federated_learning_completed = False

def create_default_model():
    # Create a simple default model
    model = Sequential()
    
    model.add(Dense(64, input_dim=13, activation='relu'))  # Adjust input_dim based on your feature dimensions
    model.add(Dense(1, activation='softmax'))  # Assuming binary classification

    # Compile the model with an appropriate loss and optimizer
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

aggregated_model = create_default_model()
# Flower server configuration
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd,
        results,
        failures
    ):
        try:
            print(f"Server - Starting aggregation for round {rnd}...")
            aggregated_weights = super().aggregate_fit(rnd, results, failures)
            if aggregated_weights is not None:
                # Save aggregated_weights
                print(f"Server - Saving round {rnd} aggregated_weights...")
                np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
            print(f"Server - Aggregation complete for round {rnd}!")
            return aggregated_weights
        except Exception as e:
            print(f"Error saving/aggregating weights: {e}")
            return None

# Function to start the Flower server
def run_flower_server():
    server_address = "localhost:5000"
    strategy = SaveModelStrategy()
    server_config = fl.server.ServerConfig(num_rounds=8)
    fl.server.start_server(
        server_address=server_address,
        config=server_config,
        grpc_max_message_length=1024*1024*1024,
        strategy=strategy
    )

# Function to start the Flask server
def run_flask_server():
    app.run(debug=True, port=8000) # Change the port number as needed

# Flask endpoints
@app.route('/')
def index():
    global fl_status

    # Pass fl_status to the Jinja template
    return render_template('index.html', fl_status=fl_status)
    

@app.route('/results')
def results_page():
    return render_template('results.html')

@app.route('/start-federated-learning', methods=['POST'])
def start_federated_learning():
    # Placeholder for starting federated learning
    return jsonify({'message': 'Federated Learning Server Started!'})

@app.route('/trigger-client-execution', methods=['POST'])
def trigger_client_execution():
    # Placeholder for triggering client execution
    subprocess.Popen(["python", "client_2.py"])
    subprocess.Popen(["python", "client_3.py"])
    subprocess.Popen(["python", "client_1.py"])
    # Federated Learning is completed, update the completion status
    global federated_learning_completed 
    federated_learning_completed = True
    return jsonify({'message': 'Client Execution Triggered!'})


@app.route('/check-federated-learning-status', methods=['GET'])
def check_federated_learning_status():
    global federated_learning_completed
    federated_learning_completed=True
    return jsonify({'completed': federated_learning_completed})


@app.route('/api/predict', methods=['POST'])
def predict():
    print("Received a POST request to /api/predict")

    try:
        # Get input data from the request
        data = request.get_json()


        # Extract features (age, sex, cp, etc.) from the input data
        age = float(data.get('age'))if data.get('age') is not None else 0.0
        sex = float(data.get('sex'))if data.get('sex') is not None else 0.0
        cp = float(data.get('cp'))if data.get('cp') is not None else 0.0
        trestbps = float(data.get('trestbps'))if data.get('trestbps') is not None else 0.0
        chol = float(data.get('chol'))if data.get('chol') is not None else 0.0
        fbs = float(data.get('fbs'))if data.get('fbs') is not None else 0.0
        restecg = float(data.get('restecg'))if data.get('restecg') is not None else 0.0
        thalach = float(data.get('thalach'))if data.get('thalach') is not None else 0.0
        exang = float(data.get('exang'))if data.get('exang') is not None else 0.0
        oldpeak = float(data.get('oldpeak'))if data.get('oldpeak') is not None else 0.0
        slope = float(data.get('slope'))if data.get('slope') is not None else 0.0
        ca = float(data.get('ca'))if data.get('ca') is not None else 0.0
        thal = float(data.get('thal'))if data.get('thal') is not None else 0.0

        input_data = np.array([[age, sex, cp, trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])  # Adjust with all features
        print("Input data:", input_data)

        # Get the model from Flower server
        global aggregated_model
        print("Aggregated Model:", aggregated_model)

        # Check if the aggregated model is available
        if aggregated_model is not None:
            # Use the aggregated model to make predictions
            predictions = aggregated_model.predict(input_data)

            # Assuming binary classification with a threshold of 0.5
            threshold = 0.5
            binary_prediction = (predictions > threshold).astype(int)

            # Return the prediction in JSON format
            return jsonify({'prediction': binary_prediction.tolist()})
        else:
            return jsonify({'error': 'Failed to get aggregated model from federated server'}), 500

    except Exception as e:
        print(f"Error predicting: {str(e)}")
        traceback.print_exc()  # Add this line to print the full traceback

        return jsonify({'error': 'Failed to make prediction'}), 500
# Main function to start both Flower and Flask servers
def main():
    # Start Flower server in a separate thread
    flower_thread = threading.Thread(target=run_flower_server)
    flower_thread.start()

    # Start Flask server in the main thread
    run_flask_server()

if __name__ == "__main__":
    main()
