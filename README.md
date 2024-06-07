Federated learning, a decentralized
model training paradigm, is implemented using the
Flower framework in this research. The study
focuses on predicting heart disease collaboratively
across multiple clients while preserving data
privacy. A custom neural network model is trained
on clients, and the federated learning strategy of
Flower facilitates model updates aggregation. The
experiments showcase the effectiveness of
collaborative model training, emphasizing data
privacy preservation and reduced communication
overhead. The results contribute to the growing
field of federated learning applications and
highlight the unique features of the Flower
framework.

Prerequisites
Before running this project, ensure you have the following dependencies installed:

Flower
TensorFlow
Pandas
Keras


How to Run
Server Setup: Open a terminal and run the following command to start the server:

python server.py 5000
Replace 5000 with your desired port number if necessary.

Client Setup: Open new terminals for each client you want to run. In each terminal, run the following command to start a client:
python client.py 5000
Again, replace 5000 with the same port number used for the server if you changed it.

Accessing the Interface: Once the server and clients are connected successfully, open a web browser and navigate to http://localhost:8000. You will see the user interface.

Running Federated Learning (FL):

On the interface, there will be a button to start FL. Click on it to initiate the federated learning process.
After the FL completion, you can click on the "Check Results" button.
Form Submission:

Fill out the form with the required details.
Submit the form to get the results.
