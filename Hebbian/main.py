import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Ensure the data type is float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Define the neural network model
def create_model(input_shape):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

# Initialize model weights with quantum-inspired distribution
def initialize_weights(model):
    for layer in model.layers:
        if isinstance(layer, Dense):
            layer.kernel_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
            layer.bias_initializer = tf.keras.initializers.Zeros()
    model.build(input_shape=(None, *input_shape))

# Generate offspring networks by perturbing weights
def generate_offspring(model, num_offspring, perturbation_std=0.01):
    offspring = []
    for _ in range(num_offspring):
        new_model = clone_model(model)
        new_model.set_weights(model.get_weights())
        for layer in new_model.layers:
            if isinstance(layer, Dense):
                weights, biases = layer.get_weights()
                weights += np.random.randn(*weights.shape) * perturbation_std
                biases += np.random.randn(*biases.shape) * perturbation_std
                layer.set_weights([weights, biases])
        offspring.append(new_model)
    return offspring

# Evaluate offspring networks
def evaluate_offspring(offspring, x_data, y_data):
    losses = []
    for model in offspring:
        model.compile(optimizer=SGD(learning_rate=learning_rate), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
        loss = model.evaluate(x_data, y_data, verbose=0)[0]  # Only get the loss value
        losses.append(loss)
    return losses

# Select top-performing offspring
def select_top_offspring(offspring, losses, top_k=1):
    # Zip losses and offspring, sort by losses, and then extract the top_k offspring
    sorted_indices = np.argsort(losses)
    top_offspring = [offspring[i] for i in sorted_indices[:top_k]]
    return top_offspring

# Hebbian learning rule for weight update
def hebbian_update(model, x_data, y_data, learning_rate=0.01):
    with tf.GradientTape() as tape:
        predictions = model(x_data)
        loss = SparseCategoricalCrossentropy()(y_data, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    for i, var in enumerate(model.trainable_variables):
        if 'kernel' in var.name:
            var.assign_sub(gradients[i] * learning_rate)

# Apply sparse updates to weights
def apply_sparse_updates(model, sparsity_factor=0.1):
    for layer in model.layers:
        if isinstance(layer, Dense):
            weights, biases = layer.get_weights()
            mask = (np.random.rand(*weights.shape) < sparsity_factor).astype(np.float32)
            weights *= mask
            layer.set_weights([weights, biases])

# Hyperparameters
input_shape = (28, 28)
num_offspring = 10
num_epochs = 20
learning_rate = 0.01
sparsity_factor = 0.1

# Initialize model
model = create_model(input_shape)
initialize_weights(model)
model.compile(optimizer=SGD(learning_rate=learning_rate), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

# Training loop
for epoch in range(num_epochs):
    for i in range(0, len(x_train), 32):
        x_batch = x_train[i:i+32]
        y_batch = y_train[i:i+32]
        
        # Generate and evaluate offspring
        offspring = generate_offspring(model, num_offspring)
        losses = evaluate_offspring(offspring, x_batch, y_batch)
        
        # Select top-performing offspring
        top_offspring = select_top_offspring(offspring, losses, top_k=1)
        model.set_weights(top_offspring[0].get_weights())
        
        # Apply Hebbian learning
        hebbian_update(model, x_batch, y_batch, learning_rate)
        
        # Apply sparse weight updates
        apply_sparse_updates(model, sparsity_factor)
        
    # Evaluate on validation data
    val_loss, val_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Epoch {epoch+1}/{num_epochs} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

# pip install qiskit
# import numpy as np
# import tensorflow as tf
# from qiskit import QuantumCircuit, Aer, transpile, assemble
# from qiskit.providers.aer import AerSimulator

# class NEQP:
#     def __init__(self, n_qubits, n_layers):
#         self.n_qubits = n_qubits
#         self.n_layers = n_layers
#         self.quantum_circuit = QuantumCircuit(n_qubits)
#         self.simulator = AerSimulator()

#     def add_layer(self, theta):
#         for i in range(self.n_qubits):
#             self.quantum_circuit.rx(theta[i], i)
#         for i in range(self.n_qubits - 1):
#             self.quantum_circuit.cx(i, i + 1)

#     def build_circuit(self, thetas):
#         for i in range(self.n_layers):
#             self.add_layer(thetas[i])
#         self.quantum_circuit.measure_all()

#     def simulate(self, thetas):
#         self.build_circuit(thetas)
#         compiled_circuit = transpile(self.quantum_circuit, self.simulator)
#         qobj = assemble(compiled_circuit)
#         result = self.simulator.run(qobj).result()
#         counts = result.get_counts()
#         return counts

#     def evolve_thetas(self, thetas, fitness):
#         new_thetas = thetas + np.random.randn(*thetas.shape) * fitness
#         return new_thetas

#     def train(self, epochs, initial_thetas, data, labels):
#         thetas = initial_thetas
#         for epoch in range(epochs):
#             fitness = 0
#             for i in range(len(data)):
#                 counts = self.simulate(thetas)
#                 fitness += self.calculate_fitness(counts, labels[i])
#             fitness /= len(data)
#             thetas = self.evolve_thetas(thetas, fitness)
#             print(f"Epoch {epoch + 1}: Fitness = {fitness}")
#         return thetas

#     def calculate_fitness(self, counts, label):
#         fitness = sum(counts.values())  # Simplified fitness function
#         return fitness

# def preprocess_mnist():
#     (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#     x_train, x_test = x_train / 255.0, x_test / 255.0
#     x_train = x_train.reshape(-1, 28, 28, 1)
#     x_test = x_test.reshape(-1, 28, 28, 1)
#     y_train = tf.keras.utils.to_categorical(y_train, 10)
#     y_test = tf.keras.utils.to_categorical(y_test, 10)
#     return (x_train, y_train), (x_test, y_test)

# def build_standard_model():
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Flatten(input_shape=(28, 28)),
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dense(10, activation='softmax')
#     ])
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# def main():
#     n_qubits = 3
#     n_layers = 3
#     epochs = 10
#     initial_thetas = np.random.randn(n_layers, n_qubits)
    
#     # Load and preprocess MNIST data
#     (x_train, y_train), (x_test, y_test) = preprocess_mnist()
    
#     # Train standard model with backpropagation
#     model = build_standard_model()
#     model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
#     loss, acc = model.evaluate(x_test, y_test)
#     print(f"Backpropagation model accuracy: {acc}")

#     # Train NEQP model
#     neqp = NEQP(n_qubits, n_layers)
#     trained_thetas = neqp.train(epochs, initial_thetas, x_train, y_train)
#     print("Trained Parameters:", trained_thetas)

# if __name__ == "__main__":
#     main()
#######################################################
# import numpy as np
# import tensorflow as tf
# from qiskit import QuantumCircuit, Aer, transpile, assemble
# from qiskit.visualization import plot_histogram
# from qiskit.providers.aer import AerSimulator

# class NEQP:
#     def __init__(self, n_qubits, n_layers):
#         self.n_qubits = n_qubits
#         self.n_layers = n_layers
#         self.quantum_circuit = QuantumCircuit(n_qubits)
#         self.simulator = AerSimulator()

#     def add_layer(self, theta):
#         for i in range(self.n_qubits):
#             self.quantum_circuit.rx(theta[i], i)
#         for i in range(self.n_qubits - 1):
#             self.quantum_circuit.cx(i, i + 1)

#     def build_circuit(self, thetas):
#         for i in range(self.n_layers):
#             self.add_layer(thetas[i])
#         self.quantum_circuit.measure_all()

#     def simulate(self, thetas):
#         self.build_circuit(thetas)
#         compiled_circuit = transpile(self.quantum_circuit, self.simulator)
#         qobj = assemble(compiled_circuit)
#         result = self.simulator.run(qobj).result()
#         counts = result.get_counts()
#         return counts

#     def evolve_thetas(self, thetas, fitness):
#         new_thetas = thetas + np.random.randn(*thetas.shape) * fitness
#         return new_thetas

#     def train(self, epochs, initial_thetas):
#         thetas = initial_thetas
#         for epoch in range(epochs):
#             counts = self.simulate(thetas)
#             fitness = self.calculate_fitness(counts)
#             thetas = self.evolve_thetas(thetas, fitness)
#             print(f"Epoch {epoch + 1}: Fitness = {fitness}")
#         return thetas

#     def calculate_fitness(self, counts):
#         fitness = sum(counts.values())
#         return fitness

# if __name__ == "__main__":
#     n_qubits = 3
#     n_layers = 3
#     epochs = 10
#     initial_thetas = np.random.randn(n_layers, n_qubits)

#     neqp = NEQP(n_qubits, n_layers)
#     trained_thetas = neqp.train(epochs, initial_thetas)
#     print("Trained Parameters:", trained_thetas)
