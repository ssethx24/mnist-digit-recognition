from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["PATH"] += os.pathsep + "C:\\Program Files\\Graphviz\\bin" # to print the architecture diagram amd
# set the environment path for Graphvitz if not set


# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Add a channel dimension (required by CNNs)
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Visualize the first 10 images with their labels
plt.figure(figsize=(10, 1))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(train_images[i].reshape(28, 28), cmap='gray')
    plt.title(train_labels[i])
    plt.axis('off')
plt.tight_layout()
plt.savefig("mnist_sample_images.png")  # Save the visualization
plt.show()

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 output classes for digits 0-9
])

# Print model summary and save architecture diagram
model.summary()
plot_model(model, to_file="model_architecture.png", show_shapes=True, show_layer_names=True)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=5,
                    validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test accuracy: {test_acc}")

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.savefig("accuracy_graph.png")  # Save accuracy graph
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig("loss_graph.png")  # Save loss graph
plt.show()

# Visualize predictions for the first 10 test images
plt.figure(figsize=(10, 1))
for i in range(10):
    test_image = test_images[i].reshape(1, 28, 28, 1)
    prediction = model.predict(test_image)
    predicted_label = np.argmax(prediction)

    plt.subplot(1, 10, i + 1)
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"{predicted_label}", fontsize=8)
    plt.axis('off')

plt.suptitle("Predicted Labels for First 10 Test Images", y=0.9)
plt.tight_layout()
plt.savefig("predictions.png")  # Save predictions visualization
plt.show()
