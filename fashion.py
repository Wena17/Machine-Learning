import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(256, activation = "relu"),
    keras.layers.Dense(len(class_names), activation="softmax")
])

model.summary()

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=10)

model.save("fashion.h5")

# model = keras.models.load_model("fashion.h5")

prediction = model.predict(test_images)

print("\nRunning tests")
text_loss, test_acc = model.evaluate(test_images, test_labels)

for i in range(len(test_labels)):
    if np.argmax(prediction[i]) != test_labels[i]:
        plt.grid(False)
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        plt.xlabel("Actual: {}".format(class_names[test_labels[i]]))
        plt.title("Prediction: {}".format(class_names[np.argmax(prediction[i])]))
        plt.show()
