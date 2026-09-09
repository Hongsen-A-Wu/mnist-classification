# MNIST Handwritten Digit Classification

A classification project built with TensorFlow using the MNIST dataset.

The model is a fully connected neural network implemented with TensorFlow operations and trainable variables. Each `28 × 28` grayscale image is flattened into a vector and passed through several dense layers with ReLU activations. The final layer outputs 10 logits corresponding to digits `0` through `9`.

The model uses `SparseCategoricalCrossentropy(from_logits=True)` as the loss function, so the output logits are passed directly to the loss without applying softmax first. Predictions are obtained using `tf.argmax()` to select the class with the highest output score.

Training is performed using mini-batches, `tf.GradientTape()` for automatic differentiation, and the Adam optimizer for updating model parameters.

## Results

After 4 epochs:

* Training Loss: 0.0542
* Training Accuracy: 98.3%
* Test Loss: 0.0701
* Test Accuracy: 97.9%

## Technologies

* Python
* TensorFlow
* Numpy
* MNIST Dataset

## Future Improvements

Possible improvements include using convolutional neural networks, dropout, batch normalization, and learning rate scheduling to improve image feature extraction and generalization.
