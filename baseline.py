"""
Epoch: 1 Loss: 0.21344085 Accuracy: 0.9338667
Epoch: 2 Loss: 0.09374509 Accuracy: 0.97108334
Epoch: 3 Loss: 0.068478525 Accuracy: 0.97945
Loss: 0.0838993 Accuracy: 0.9768
"""

import tensorflow as tf
import numpy as np

# Load MNIST dataset, get x_train.shape = (60000,28,28), x_train.dtype = uint8, pixel: 0~255
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()

# Convert pixel value from 0~255 to 0~1. 
# Normalization. The input scale influence the gradient, if the scale is large, the change of parameters is more rapid
x_train = x_train.astype(np.float32)/255.0
x_test = x_test.astype(np.float32)/255.0

# Flatten layer and flat 28*28 images into vectors
flatten = tf.keras.layers.Flatten()
x_train = flatten(x_train)
x_test = flatten(x_test)

y_train = tf.cast(y_train,tf.float32)
y_test = tf.cast(y_test,tf.float32)

# Define batch size
batch_size = 32

# Create tensorflow datasets
train_datasets = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)
).shuffle(len(x_train)).batch(batch_size) # len(x_train) only use the first shape of x_train 60000

test_datasets = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)
).batch(batch_size)

# Initializer and parameters
initializer = tf.keras.initializers.HeNormal(seed=1)

w1 = tf.Variable(initializer(shape=(784,392)))
b1 = tf.Variable(tf.zeros((392,)))

w2 = tf.Variable(initializer(shape=(392,196)))
b2 = tf.Variable(tf.zeros((196,)))

w3 = tf.Variable(initializer(shape=(196,98)))
b3 = tf.Variable(tf.zeros((98,)))

w4 = tf.Variable(initializer(shape=(98,49)))
b4 = tf.Variable(tf.zeros((49,)))

w5 = tf.Variable(initializer(shape=(49,10)))
b5 = tf.Variable(tf.zeros((10,)))

# Forward propogation
def forward(x):
    z1 = tf.nn.relu(
        tf.matmul(x,w1)+b1
    )
    
    z2 = tf.nn.relu(
        tf.matmul(z1,w2)+b2
    )
    
    z3 = tf.nn.relu(
        tf.matmul(z2,w3)+b3
    )
    
    z4 = tf.nn.relu(
        tf.matmul(z3,w4)+b4
    )
    
    y_pre = tf.nn.softmax(
        tf.matmul(z4,w5)+b5
    )
    
    return y_pre
    
# Loss function
# Use SpareCategoricalCrossentropy since there are more than two outcomes
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Training
epochs = 3

training_variables = [
    w1,w2,w3,w4,w5,
    b1,b2,b3,b4,b5
]

for epoch in range(epochs):
    total_loss = 0.0
    correct = 0
    total = 0
    
    for x_batch, y_batch in train_datasets:
        with tf.GradientTape() as tape:
            y_pre = forward(x_batch)
            loss = loss_fn(
                y_batch,
                y_pre
            )
            
        gradients = tape.gradient(
            loss,
            training_variables
        )
        
        optimizer.apply_gradients(
            zip(gradients,training_variables)
        )
        
        pred = tf.cast(
            tf.argmax(y_pre,axis=1),
            tf.float32
        )
        correct += tf.reduce_sum(
            tf.cast(pred == y_batch,tf.float32)
        )
        total+=tf.size(y_batch)
        total_loss+=loss
        
    accuracy = correct/tf.cast(total,tf.float32)
        
    print(
        "Epoch:",epoch + 1,
        "Loss:",total_loss.numpy() / len(train_datasets),
        "Accuracy:",accuracy.numpy()
    )
    
# Test model

test_loss = 0.0
correct = 0
total = 0

for x_batch, y_batch in test_datasets:
    
    y_pre = forward(x_batch)
    loss = loss_fn(
        y_batch,
        y_pre
    )
        
    pred = tf.cast(
        tf.argmax(y_pre,axis=1),
        tf.float32
    )
    correct += tf.reduce_sum(
        tf.cast(pred == y_batch,tf.float32)
    )
    total+=tf.size(y_batch)
    test_loss+=loss
        
test_accuracy = correct/tf.cast(total,tf.float32)
        
print(
    "Loss:",test_loss.numpy() / len(test_datasets),
    "Accuracy:",test_accuracy.numpy()
)
