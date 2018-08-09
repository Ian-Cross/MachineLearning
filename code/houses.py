import tensorflow as tf

# Input, output & params
y = tf.placeholder(tf.float32)
m = tf.Variable([1.0], tf.float32)
b = tf.Variable([1.0], tf.float32)
x = tf.placeholder(tf.float32)

# Model
prediction = (10 + m) * x + b

# Loss
loss = tf.reduce_sum(tf.square(prediction - y))

# Optimize
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# Training Data
bedrooms = [0, 1, 2, 3, 4]
price = [240, 255, 270, 285, 300]

# Session to run our code
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Train our model 1000 times on training data
    for i in range(1000):
        sess.run(train, {x: bedrooms, y: price})

    # Print trained values for m and b
    val_m, val_b = sess.run([m, b])
    print("Value of m is %s and value of b is %s." % (val_m, val_b))