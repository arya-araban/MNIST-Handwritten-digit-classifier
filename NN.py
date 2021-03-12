import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0
# summarize loaded dataset
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

model = tf.keras.models.Sequential([  # the sequential model we use for training
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # flattening the 28*28 first layer into a 784 input layer one
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)  # this is for output -- we later specify it as SOFTMAX
])
predictions = model(x_train[:1]).numpy()  # we take the first example as a representation
tf.nn.softmax(
    predictions).numpy()  # gives the output for first example -- note the weights have been randomly  initalized here

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5)  # do the training here, save info in history
model.save('my_model.h5')
new_model = tf.keras.models.load_model('my_model.h5')
plt.plot(history.history['loss'], label='MAE (training data)')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()
print(new_model.evaluate(x_test, y_test, verbose=2))
