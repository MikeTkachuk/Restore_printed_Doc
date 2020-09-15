import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from data_process import *


@tf.graph_util
def custom_loss(y_true, y_pred):
    loss = tf.square(y_true - y_pred)
    return tf.reduce_mean(loss)


examples = []
for i in range(11):
    examples.append([f"masked_{i}_0.jpg", f"clear_{i}.jpg"])

generate_sample_key = 0

sample_size = 128
if generate_sample_key:
    input_data, output_data = create_samples_tensor(r"C:/Users/Michael/Downloads/Raw_images/",
                                                    examples[:10], sample_size=sample_size)

    input_test, output_test = create_samples_tensor(r"C:/Users/Michael/Downloads/Raw_images/",
                                                    examples[10:], sample_size=sample_size)

    np.save(r"C:\Users\Michael\Downloads\Samples\input_data", input_data.numpy())
    np.save(r"C:\Users\Michael\Downloads\Samples\output_data", output_data.numpy())
    np.save(r"C:\Users\Michael\Downloads\Samples\input_test", input_test.numpy())
    np.save(r"C:\Users\Michael\Downloads\Samples\output_test", output_test.numpy())

else:
    input_data = np.load(r"C:\Users\Michael\Downloads\Samples\input_data.npy")
    output_data = np.load(r"C:\Users\Michael\Downloads\Samples\output_data.npy")
    input_test = np.load(r"C:\Users\Michael\Downloads\Samples\input_test.npy")
    output_test = np.load(r"C:\Users\Michael\Downloads\Samples\output_test.npy")


inputs = tf.keras.Input(shape=(sample_size,sample_size,3))
initializer_x = tf.keras.layers.Conv2D(5, 3, (1,1), padding="valid", activation="relu")(inputs)
initializer_x = tf.keras.layers.Conv2D(10, 3, (1,1), padding="valid", activation="relu")(initializer_x)
initializer_x = tf.keras.layers.Conv2D(20, 3, (1,1), padding="valid", activation="relu")(initializer_x)

initializer_x = tf.keras.layers.Conv2DTranspose(20, 3, (1,1), padding="valid", activation="relu")(initializer_x)
initializer_x = tf.keras.layers.Conv2DTranspose(15, 3, (1,1), padding="valid", activation="relu")(initializer_x)

initializer_x = tf.keras.layers.Conv2DTranspose(3, 3, (1,1), padding="valid", activation="sigmoid")(initializer_x)
initializer_x = tf.keras.layers.add([tf.keras.layers.multiply([inputs, 1 - initializer_x]), initializer_x])

load_model = 1
model = tf.keras.Model(
    inputs=inputs, outputs=initializer_x) if not load_model else tf.keras.models.load_model(
        r"C:\Users\Michael\Downloads\Checkpoint\auto_epoch40_lr_0.01")

model_name = "auto"
learning_rate = 0.001

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=custom_loss)

callback = [tf.keras.callbacks.ModelCheckpoint(
    filepath=r"C:/Users/Michael/Downloads/Checkpoint/{}_epoch{}_lr_{}".format(model_name, "{epoch}", learning_rate))]

model.summary()

model.fit(input_data, output_data, batch_size=64, callbacks=callback,
          epochs=40, validation_data=(input_test[:len(input_test)//2],output_test[:len(input_test)//2]))
model.evaluate(input_test, output_test)

for i in range(len(input_test) // 2, len(input_test)):
    if tf.random.uniform(shape=[],maxval=1) > 0.07:
        continue
    save_image(input_test[i] * 255, r"C:/Users/Michael/Downloads/Project Saves/marked{}".format(i))
    save_image(model(tf.expand_dims(input_test[i], 0)) * 255,
               r"C:/Users/Michael/Downloads/Project Saves/model_performance{}".format(i))
