import tensorflow as tf
import matplotlib.pyplot as plt
import random
import datetime
from data_process import *
from model import *


def custom_loss(y_true, y_pred):
    loss = tf.square(y_true - y_pred)
    return 100 * tf.reduce_mean(loss)


examples = []
for m in range(3):
    for i in range(10):
        examples.append([f"masked_{i}_{m}.jpg", f"clear_{i}.jpg"])

generate_sample_key = 0

sample_size = 128
if generate_sample_key:
    r"""input_data, output_data = create_samples_tensor(r"C:/Users/Michael/Downloads/Raw_images/",
                                                    examples, sample_size=sample_size)
    np.save(r"C:\Users\Michael\Downloads\Samples\input_data", input_data.numpy())
    np.save(r"C:\Users\Michael\Downloads\Samples\output_data", output_data.numpy())
    """
    input_data = np.load(r"C:\Users\Michael\Downloads\Samples\input_data.npy")
    output_data = np.load(r"C:\Users\Michael\Downloads\Samples\output_data.npy")
    shuffle = list(range(len(input_data)))
    random.shuffle(shuffle)
    input_train, output_train = input_data[shuffle[:len(input_data)//5*4]], output_data[shuffle[:len(input_data)//5*4]]
    input_test, output_test = input_data[shuffle[len(input_data)//5*4]:], output_data[shuffle[len(input_data)//5*4]:]

    np.save(r"C:\Users\Michael\Downloads\Samples\input_train", input_train)
    np.save(r"C:\Users\Michael\Downloads\Samples\output_train", output_train)
    np.save(r"C:\Users\Michael\Downloads\Samples\input_test", input_test)
    np.save(r"C:\Users\Michael\Downloads\Samples\output_test", output_test)

else:
    input_train = np.load(r"C:\Users\Michael\Downloads\Samples\input_train.npy")
    output_train = np.load(r"C:\Users\Michael\Downloads\Samples\output_train.npy")
    input_test = np.load(r"C:\Users\Michael\Downloads\Samples\input_test.npy")
    output_test = np.load(r"C:\Users\Michael\Downloads\Samples\output_test.npy")


load_model = 0
model_name = "auto_sigma_lay4x4_{}".format(datetime.datetime.now().strftime("%Y.%m.%d_%H-%M"))
tensorboard_dir = "C:/Users/Michael/Downloads/TensorBoard"
model = get_uncompiled_model(sample_size=sample_size,
                             model_name=model_name,
                             tensorboard_dir=tensorboard_dir) if not load_model else tf.keras.models.load_model(
        "C:/Users/Michael/Downloads/Checkpoint", compile=0)

learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=custom_loss)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=r"C:/Users/Michael/Downloads/Checkpoint/{}_epoch{}_lr_{}".format(model_name, "{epoch}", learning_rate))
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir=r"C:/Users/Michael/Downloads/TensorBoard/{}".format(model_name))

model.summary()
model.fit(input_train[:len(input_train)//20],
          output_train[:len(input_train)//20],
          batch_size=64,
          callbacks=[tensorboard,],
          epochs=6, validation_data=(input_test[:len(input_test)//10],output_test[:len(input_test)//10]))

model.evaluate(input_test, output_test)

for i in range(len(input_test)):
    if tf.random.uniform(shape=[],maxval=1) > 0.004:
        continue
    save_image(input_train[i] * 255, r"C:/Users/Michael/Downloads/Project Saves/marked_train{}".format(i))
    save_image(model(tf.expand_dims(input_train[i], 0)) * 255,
               r"C:/Users/Michael/Downloads/Project Saves/model_performance_train{}".format(i))
    save_image(input_test[i] * 255, r"C:/Users/Michael/Downloads/Project Saves/marked_test{}".format(i))
    save_image(model(tf.expand_dims(input_test[i], 0)) * 255,
               r"C:/Users/Michael/Downloads/Project Saves/model_performance_test{}".format(i))
