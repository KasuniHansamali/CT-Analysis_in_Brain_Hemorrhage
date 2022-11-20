from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import pandas as pd
import numpy as np
import os

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)

my_path = os.path.abspath(os.path.dirname(__file__))

# define model paths
mobileNet_model_path = os.path.join(my_path, "./models/MobileNetV2.h5")

def predict_single(image):
    image = [image]

    # load models
    model1_mobilenet = load_model(mobileNet_model_path)

    batch_size = 1
    nb_samples = 1

    df_file = pd.DataFrame({
        'filename': image
    })

    test_single_gen = ImageDataGenerator(rescale=1. / 255).flow_from_dataframe(
        df_file,
        x_col='filename',
        y_col=None,
        class_mode=None,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=False
    )

    # predict = model1_mobilenet.predict(test_single_gen, steps=np.ceil(nb_samples / batch_size))


    preds = model1_mobilenet.predict(test_single_gen, steps=np.ceil(nb_samples / batch_size))
    preds = np.array(preds)
    df_file['category'] = np.argmax(preds, axis=-1)

    # weighted_preds = np.tensordot(preds, weights, axes=((0), (0)))
    # weighted_ensemble_prediction = np.argmax(weighted_preds, axis=-1)

    label_map = {0: 'Hemorrhagic', 1: 'Normal'}
    df_file['category'] = df_file['category'].replace(label_map)

    labels = ['Hemorrhagic', 'Normal']

    # df = pd.DataFrame(
    #     data=weighted_preds,
    #     columns=labels)

    return df_file['category'][0]
