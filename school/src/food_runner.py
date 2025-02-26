from tensorflow import keras
import pathlib
import tensorflow as tf
import numpy as np

img_height = 180
img_width = 180

show_images = False


validation_path = pathlib.Path('../library/kaggle/test').with_suffix('')

train_ds = tf.keras.utils.image_dataset_from_directory(
  validation_path,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=32)

class_names = train_ds.class_names
print(class_names)

model = tf.keras.models.load_model('food_model_100.h5')
model.summary()
prediction_map = {}
for food_img_path in validation_path.glob('*/*.jpg'):
    img = keras.utils.load_img(
        food_img_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])


    if str(food_img_path).split('/')[4] not in prediction_map:
        prediction_map[str(food_img_path).split('/')[4]] = {'count': 0, 'correct': 0}
    if str(food_img_path).split('/')[4] == class_names[np.argmax(score)]:
        prediction_map[str(food_img_path).split('/')[4]]['correct'] += 1
    prediction_map[str(food_img_path).split('/')[4]]['count'] += 1

    scores = {class_name : "{:.2f}%".format(100* np.max(score)) for class_name, score in zip(class_names, score)}

    print(
        "This image {}/{} -> {} @ {:.2f}% confidence."
        .format(str(food_img_path).split('/')[4], str(food_img_path).split('/')[5], class_names[np.argmax(score)], 100 * np.max(score))
    )
    for k, v in scores.items():
        if "0.00%" == v:
            continue
        print(f"{k} -> {v}")
    if show_images:
        img.show()
        input("Press Enter to continue...")



           
for key in prediction_map:
    print(f"{key} -> {prediction_map[key]['correct']} / {prediction_map[key]['count']}")

