import time

import numpy as np
import tensorflow as tf


def get_dataset(x, y, batch_size):
    numpy_tensor_slices = tf.data.Dataset.from_tensor_slices(x, y)
    dataset = tf.data.Dataset.from_generator(numpy_tensor_slices,
                                             output_types=(tf.string, tf.uint8),
                                             args=(x, y,)).shuffle(buffer_size=len(x))
    return dataset.prefetch(5).batch(batch_size)

def get_dataset(x, y, data_processor, batch_size):
    def image_loader(image_path, label):
        img = np.load(image_path.numpy())

        return processed_image, label

    def tf_image_loader(image_path, label):
        return tensorflow.py_function(func=image_loader,
                              inp=(image_path, label),
                              Tout=(tensorflow.float64,  # (H,W,3) img
                                    tensorflow.uint8))  # label

    dataset = tensorflow.data.Dataset.from_generator(image_path_and_label,
                                             output_types=(tensorflow.string, tensorflow.uint8),
                                             args=(x, y,)).shuffle(buffer_size=len(x))
    dataset = dataset.map(tf_image_loader, num_parallel_calls=8)

    return dataset.prefetch(5).batch(batch_size)


def get_predictions_for_dataset(dataset, model, threshold=.5):
    pred_y_batches = []
    actual_y_batches = []
    for batch_x, batch_y in dataset.make_one_shot_iterator():
        pred_y_batches.append(model.predict(batch_x))
        actual_y_batches.append(batch_y)

    pred_y_prob = []
    pred_y = []
    for pred_y_batch in pred_y_batches:
        for pred in pred_y_batch:
            pred_y_prob.append(pred)
            pred = 0 if pred < threshold else 1
            pred_y.append(pred)

    actual_y = []
    for actual_y_batch in actual_y_batches:
        for actual in actual_y_batch:
            actual_y.append(actual)

    return np.ravel(np.array(actual_y)), np.ravel(np.array(pred_y)), np.ravel(np.array(pred_y_prob))


def sanity_check_dataset(x, y, batch_size):
    dataset = get_dataset(x, y, batch_size=batch_size, data_processor=None)
    num_outputs = 0
    train_iter = dataset.make_one_shot_iterator()
    imgs, labels = train_iter.get_next()
    print(imgs.shape, imgs.numpy()[0][0][0], labels.shape, labels.numpy()[0])

    for batch_x, batch_y in dataset.make_one_shot_iterator():
        batch_x = batch_x.numpy()
        print(batch_x.mean(), batch_x.std(), batch_x.min(), batch_x.max())
        num_outputs += 1
        if num_outputs > 4:
            break

    start = time.time()
    num_batches = 0
    num_els = 0
    for x, y in dataset.make_one_shot_iterator():
        num_batches += 1
        num_els += len(x)
        continue
    print('\n')
    print(f'Dataset finished {num_batches} batches with {num_els} elements in {time.time() - start}')
    return dataset
