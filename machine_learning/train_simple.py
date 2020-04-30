import time
from copy import copy

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_curve
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import binary_crossentropy

from machine_learning.keras.dataset import get_dataset, get_predictions_for_dataset
from machine_learning.serialization_utils import numpy_to_json, sklearn_precision_recall_curve_to_dict


def train_keras_model(*, model, x_train, y_train, x_valid, y_valid,
                      optimizer, lr,
                      batch_size=128,
                      n_epochs=100,
                      early_stopping_patience=6, metric_to_monitor='accuracy'):

    if x_valid is not None:
        print(f'len(valid): {len(x_valid)}')
        metric_to_monitor = f'val_{metric_to_monitor}'
        valid_generator = get_dataset(x=x_valid, y=y_valid,
                                      data_processor=image_processor,
                                      data_stats=data_stats, batch_size=batch_size)
    else:
        valid_generator = None

    metrics = ['accuracy']
    loss = 'binary_crossentropy'

    optimizer = optimizer(learning_rate=lr)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    verbosity = 0
    train_generator = get_dataset(x=x_train, y=y_train,
                                  data_processor=image_processor,
                                  data_stats=data_stats, batch_size=batch_size)

    callbacks = [
        EarlyStopping(monitor=metric_to_monitor, patience=early_stopping_patience, verbose=verbosity),
        ReduceLROnPlateau(monitor=metric_to_monitor, factor=0.5, patience=early_stopping_patience, min_lr=1e-6),
    ]

    history = model.fit_generator(train_generator, initial_epoch=0,
                                  epochs=n_epochs,
                                  callbacks=callbacks,
                                  validation_data=valid_generator,
                                  shuffle=True, verbose=1)


    y_actual_train, y_pred_train, y_pred_probs_train = get_predictions_for_dataset(train_generator, model)
    train_loss = binary_crossentropy(y_actual_train, y_pred_probs_train).numpy().tolist()

    # add more stats
    best_model_metadata.update({
        'history': history.history,
        'accuracy_train': accuracy_score(y_actual_train, y_pred_train),
        'f1_score_train': f1_score(y_actual_train, y_pred_train),
        'train_loss': train_loss,
        'loss': train_loss,
        'y_actual_train': y_actual_train,
        'y_pred_train': y_pred_train,
        'y_pred_probs_train': y_pred_probs_train,
        'epochs_trained': len(history.history['loss']),
        'elapsed_train_time': time.time() - train_start_time
    })

    if valid_generator is not None:
        y_actual_valid, y_pred_valid, y_pred_probs_valid = get_predictions_for_dataset(valid_generator, best_model)
        valid_loss = binary_crossentropy(y_actual_valid, y_pred_probs_valid).numpy().tolist()
        best_model_metadata.update({
            'accuracy_valid': accuracy_score(y_actual_valid, y_pred_valid),
            'f1_score_valid': f1_score(y_actual_valid, y_pred_valid),
            'confusion_matrix': confusion_matrix(y_actual_valid, y_pred_valid),
            'precision_recall_curve': precision_recall_curve(y_actual_valid, y_pred_valid),
            'y_actual_valid': y_actual_valid,
            'y_pred_valid': y_pred_valid,
            'y_pred_probs_valid': y_pred_probs_valid,
            'loss': valid_loss
        })

    serializable_metadata = copy(best_model_metadata)
    json_serializable_history = {}
    for k, v in history.history.items():
        # TODO - decide whether to keep these variables because we can get them from the model?
        # if k.contains("y_actual") or k.contains("y_pred"):
        #     continue
        json_serializable_history[k] = list(map(float, v))
    serializable_metadata['history'] = json_serializable_history

    if valid_generator is not None:
        serializable_metadata['confusion_matrix'] = numpy_to_json(serializable_metadata['confusion_matrix'])
        serializable_metadata['precision_recall_curve'] = sklearn_precision_recall_curve_to_dict(
            serializable_metadata['precision_recall_curve'])

    datasets = ["train", "valid"] if valid_generator is not None else ["train"]
    for dataset in datasets:
        for variable in ["y_actual", "y_pred", "y_pred_probs"]:
            serializable_metadata[f"{variable}_{dataset}"] = serializable_metadata[f"{variable}_{dataset}"].tolist()

    return best_model_metadata


print('RMSE train', rmse_train, '% difference from baseline', (rmse_train / rmse_train_baseline - 1) * 100)
print('RMSE test', rmse_test, '% difference from baseline', (rmse_test / rmse_test_baseline - 1) * 100)
print('R2 train', r2_train, '% difference from baseline', (r2_train / r2_train_baseline - 1) * 100)
print('R2 test', r2_test, '% difference from baseline', (r2_test / r2_test_baseline - 1) * 100)