import numpy as np
import tensorflow as tf
from src.model import get_model
import os


def train_model(x_train, x_test, y_train, y_test,
                pt_weights=None, h5_name='temp'):
    tf.keras.backend.clear_session()
    model = get_model(1e-5)
    h_file = f'{h5_name}.h5'
    if pt_weights is not None:
        model.set_weights(pt_weights)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            h_file, 'val_binary_crossentropy',
            save_best_only=True, verbose=0),
        tf.keras.callbacks.EarlyStopping('val_binary_crossentropy', patience=30)
    ]
    hist = model.fit(x_train, y_train, verbose=0, epochs=2000,
                     callbacks=callbacks,
                     validation_data=(x_test, y_test),
                     batch_size=128)
    model.load_weights(h_file)
    os.remove(h_file)
    weights = np.array(model.get_weights(), dtype=object)
    return weights, min(hist.history['val_binary_crossentropy'])


def train_fixed_soup(x_train, x_test, y_train, y_test, f_name,
                     n_iter=40, n_soup=5):
    soup = []
    scores = []
    best_score = 1
    best_soup = []

    for i in range(n_soup):
        w, s = train_model(x_train, x_test, y_train, y_test)
        soup.append(w)
        scores.append(s)

    for j in range(n_iter):
        w, s = train_model(x_train, x_test, y_train, y_test)

        if s < max(scores):
            i = np.argmax(scores)
            del soup[i]
            del scores[i]
            soup.append(w)
            scores.append(s)

            soup_avg = np.mean(soup, axis=0)
            _, val = train_model(x_train, x_test, y_train, y_test,
                                 soup_avg, f_name)
            if val < best_score:
                best_score = val
                best_soup = soup_avg
                np.save(str(f_name) + '.npy', best_soup)

    return best_soup, best_score
