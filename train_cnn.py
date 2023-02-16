import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


df = pd.read_csv('data/XSS_dataset.csv', encoding='utf-8-sig')
df.head()


def data2char_index(X, max_len):
    alphabet = " abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    result = []
    for data in X:
        mat = []
        for ch in data:
            if ch not in alphabet:
                continue
            mat.append(alphabet.index(ch))
        result.append(mat)
    X_char = tf.keras.preprocessing.sequence.pad_sequences(np.array(result, dtype=object), padding='post',
                                                           truncating='post', maxlen=max_len)
    return X_char


data = df['Sentence'].values
label = df['Label'].values

trainX, testX, y_train, y_test = train_test_split(
    data, label, test_size=0.2, random_state=42)

x_train = data2char_index(trainX, max_len=1000)
x_test = data2char_index(testX, max_len=1000)


x_train.shape

x_test.shape


def get_charcnn_model(max_len):
    main_input = tf.keras.layers.Input(shape=(max_len,))

    embedder = tf.keras.layers.Embedding(
        input_dim=70,
        output_dim=80,
        input_length=max_len,
        trainable=False
    )
    embed = embedder(main_input)
    #chập
    cnn1 = tf.keras.layers.Conv1D(
        32, 5, padding='same', strides=1, activation='relu')(embed)
    cnn1 = tf.keras.layers.MaxPooling1D(pool_size=12)(cnn1)

    cnn2 = tf.keras.layers.Conv1D(
        32, 10, padding='same', strides=1, activation='relu')(embed)
    cnn2 = tf.keras.layers.MaxPooling1D(pool_size=11)(cnn2)

    cnn3 = tf.keras.layers.Conv1D(
        32, 15, padding='same', strides=1, activation='relu')(embed)
    cnn3 = tf.keras.layers.MaxPooling1D(pool_size=10)(cnn3)

    cnn = tf.keras.layers.concatenate([cnn1, cnn2, cnn3], axis=1)

    flat = tf.keras.layers.Flatten()(cnn)

    drop = tf.keras.layers.Dropout(0.2)(flat)


    dense1 = tf.keras.layers.Dense(1024, activation='relu')(drop)
    dense2 = tf.keras.layers.Dense(128, activation='relu')(dense1)
    
    main_output = tf.keras.layers.Dense(1, activation='sigmoid')(dense2)
    model = tf.keras.Model(inputs=main_input, outputs=main_output)
    return model


model = get_charcnn_model(max_len=1000)
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir='logs/', histogram_freq=0, write_graph=True, write_images=True)

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']

)
model.summary()


batch_size = 128
num_epoch = 20
model_log = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=num_epoch,
    verbose=1,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard]
)

model.save('model.h5')

pred = model.predict(x_test)
y_pred = np.int64(pred > 0.5)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(" Accuracy : {0} \n Precision : {1} \n Recall : {2}".format(
    accuracy, precision, recall))
