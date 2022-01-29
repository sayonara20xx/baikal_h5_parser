from keras.backend import shape
from numpy.core.fromnumeric import trace
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import Sequential, layers, activations

import pandas as pd
import numpy as np

# if "Data cardinality is ambiguous" error appear there:
# 1. check data shapes: there's dims purposes (sample_num, required_data, )
# 2. it may be because of small size of `train` or `val` ndarrays, it must be greater than 1
# in my case i need to set at `val` data slice not [n] but [n:], training loop requires 2dim data

# validation: start this script in python REPL mode and use this with method your data:
# model.predict_classes([[{your_input_data}]]) # param should be list of lists or 2dim ndarray
# or just add at the required place in source file

def normalize_data(data_samples_list : np.array):
    # у значений вероятности очень больной разброс значений даже после такой нормализации
    # возможно, нужен совсем другой способ
    data_array_collumns = data_samples_list.swapaxes(0,1)
    for col in data_array_collumns:
        t_median = np.median(col)
        t_max = np.max(col)
        t_min = np.min(col)
        for (i, elem) in zip(range(len(col)), col):
            # to [-1, 1]
            elem = (elem - t_median)/(t_max - t_min)
            col[i] = elem

    print(data_array_collumns.swapaxes(0,1))

def load_data_from_csv(filename : str):
    # loading and returning ndarray type
    data_frame = pd.read_csv(filename, index_col=0)
    return data_frame.to_numpy()

def validate_shapes(x_train, y_train, x_val, y_val, verbose=1):
    if (verbose == 1):
        print(x_train.shape)
        print(y_train.shape)
        
        print(x_train)
        print(y_train)

        print("----------------------------------------")

        print(x_val.shape)
        print(y_val.shape)

        print(x_val)
        print(y_val)

def first_model():
    default_csv_data_folder = "./baikal_h5_parser/csv_data"

    input_data = load_data_from_csv(default_csv_data_folder + "/input-16424656810.csv")
    output_data = load_data_from_csv(default_csv_data_folder + "/output-16424656810.csv")

    # захардкожено
    x_train = input_data[:50]
    y_train = output_data[:50]

    x_val = input_data[50:]
    y_val = output_data[50:]

    model = Sequential()
    model.add(layers.Dense(250, activation=activations.selu, input_shape=(7,)))
    model.add(layers.Dense(2, activation=activations.tanh))
    model.summary()

    validate_shapes(x_train, y_train, x_val, y_val, verbose=1)

    # нормализация здесь нужна 100%
    # найти медиану и макс/мин значения, можно с помощью либ
    normalize_data(y_train)
    normalize_data(y_val)

    validate_shapes(x_train, y_train, x_val, y_val, verbose=1)

    model.compile(
    optimizer=Adam(learning_rate=0.00025, beta_1=0.93, beta_2=0.999),
    loss=MeanSquaredError(reduction="auto", name="mean_squared_error")
    )

    history = model.fit(
    x_train,
    y_train,
    epochs=250,
    validation_data=(x_val, y_val),
    )

    print(model.predict([[0, 0, 0, 0, 0, 0, 1]]))


if __name__ == "__main__":
    first_model()