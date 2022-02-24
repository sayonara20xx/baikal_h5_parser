from keras.backend import shape
from numpy.core.fromnumeric import trace
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError
from tensorflow.keras import Sequential, layers, activations, Model, Input, utils

import pandas as pd
import numpy as np

# if "Data cardinality is ambiguous" error appear there:
# 1. check data shapes: there's dims purposes (sample_num, required_data, )
# 2. it may be because of small size of `train` or `val` ndarrays, it must be greater than 1
# in my case i need to set at `val` data slice not [n] but [n:], training loop requires 2dim data

# validation: start this script in python REPL mode and use this with method your data:
# model.predict_classes([[{your_input_data}]]) # param should be list of lists or 2dim ndarray
# or just add at the required place in source file


def load_multiple_csvs(input_data : list, output_data : list, default_csv_data_folder : str = "./csv_data"):
    '''
        returns nothing!\n
        fill two input arrays with data, packs it into list, use [0]
    '''
    import os
    csvs_filenames = os.listdir(default_csv_data_folder)
    
    input_numpy_arrays = []
    output_numpy_arrays = []
    for csv_filename in csvs_filenames:
        relative_path = default_csv_data_folder + "/" + csv_filename
        if ("input" in csv_filename):
            current_array = load_data_from_csv(relative_path)
            input_numpy_arrays.append(current_array)
        elif ("output" in csv_filename):
            current_array = load_data_from_csv(relative_path)
            output_numpy_arrays.append(current_array)

    input_data.append(np.concatenate(input_numpy_arrays))
    output_data.append(np.concatenate(output_numpy_arrays))


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


def split_dataset(input_data : np.array, output_data : np.array, ratio : float = 0.9):
    split_idx = int(len(input_data)*ratio)
    #print(split_idx)
    return (input_data[:split_idx], output_data[:split_idx], input_data[split_idx:], output_data[split_idx:])


def first_model():
    input_data = []
    output_data = []

    load_multiple_csvs(input_data, output_data)

    x_train, y_train, x_val, y_val = split_dataset(input_data[0], output_data[0], ratio=0.9)

    model = Sequential()
    model.add(layers.Dense(500, activation=activations.selu, input_shape=(7,)))
    model.add(layers.Dense(350, activation=activations.sigmoid))
    model.add(layers.Dense(150, activation=activations.tanh))
    model.add(layers.Dense(2, activation=activations.tanh))
    model.summary()

    #validate_shapes(x_train, y_train, x_val, y_val, verbose=1)

    # нормализация здесь нужна 100%
    # найти медиану и макс/мин значения, можно с помощью либ
    normalize_data(y_train)
    normalize_data(y_val)

    validate_shapes(x_train, y_train, x_val, y_val, verbose=1)

    model.compile(
    optimizer=Adam(learning_rate=0.00005, beta_1=0.93, beta_2=0.999),
    loss=MeanSquaredError(reduction="auto", name="mean_squared_error")
    )

    history = model.fit(
    x_train,
    y_train,
    epochs=5000,
    validation_data=(x_val, y_val),
    batch_size=1,
    shuffle=True
    )

    print(model.predict([[0, 0, 0, 0, 0, 0, 1]]))


def secondModels():
    # loading data
    # both models have same input dataset, output is different
    # for t_0 first column of origin output data is suitable
    # second one for probability

    input_data = []
    output_data = []
    load_multiple_csvs(input_data, output_data)

    # normalizing data
    #normalize_data(input_data[0])
    normalize_data(output_data[0])

    output_data_swaped = np.array(output_data[0]).swapaxes(0, 1)
    output_data_time_0 = output_data_swaped[0]
    output_data_probability = output_data_swaped[1]

    # splitting data other way
    curr_ratio = 0.9
    x_train, y_train_t, x_val, y_val_t = split_dataset(input_data[0], output_data_time_0, ratio=curr_ratio)
    p_split_idx = int(len(output_data_probability)*curr_ratio)
    y_train_p, y_val_p = (output_data_probability[:p_split_idx], output_data_probability[p_split_idx:])

    # checking data appearance
    print("For model 1 (training data): ", x_train.shape, " ", y_train_t.shape)
    print("Data example: ", x_train[0], y_train_t[0])
    print("For model 1 (validation data): ", x_val.shape, " ", y_val_t.shape)
    print("Data example: ", x_val[0], y_val_t[0])
    print(" ----------------------------------------")
    print("For model 2 (training data): ", x_train.shape, " ", y_train_p.shape)
    print("Data example: ", x_train[0], y_train_p[0])
    print("For model 2 (validation data): ", x_val.shape, " ", y_val_p.shape)
    print("Data example: ", x_val[0], y_val_p[0])


    # creating models
    # for time_0 prediction
    model_t = Sequential()
    model_t.add(layers.Dense(250, activation=activations.selu, input_shape=(7,)))
    model_t.add(layers.Dense(140, activation=activations.sigmoid))
    model_t.add(layers.Dense(50, activation=activations.tanh))
    model_t.add(layers.Dense(1))
    model_t.summary()

    # for probability prediction
    model_p = Sequential()
    model_p.add(layers.Dense(500, activation=activations.selu, input_shape=(7,)))
    model_p.add(layers.Dense(350, activation=activations.sigmoid))
    model_p.add(layers.Dense(150, activation=activations.tanh))
    model_p.add(layers.Dense(1, activation=activations.tanh))
    model_p.summary()

    # compiling models
    model_t.compile(
    optimizer=Adam(learning_rate=0.0000005, beta_1=0.93, beta_2=0.999),
    loss=MeanSquaredError(reduction="auto", name="mean_square_error")
    )

    model_p.compile(
    optimizer=Adam(learning_rate=0.000001, beta_1=0.93, beta_2=0.999),
    loss=MeanSquaredError(reduction="auto", name="mean_square_error")
    )

    # training models
    history_t = model_t.fit(
    x_train,
    y_train_t,
    epochs=500,
    validation_data=(x_val, y_val_t),
    batch_size=500,
    shuffle=True
    )

    history_p = model_p.fit(
    x_train,
    y_train_p,
    epochs=200,
    validation_data=(x_val, y_val_p),
    batch_size=10,
    shuffle=True
    )


def unifiedModel():
    # loading data
    # both models have same input dataset, output is different
    # for t_0 first column of origin output data is suitable
    # second one for probability

    input_data = []
    output_data = []
    load_multiple_csvs(input_data, output_data)

    # normalizing data
    #normalize_data(input_data[0])
    normalize_data(output_data[0])

    output_data_swaped = np.array(output_data[0]).swapaxes(0, 1)
    output_data_time_0 = output_data_swaped[0]
    output_data_probability = output_data_swaped[1]

    # splitting data other way
    curr_ratio = 0.9
    x_train, y_train_t, x_val, y_val_t = split_dataset(input_data[0], output_data_time_0, ratio=curr_ratio)
    p_split_idx = int(len(output_data_probability)*curr_ratio)
    y_train_p, y_val_p = (output_data_probability[:p_split_idx], output_data_probability[p_split_idx:])

    # checking data appearance
    print("For model 1 (training data): ", x_train.shape, " ", y_train_t.shape)
    print("Data example: ", x_train[0], y_train_t[0])
    print("For model 1 (validation data): ", x_val.shape, " ", y_val_t.shape)
    print("Data example: ", x_val[0], y_val_t[0])
    print(" ----------------------------------------")
    print("For model 2 (training data): ", x_train.shape, " ", y_train_p.shape)
    print("Data example: ", x_train[0], y_train_p[0])
    print("For model 2 (validation data): ", x_val.shape, " ", y_val_p.shape)
    print("Data example: ", x_val[0], y_val_p[0])

    # creating models
    event_input = Input(shape=(7, ), name="event_input")

    x1 = layers.Dense(150, activation=activations.elu)(event_input)
    x2 = layers.Dense(100, activation=activations.tanh)(x1)
    x3 = layers.Dense(70, activation=activations.tanh)(x2)
    x4 = layers.Dense(20, activation=activations.tanh)(x3)

    time_output = layers.Dense(1, name="time_output")(x4)
    prob_output = layers.Dense(1, name="prob_output")(x4)
    
    model = Model(
    inputs=[event_input], outputs=[time_output, prob_output] 
    )

    model.compile(
    optimizer=Adam(learning_rate=0.005, beta_1=0.93, beta_2=0.999),
    loss={
        "time_output": MeanSquaredError(reduction="auto", name="mean_squared_error"),
        #"prob_output": MeanAbsolutePercentageError(reduction="auto", name="mean_absolute_percentage_error")
        "prob_output": MeanSquaredError(reduction="auto", name="mean_squared_error")
    }
    )

    model.summary()

    model.fit(
    x={"event_input": x_train},
    y={"time_output": y_train_t, "prob_output": y_train_p},
    validation_data=(
        {"event_input": x_val},
        {"time_output": y_val_t, "prob_output": y_val_p}
    ),
    batch_size=256,
    epochs=2000,
    shuffle=True
    )



if __name__ == "__main__":
    unifiedModel()
