import argparse
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Input, SimpleRNN
from tensorflow.keras.models import Sequential


# create or retrieve the model results_df
def get_model_results_df(model_results_df_name):
    model_results_df_cols = [
        "model_name",
        "model_type",
        "avocado_type",
        "dense_layers",
        "MAE",
        "MSE",
        "R2",
        "loss",
        "val_loss",
        "y_test",
        "y_pred",
        "epoch_count",
        "lr_vals",
        "first_layer_units",
        "segments",
    ]

    if os.path.exists(model_results_df_name):  # try to load if exists
        with open(model_results_df_name, "rb") as pkl_file:
            return pickle.load(pkl_file)
    else:  # generate a fresh one
        df = pd.DataFrame(columns=model_results_df_cols)
        df.to_pickle(model_results_df_name)  # store
        return get_model_results_df(
            model_results_df_name
        )  # recursive call after save will return the pkl load


# results_df updater
def update_model_results_df(model_results_df_name, new_row):
    results_df = get_model_results_df(model_results_df_name)
    new_row_df = pd.DataFrame(data=[new_row], columns=results_df.columns.to_list())

    # updated df
    updated_model_results_df = pd.concat([results_df, new_row_df], ignore_index=True)

    # save to pkl on every tick
    updated_model_results_df.to_pickle(model_results_df_name)


# model callbacks
def get_model_callbacks(path, monitor, model):
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=path, monitor=monitor, save_best_only=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor, factor=0.05, patience=7, verbose=0
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=monitor, patience=15, verbose=0, start_from_epoch=40
    )

    # Custom callback to track and save the LR changes. Triggered on every epoch end.
    lambda_callback_lr_change = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: register_lr_change(epoch, model)
    )

    # return [model_checkpoint, reduce_lr, early_stopping, lambda_callback_lr_change]
    return [reduce_lr, early_stopping, lambda_callback_lr_change]


# custom callback function to handle and track the LR changes in the model fitting. Triggered on every epoch end.
def register_lr_change(epoch, model):
    # current_lr is a global list which is cleared before each model fitting
    current_lr = model.optimizer.lr.numpy()

    # when the LR differs than the previous value, current epoch is saved with the new value
    if current_lr not in lr_vals.values():
        lr_vals[epoch] = current_lr


# preprocess of data
def preprocess_data(data, T):
    X = list()
    y = list()

    for t in range(len(data) - T):
        # append segment
        x = data[t : t + T]
        X.append(x)

        # append next item
        y_ = data[t + T]
        y.append(y_)

    # reshape
    X = np.array(X).reshape(-1, T, 1)
    y = np.array(y)

    return X, y


# dynamic model generator
def generate_model(model_specs):
    # init values
    input_shape = (model_specs["segments"], 1)
    max_dropout = 0.5
    min_dropout = 0.1
    return_sequences = True

    model = Sequential()  # always Sequential

    # dynamic type received
    if model_specs["type"] == "RNN":
        model.add(Input(shape=input_shape))  # RNN input layer
        model.add(
            SimpleRNN(units=model_specs["type_units"], activation="relu")
        )  # RNN layer
    elif model_specs["type"] == "LSTM":
        model.add(
            LSTM(
                units=model_specs["type_units"],
                return_sequences=return_sequences,
                input_shape=input_shape,
            )
        )
        model.add(Dropout(max_dropout))  # dropout for LSTM
    elif model_specs["type"] == "GRU":
        model.add(
            GRU(
                units=model_specs["type_units"],
                return_sequences=return_sequences,
                input_shape=input_shape,
                activation="tanh",
            )
        )
        model.add(Dropout(max_dropout))  # dropout for GRU
    else:
        raise Exception("type not compatible")

    # dropout value calc vars #
    current_dl_layer = 0
    total_dense_layers = sum([x[0] for x in model_specs["dense_layers"]])
    step_dropout_val = 0.0  # dropout value on each step
    step_unit = (max_dropout - min_dropout) / (
        total_dense_layers - 2
    )  # dropout gain or loss on each step

    # dense layers generation following the received specification
    for dense_layer in model_specs["dense_layers"]:
        for i in range(
            dense_layer[0]
        ):  # dense_layer[0] is the number of dense layers with an specific unit
            current_dl_layer += 1
            next_layer = None

            # for the last LSTM/GRU layer, return_sequences is False and input_shape is None
            if current_dl_layer == total_dense_layers:
                return_sequences = False
                if model_specs["type"] == "LSTM":
                    next_layer = LSTM(units=dl_units, return_sequences=return_sequences)
                elif model_specs["type"] == "GRU":
                    next_layer = GRU(
                        units=dl_units,
                        return_sequences=return_sequences,
                        input_shape=input_shape,
                        activation="tanh",
                    )

            # calc dropout for LSTM/GRU
            if model_specs["type"] == "LSTM":
                # LSTM: dropout starts from min to max, ascending on each layer step
                dropout_val = min_dropout + (
                    current_dl_layer / (total_dense_layers)
                ) * (max_dropout - min_dropout)
                dropout_val = round(dropout_val, 1)  # round 1 decimal
            elif model_specs["type"] == "GRU":
                # GRU: dropout starts on max_dropout-0.1, decreases and then increases again
                if current_dl_layer == 1 or current_dl_layer == total_dense_layers:
                    # first and last layers set max_dropout-0.1
                    step_dropout_val = max_dropout - 0.1
                else:
                    if current_dl_layer <= (
                        total_dense_layers // 2 + 1.3
                    ):  # 1.3 is a custom offset
                        # first half (+1.3), decreases
                        step_dropout_val = step_dropout_val - step_unit
                    else:
                        # second half, increases
                        step_dropout_val = step_dropout_val + (
                            step_unit * 1.3
                        )  # 1.3 is a custom offset

                dropout_val = round(step_dropout_val, 1)  # round to 1 decimal

            dl_units = dense_layer[1]  # dense_layer[1] is the units value

            if next_layer:
                model.add(next_layer)  # when is the last one without input_shape
            elif model_specs["type"] == "RNN":
                model.add(Dense(units=dl_units, activation="relu"))
            elif model_specs["type"] == "LSTM":
                model.add(
                    LSTM(
                        units=dl_units,
                        return_sequences=return_sequences,
                        input_shape=input_shape,
                    )
                )  # + return_sequences and input_shape
            elif model_specs["type"] == "GRU":
                model.add(
                    GRU(
                        units=dl_units,
                        return_sequences=return_sequences,
                        input_shape=input_shape,
                        activation="tanh",
                    )
                )  # + return_sequences and input_shape

            if model_specs["type"] in ["GRU", "LSTM"]:  # GRU/LSTM add Dropout
                model.add(Dropout(dropout_val))

    # last layer should be units=1 because of the model ( time series forecast )
    model.add(Dense(units=1))

    return model


# model executor
def execute_models(data_df, model_specs, model_results_df_name, offset=False):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # disable logging

    # init some count vars for the offset
    type_steps = -1
    steps = -1

    # set the iterator
    ms_iterator = model_specs

    types = data_df["Type"].unique()  # get the different types
    for c_type in types:
        type_steps += 1
        c_type_data_df = data_df.copy()
        c_type_data_df = c_type_data_df[c_type_data_df["Type"] == c_type].sort_values(
            "Date", ascending=True
        )  # select only the current type, resort by date
        c_type_data_df = (
            c_type_data_df.groupby("Date")["Average Price"]
            .mean()
            .reset_index(drop=True)
        )  # final data is avg price per date

        # offset allows the partial execution if the previous one fails or stop early
        if offset and type_steps in offset.keys():
            offset_value = offset[type_steps]
            ms_iterator = model_specs[offset_value:]
            print(f"OFFSET: Jumping to model_specs[{offset_value}:]")
        else:
            ms_iterator = model_specs

        # iterate over each model_spec received
        for model_spec in tqdm.tqdm(ms_iterator):
            # set default specs values for missing configuration
            if "segments" not in model_spec.keys():
                model_spec["segments"] = 10  # T=10 as default
            if "epoch" not in model_spec.keys():
                model_spec["epochs"] = 200  # 200 epochs as default

            # perform the preprocess of data
            X, y = preprocess_data(c_type_data_df, model_spec["segments"])

            # Train Test Split, lineal 80/20, con el parámetro de shuffle = False, se quita la aleatoriedad del split train-test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, shuffle=False
            )

            # total_dense_layers = sum([x[0] for x in model_spec["dense_layers"]]) # debug
            # print(f"type: {model_spec['type']} - dense_layers: {total_dense_layers} - type_units: {model_spec['type_units']} - segments: {model_spec['segments']}")

            # model generation
            model = generate_model(model_spec)
            model.compile(
                optimizer="adam", loss="mse"
            )  # se compila el modelo con el optimizador Adam
            # model.summary() # comment for quiet mode

            # generate unique name for the model backup file
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d%H%M%S")
            model_name_path = (
                f'models/model_trained_{c_type}_{model_spec["type"]}_{timestamp}.keras'
            )

            # get model callbacks
            callbacks = get_model_callbacks(model_name_path, "val_loss", model)

            # train process
            lr_vals.clear()  # global list used for LR change tracking should be cleared before the train
            history = model.fit(
                x=X_train,
                y=y_train,
                validation_data=(X_test, y_test),
                epochs=model_spec["epochs"],
                callbacks=callbacks,
                verbose=False,
            )

            # trigger the prediction
            y_pred = model.predict(X_test)
            y_pred = y_pred[:, 0]

            # get model stats
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # save stats to the results df
            results_list = [
                model_name_path,
                model_spec["type"],
                c_type,
                model_spec["dense_layers"],
                mae,
                mse,
                r2,
                history.history["loss"],
                history.history["val_loss"],
                y_test,
                y_pred,
                max(history.epoch),
                lr_vals,
                model_spec["type_units"],
                model_spec["segments"],
            ]

            # update model results with the results
            update_model_results_df(model_results_df_name, results_list)


# generates a list of different models specs for testing and experimentation, depending on the configuration and values setted
def get_model_specs():
    # dense_layers list, first value is the number of layers and second one is the value
    dense_layers_list = [
        [
            [5, 256],
            [3, 128],
            [2, 64],
            [1, 32],
            [1, 16],
        ],
        [
            [2, 512],
            [2, 256],
            [2, 128],
            [2, 64],
            [1, 32],
            [1, 16],
        ],
        [
            [1, 1024],
            [1, 512],
            [1, 256],
            [1, 128],
            [1, 64],
            [1, 32],
            [1, 16],
        ],
        [
            [1, 2048],
            [1, 1024],
            [1, 512],
            [1, 256],
            [1, 128],
            [1, 64],
            [1, 32],
            [1, 16],
        ],
        [
            [1, 4096],
            [1, 2048],
            [1, 1024],
            [1, 512],
            [1, 256],
            [1, 128],
            [1, 64],
            [1, 32],
            [1, 16],
        ],
    ]

    model_specs = []  # list to save specs

    model_types = ["RNN", "LSTM", "GRU"]

    for t in model_types:
        for n in range(40, 200, 20):  # first layer unit
            for s in range(6, 16, 2):  # segments
                for dense_layers in dense_layers_list:
                    model_spec = {
                        "type": t,
                        "type_units": n,  # units value for the first layer
                        "segments": s,
                        "epochs": 200,
                        "dense_layers": dense_layers,
                    }
                    model_specs.append(model_spec)

    return model_specs


# graph the statistics (loss/val_loss + forecast + ratios + specs) of the models' performance
def plot_result(data, window_title="Figure"):
    # for idx, row in data.iterrows():
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), num=window_title)
    model_name = data["model_name"]
    fig.suptitle(f"MODEL: {model_name}", x=0.05, y=0.95, fontsize=18, ha="left")
    axes[0].plot(data["loss"], label="loss")
    axes[0].plot(data["val_loss"], label="val_loss")
    axes[0].legend()

    axes[1].plot(data["y_test"], label="forecast target")
    axes[1].plot(data["y_pred"], label="forecast prediction")
    axes[1].legend()

    # additional info: ratios + dense layers count, lr changes, epochs
    total_dense_layers = sum([x[0] for x in data["dense_layers"]])
    info_text = [
        "=Metrics=",
        f"MAE: {data['MAE']:.4f}",
        f"MSE: {data['MSE']:.4f}",
        f"R2: {data['R2']:.4f}",
        "",
        "=Config=",
        f"Model Type: {data['model_type']}",
        f"Dense Layers: {total_dense_layers}",
        f"LR Variations: {len(data['lr_vals'])}",
        f"Epochs: {data['epoch_count']}",
        f"Segments: {data['segments']}",
        f"1st layer units: {data['first_layer_units']}",
    ]

    info_text = "\n".join(info_text)
    fig.text(0.98, 0.98, info_text, fontsize=11, ha="right", va="top")
    plt.show()


# global configuration
lr_vals = {}  # global dict to save LR changes, used by a callback


def confirm_action(text="¿Do you want to continue?"):
    response = input(f"{text} (y/n): ").strip().lower()
    if response == "y":
        return True
    elif response == "n":
        print("Operation cancelled.")
        return False
    else:
        print("Wrong value. Please, answer with 'y' or 'n' options.")
        return (
            confirm_action()
        )  # Vuelve a pedir confirmación si la entrada no es válida.


def execute():
    # df to store the model performace stats, one file per day
    model_results_df_name = f'model_results_df_{datetime.now().strftime("%Y%m%d")}.pkl'
    get_model_results_df(model_results_df_name)
    print(f"Output info saved to: {model_results_df_name}")

    # load data df
    avocado_data_pkl = "Data/avocado_dataset_preprocessed.pkl"

    if os.path.exists(avocado_data_pkl):
        with open(avocado_data_pkl, "rb") as pkl_file:
            data_df = pickle.load(pkl_file)

    # get the list of models and its specs to be launched
    model_specs = get_model_specs()

    print(f"Number of models to be launched: {len(model_specs)}")

    if confirm_action():
        print("Excuting models, please be patient...")
        # execute the list of models for the current df
        execute_models(
            data_df=data_df,
            model_specs=model_specs,
            model_results_df_name=model_results_df_name,
        )
    else:
        print("Models execution aborted by the user.")

    return get_model_results_df(model_results_df_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter models executor")

    parser.add_argument(
        "-e",
        "--execute",
        action="store_true",
        help="Build and execute models. WARNING! THIS CAN BE A VERY HEAVY PROCESS",
    )

    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Plot models' performance results",
    )

    args = parser.parse_args()

    model_results_df = None
    all_models_df_name = "Data/all_models_full_df.pkl"
    if os.path.exists(all_models_df_name):
        with open(all_models_df_name, "rb") as pkl_file:
            model_results_df = pickle.load(pkl_file)

    if args.execute:
        model_results_df = execute()

    avocado_types = model_results_df["avocado_type"].unique()
    model_types = model_results_df["model_type"].unique()

    total_results = len(model_types) * len(avocado_types)
    current_result = 0
    for model_type in model_types:
        print()
        print(f"Results for {model_type}")
        data = model_results_df[model_results_df["model_type"] == model_type]
        for avocado_type in avocado_types:
            print(f"AVOCADO TYPE: {avocado_type}")
            filtered_data = data[data["avocado_type"] == avocado_type]
            filtered_data = filtered_data.sort_values("R2", ascending=False).head(1)
            for idx, row in filtered_data.iterrows():
                total_dense_layers = sum([x[0] for x in row["dense_layers"]])
                info_text = f"R2: {row['R2']:.4f} / {row['model_name']} / MAE: {row['MAE']:.4f} / MSE: {row['MSE']:.4f}"
                print(info_text)
                if args.plot:
                    input("\nResult will be plotted.\nPress any key to continue...\n")
                    current_result += 1
                    plot_result(row, f"Figure {current_result} of {total_results}")
