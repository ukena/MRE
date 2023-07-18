import gc
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from keras.layers import LSTM, RepeatVector, Dense, TimeDistributed, Masking, Dropout


def create_model(X_train, activation="tanh", optimizer=Adam(learning_rate=0.001), neurons=32):
    n_steps = X_train.shape[1]
    n_features = X_train.shape[2]

    model = Sequential()
    # Maskierungsschicht, um die Nullen zu ignorieren
    model.add(Masking(mask_value=0., input_shape=(None, n_features)))

    # Encoder
    model.add(LSTM(neurons, activation=activation, input_shape=(n_steps, n_features)))
    model.add(Dropout(0.2))
    # Codierer / Decoder umkehrbarer Vektor
    model.add(RepeatVector(n_steps))
    # Decoder
    model.add(LSTM(neurons, activation=activation, return_sequences=True))
    model.add(Dropout(0.2))
    # TimeDistributed wird verwendet, um denselben Dense Layer auf jeden Zeitschritt der Eingabesequenz anzuwenden.
    model.add(TimeDistributed(Dense(n_features)))
    # model.compile(optimizer=Adam(learning_rate=learning_rate, clipvalue=1.0), loss=safe_mae)
    model.compile(optimizer=optimizer, loss="mae")
    return model


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    def convert_timestamp(df) -> pd.DataFrame:
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
        df["hour"] = df["transaction_date"].dt.hour
        df["day_of_week"] = df["transaction_date"].dt.dayofweek
        df["month"] = df["transaction_date"].dt.month
        df["day_of_year"] = df["transaction_date"].dt.dayofyear
        # Umwandeln der extrahierten Merkmale in zyklische Koordinaten
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 23.0)
        df['hour_cos'] = np.cos(2 * np.pi * df["hour"] / 23.0)

        df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 6.0)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 6.0)

        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 11.0)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 11.0)

        df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.0)
        df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.0)

        # Entfernen der ursprünglichen und nicht zyklischen Merkmale
        df.drop(["transaction_date", "hour", "day_of_week", "month", "day_of_year"], axis=1, inplace=True)
        return df

    # create a new volume column by multiplying shares and price and making it negative if the transaction was a sale
    df["volume"] = df.apply(lambda row: round(row["shares"] * row["price"], 2) if row["acquired"] else round(-row["shares"] * row["price"], 2), axis=1)

    # one-hot encode categorical columns
    # df = pd.get_dummies(df, columns=["type", "code", "cpy_symbol", "rpt_title"])

    # converting date columns to periodic features
    df = convert_timestamp(df)

    # standardizing numeric columns
    for col in ["shares", "price", "shares_after", "volume"]:
        df[col] = StandardScaler().fit_transform(df[col].values.reshape(-1, 1))

    # for col in df.columns:
    #     df[col] = (df[col] - df[col].mean()) / df[col].std()

    # replace all nan with 0
    df.fillna(0.001, inplace=True)
    # replace all 0 in df with 0.001
    df.replace(0, 0.001, inplace=True)

    # convert all boolean columns to numeric
    df = df.replace({True: 1, False: -1})

    df["rpt_cik"] = df["rpt_cik"].astype(int)
    df["cik"] = df["cik"].astype(int)

    df["rpt_cik"] = df["rpt_cik"] / df["rpt_cik"].max()
    df["cik"] = df["cik"] / df["cik"].max()
    return df


def get_sequences(df: object, full: object = False) -> object:
    """
    If full is True, then the whole df is returned with all its sequences. Otherwise, the df is splitted in test and training df.
    :param df:
    :param full:
    :return:
    """
    longest_seq = df.groupby("rpt_cik").size().max() if not full else 9599
    print(f"[INFO] Longest sequence in df: {longest_seq}")

    if full:
        df = prepare_data(df)
        seq = df.groupby("rpt_cik", as_index=True).apply(lambda group: group.values.tolist() + [[0.0] * len(group.columns)] * (longest_seq - len(group)))

        del df
        gc.collect()

        X = np.array(seq.tolist())
        print(f"[INFO] X shape: {X.shape}")
        return X

    else:
        # get all unique rpt_cik
        unique_ciks = df["rpt_cik"].unique()

        # randomly choose 80% of the unique ciks and mark them as 'True', the rest as 'False'
        num_true = int(len(unique_ciks) * 0.8)
        mask = np.random.choice(unique_ciks, size=num_true, replace=False)

        # create a new column that is 'True' if 'rpt_cik' is in mask, and 'False' otherwise
        df["train"] = df["rpt_cik"].isin(mask)

        print(f"[INFO] Divided df into {df['train'].sum()} training and {len(df) - df['train'].sum()} test rows.")

        # prepare data where df["train"] is true separately from df["train"] is false
        df_train = prepare_data(df[df["train"] == True].drop(columns=["train"]))
        df_test = prepare_data(df[df["train"] == False].drop(columns=["train"]))

        print(f"[INFO] Training df contains {df_train['rpt_cik'].nunique()} insiders and test df contains {df_test['rpt_cik'].nunique()} insiders.")

        # if df_test or df_train contains any 0 values, print warning
        if (df_train == 0).any().any() or (df_test == 0).any().any():
            print("[WARNING] Any df contains 0 values. Those should be replaced with 0.001 to avoid them being skipped by the model.")

        # add 0.0 to all sequences to make them all equally long and use rpt_cik as index
        seq_train = df_train.groupby("rpt_cik", as_index=True).apply(lambda group: group.values.tolist() + [[0.0] * len(group.columns)] * (longest_seq - len(group)))
        seq_test = df_test.groupby("rpt_cik", as_index=True).apply(lambda group: group.values.tolist() + [[0.0] * len(group.columns)] * (longest_seq - len(group)))

        del df
        del df_train
        del df_test
        del unique_ciks
        del mask
        del num_true
        gc.collect()

        X_train = np.array(seq_train.tolist())
        del seq_train
        gc.collect()

        X_test = np.array(seq_test.tolist())
        del seq_test
        gc.collect()

        print(f"[INFO] X_train shape: {X_train.shape} | X_test shape: {X_test.shape}")
        return X_train, X_test


if __name__ == "__main__":
    df = pd.read_csv("df.csv", sep=";")
    df = pd.get_dummies(df, columns=["code"])
    df = df.drop(columns=["rpt_title", "type", "filling_id", "filling_date", "cpy_symbol", "cpy_name", "rpt_name"])

    X_train, X_test = get_sequences(df)

    model = create_model(X_train)

    history = model.fit(X_train, X_train, epochs=10, batch_size=32, validation_data=(X_test, X_test), shuffle=False)

    reconstructions = model.predict(X_test)
