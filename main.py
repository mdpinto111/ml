import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from logger import get_logger

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from sqlalchemy import create_engine, text
import joblib

from helpers.data.cleaning import (
    remove_single_value_cols,
    remove_null_cols,
    remove_irrelevant_cols,
)
from helpers.data.manipulate import make_onehot

from helpers.model.custom_callbacks import (
    checkpoint_best_callback,
    checkpoint_callback,
    LogProgressCallback,
)
from multi_collinearity import MultiCollinearityEliminator

model_type = "neural"  # "neural"  # "linear_model"
recreate_df = False
rows_top_selection = 120000
logger = get_logger("my_logger2" if model_type == "linear_model" else "my_logger")


def remove_correlated_features_advanced(df, threshold=0.9):
    mce = MultiCollinearityEliminator(df, "category", threshold)

    df_cleaned = mce.autoEliminateMulticollinearity()

    return df_cleaned


def remove_correlated_features(df, threshold=0.9):
    df = df.apply(pd.to_numeric, errors="coerce")  # Converts non-numeric values to NaN
    df = df.dropna(axis=1, how="all")  # Drop columns that are entirely NaN

    correlation_matrix = df.corr().abs()
    upper_tri = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    correlated_features = [
        column for column in upper_tri.columns if any(upper_tri[column] > threshold)
    ]
    return df.drop(columns=correlated_features)


def append_y(df):
    df_excel = pd.read_excel("./data/Book1.xlsx")

    top_50_values = df_excel["category"].value_counts().nlargest(50).index

    df_filtered_excel = df_excel[df_excel["category"].isin(top_50_values)]

    df = df.merge(df_filtered_excel[["TIK", "category"]], on="TIK", how="left")

    return df.dropna(subset=["category"])


def linear_model(X_train, X_test, y_train, y_test):
    # model = LogisticRegression(solver='saga', max_iter=1000)
    model = LogisticRegression(max_iter=1000)
    """
    linear_svc = LinearSVC(
        random_state=42,
        C=0.5445653829000621,
        dual=False,
        max_iter=1000,
        intercept_scaling=9.212225805943772,
        tol=1.6120432366603953e-06,
        verbose=1,
    )
    calibrated_classifier = CalibratedClassifierCV(linear_svc, method="sigmoid")
    pipeline = Pipeline([("classifier", calibrated_classifier)])
    """
    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", model)])

    pipeline.fit(X_train, y_train)

    y_pred_encoded = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_encoded)

    logger.info(accuracy)

    logreg = pipeline.named_steps["classifier"]
    coefficients = logreg.coef_[0]
    feature_names = X_train.columns

    top_20_indices = np.argsort(np.abs(coefficients))[-20:]
    top_20_features = [(feature_names[i], coefficients[i]) for i in top_20_indices]

    logger.info("Top 20 features that influence the prediction:\n")
    for feature, coeff in top_20_features:
        logger.info(f"{feature}: {coeff:.4f}")
    logger.info("=" * 10)
    logger.info("\n")


def build_model(input_dim, num_classes):
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),  # Input layer
            layers.Dense(1024, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),  # Regularization
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),  # Output layer
        ]
    )

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_data_before():
    if os.path.exists("./data/full_data_before.csv") and os.path.isfile(
        "./data/full_data_before.csv"
    ):
        df = pd.read_csv("./data/full_data_before.csv")
        logger.info("Loaded data before")
    else:
        servername = "bisqldwhd1"
        dbname = "MCH"
        engine = create_engine(
            f"mssql+pyodbc://@{servername}/{dbname}?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server"
        )
        df = pd.read_sql(
            f"SELECT top {rows_top_selection} * FROM [MCH].[SH\hm24].[yaakov_temp_table] WITH (NOLOCK)",
            engine,
        )
        logger.info(f"sql {rows_top_selection} selection finished")

        df.to_csv("./data/full_data_before.csv")
        logger.info("Saved data before")

    return df


logger.info("starting")


if recreate_df:
    df = get_data_before()

    df = append_y(df)
    logger.info("append_y finished")

    df = remove_irrelevant_cols(df)
    logger.info("remove_irrelevant_cols finished")

    df = remove_null_cols(df)
    logger.info("remove_null_cols finished")

    df = remove_single_value_cols(df)
    logger.info("remove_single_value_cols finished")

    # df = remove_correlated_features_advanced(df, threshold=0.95)
    # df = remove_correlated_features(df, threshold=0.95)
    # logger.info("remove_correlated_features finished")

    df = df.fillna(0)

    logger.info(f"Total rows: {len(df)}")

    logger.info(f"Total features before one-hot: {len(df.columns)}")
    df = make_onehot(df)
    logger.info(f"Total features after one-hot: {len(df.columns)}")

    df.head(100).to_csv("./data/full_data_after_sample.csv")
    df.to_csv("./data/full_data_after.csv")
    logger.info("Saved data after")
else:
    df = pd.read_csv("./data/full_data_after.csv")
    logger.info("Data loaded from csv")

    logger.info(f"Total features: {len(df.columns)}")


duplicate_columns = df.columns[df.columns.duplicated()].unique()
assert len(duplicate_columns) == 0


load_existing = True

if load_existing and os.path.exists("model_label_encoder.pkl"):
    encoder = joblib.load("model_label_encoder.pkl")
else:
    encoder = LabelEncoder()
    encoder.fit(df["category"])
    joblib.dump(encoder, "model_label_encoder.pkl")


df["category_encoded"] = encoder.transform(df["category"])


X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=["category", "category_encoded"]),
    df["category_encoded"],
    test_size=0.2,
    stratify=df["category_encoded"],
    random_state=42,
)

if model_type == "linear_model":
    linear_model(X_train, X_test, y_train, y_test)
else:
    num_classes = len(df["category_encoded"].unique())
    input_dim = df.drop(columns=["category", "category_encoded"]).shape[1]

    if load_existing and os.path.exists("model_checkpoint.keras"):
        logger.info("Loading saved model...")
        model = keras.models.load_model("model_checkpoint.keras")
    else:
        logger.info("Initializing new model...")
        model = build_model(input_dim, num_classes)

    model.summary()

    # Train the model
    model.fit(
        X_train,
        y_train,
        epochs=1500,
        batch_size=256,
        validation_data=(X_test, y_test),
        callbacks=[
            checkpoint_best_callback,
            checkpoint_callback,
            LogProgressCallback(),
        ],
    )
