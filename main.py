import os

# Disable TensorFlow OneDNN optimizations to ensure compatibility with certain hardware
# or configurations that may not fully support these optimizations.
# This can help avoid unexpected behavior or performance issues.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from logger import get_logger

# Import necessary libraries for machine learning and data processing
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

# Import database and joblib for data storage and retrieval s
from sqlalchemy import create_engine, text
import joblib
from tensorflow.keras.callbacks import ModelCheckpoint, Callback

# Custom logger for logging messages during execution
logger = get_logger("my_logger")

# Define a callback to save the model at the end of every epoch
checkpoint_callback = ModelCheckpoint(
    filepath="model_checkpoint.keras",  # Filepath to save the model
    save_weights_only=False,  # Save the full model (architecture + weights)
    save_freq="epoch",  # Save the model at the end of every epoch
    verbose=1,  # Print a message when the model is saved
)

# Define a callback to save the best model based on accuracy
checkpoint_best_callback = ModelCheckpoint(
    filepath="model_best.keras",  # Filepath to save the best model
    monitor="accuracy",  # Metric to monitor for saving the best model
    save_best_only=True,  # Save only the best model
    save_weights_only=False,  # Save the full model
    verbose=1,  # Print a message when the best model is saved
)


# Custom callback to log progress at the end of each epoch
class LogProgressCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Log training and validation metrics at the end of each epoch
        logger.info(
            f"Epoch {epoch + 1}/{self.params['epochs']}, Loss: {logs.get('loss')}, Accuracy: {logs.get('accuracy')}, Val Loss: {logs.get('val_loss')}, Val Accuracy: {logs.get('val_accuracy')}"
        )
        # Log the learning rate if available
        if "lr" in logs:
            logger.info(f"Learning rate: {logs['lr']}")


# Function to perform one-hot encoding on specified columns
def make_onehot(df):
    # Hardcoded list of columns to one-hot encode (if any)
    onehot_hardcoded = []

    # Dynamically identify columns ending with "_onehot" for one-hot encoding
    onehot_from_data = [col for col in df.columns if col.endswith("_onehot")]

    # Combine hardcoded and dynamically identified columns
    onehot_columns = onehot_from_data + onehot_hardcoded

    # Perform one-hot encoding on the specified columns
    return pd.get_dummies(df, columns=onehot_columns, prefix_sep="_")


# Function to remove irrelevant columns from the dataset
def remove_irrelevant_cols(df):
    # Drop columns based on specific patterns or names
    df = df.drop(df.filter(like="T_IDCUN", axis=1).columns, axis=1)
    df = df.drop(df.filter(like="_IDKUN", axis=1).columns, axis=1)
    df = df.drop(df.filter(like="_TIK", axis=1).columns, axis=1)
    df = df.drop(df.filter(like="_KLITA", axis=1).columns, axis=1)
    df = df.drop(df.filter(like="_TAR", axis=1).columns, axis=1)
    df = df.drop(df.filter(like="TAR_", axis=1).columns, axis=1)
    df = df.drop(df.filter(like="ANAF", axis=1).columns, axis=1)
    df = df.drop(df.filter(like="_FAX", axis=1).columns, axis=1)
    df = df.drop(df.filter(like="MIKUD", axis=1).columns, axis=1)
    df = df.drop(df.filter(like="_YESHUT", axis=1).columns, axis=1)
    df = df.drop(df.filter(like="_TELFON", axis=1).columns, axis=1)

    # Drop additional columns explicitly listed
    drop_cols = [
        "DIRA",
        "EM_MAKOR_DOH_min_DOH_YESHUV_ESEK",
        "DIRA_ESEK",
        "TEL_ESEK",
        "KIDOMET_ESEK",
        "last_tik",
        "TA_DOAR",
        "TIK",
        "LSTNGR90_NO",
        "M90",
        "T_PTIHA",
        "TMPTIHA",
        "T_KNISA",
        "TIK_KODEM",
        "TA_DOAR",
        "YSHUV",
        "YSHUV_ESEK",
        "KIDOMET_ESEK",
        "DIRA_PRATI",
        "YSHUV_PRATI",
        "TEL_KODEM",
        "KOD_EMAIL",
        "SH_PIRTEI_NISHOM_PRT_TZ_BZ_1",
        "SH_PIRTEI_NISHOM_PRT_MIN_BZ_1_onehot",
        "SH_PIRTEI_NISHOM_PRT_MZV_MISH_BZR",
        "SH_PIRTEI_NISHOM_PRT_D_MZV_MISH",
        "SH_PIRTEI_NISHOM_PRT_TZ_BZ_2",
        "SH_PIRTEI_NISHOM_PRT_MIN_BZ_2_onehot",
        "SH_PIRTEI_NISHOM_PRT_D_BKST_SHN_ZUG",
        "SH_PIRTEI_NISHOM_PRT_D_SHDR_SHN_ZUG",
        "SH_PIRTEI_NISHOM_PRT_SHANA_AHARONA_DOCH",
        "SH_PIRTEI_NISHOM_PRT_OP_D_ESK",
        "SH_PIRTEI_NISHOM_PRT_DIV_CLS_D_ESK",
        "SH_PIRTEI_NISHOM_PRT_CLS_D_ESK",
        "SH_PIRTEI_NISHOM_PRT_OP_D_ESK_NEW",
        "SH_PIRTEI_NISHOM_PRT_ZEHUT_MESHADER",
        "SH_PIRTEI_NISHOM_PRT_LAST_DATE_UPD",
        "SH_SHUMA_MAST_MST_BZ_MEVUTAL_onehot",
        "SH_SHUMA_MAST_MST_HAZHARA_ZUG_NIF_onehot",
        "SH_SHUMA_MAST_MST_MIS_MEYAZEG",
        "SH_SHUMA_MAST_MST_MIN1_onehot",
        "SH_SHUMA_MAST_MST_MIN2_onehot",
        "SH_SHUMA_MAST_MST_SUGTIK_PAIL_KODEM",
        "SH_SHUMA_MAST_MST_MOED_HUKI",
        "SH_SHUMA_MAST_MST_KOD_MISMAC",
        "SH_SHUMA_MAST_MST_KOD_NIHUL_S",
        "SH_SHUMA_MAST_MST_HACHNASA_HEV_BAIT",
        "SH_SHUMA_MAST_MST_MIS_YELADIM",
        "SH_SHUMA_MAST_MST_MAAM_103",
        "SH_SHUMA_MAST_MST_SEMEL_ORECH_SHUMA",
        "SH_SHUMA_MAST_MST_RAKAZ_ISHUR",
        "SH_SHUMA_MAST_MST_HANMAKA_KODEM",
        "SH_SHUMA_MAST_MST_YITRAT_SHANIM_LPRISA",
        "SH_SHUMOT_SHM_KOD_MAZAV",
        "SH_SHUMOT_SHM_ZEHUT_RASHUM",
        "SH_SHUMOT_SHM_MIN_RASHUM_onehot",
        "SH_SHUMOT_SHM_SEIF_SHUMA",
        "SH_SHUMOT_SHM_GERAON",
        "SH_SHUMOT_SHM_ZEHUT_MESHADER_NITUV_B",
        "SH_SHUMOT_SHM_ZEHUT_MEF_ISHUR",
        "SH_SHUMOT_SHM_SEMEL_MEASHER",
        "SH_SHUMOT_SHM_ZEHUT_BZ",
        "SH_SHUMOT_SHM_YITRAT_SHANIM_LPRISA",
        "SH_SHUMOT_SHM_SHIDUR_INT",
        "SH_SHUMOT_SHM_LLO_KOD_ISUF",
        "SH_INT_SHUMA_INT_SHM_Z_MESHADER",
        "SH_INT_SHUMA_INT_SHM_TIME_SHIDUR",
        "SH_INT_SHUMA_INT_SHM_BARCODE",
        "SH_INT_SHUMA_INT_SHM_ZEHUT_R",
        "SH_INT_SHUMA_INT_SHM_TEL_AVODA_R",
        "SH_INT_SHUMA_INT_SHM_ZEHUT_BZ",
        "SH_INT_SHUMA_INT_SHM_TEL_AVODA_BZ",
        "SH_INT_SHUMA_INT_SHM_TEL_BAIT",
        "SH_INT_SHUMA_INT_SHM_TEL_ACHER",
        "SH_INT_SHUMA_INT_SHM_MISPAR_OSEK",
        "SH_INT_SHUMA_INT_SHM_TEL_OZER",
        "SH_INT_SHUMA_INT_SHM_MIN_RASHUM",
        "SH_INT_SHUMA_INT_SHM_MIN_BZ",
        "MIS_HEVRA",
        "MM_HEVROT_MIS_REHOV",
        "MM_HEVROT_MIS_BAIT",
        "MM_HEVROT_SEMEL_ISHUV",
        "MM_HEVROT_SW_NIMRUR_KTVT_onehot",
        "MM_HEVROT_TA_DOAR",
        "MM_HEVROT_ISHUV_TA_DOAR",
        "MM_HEVROT_SUG_HEVRA_KODEM_onehot",
        "MM_HEVROT_KOD_MATARA",
        "MM_HEVROT_MIS_HEVRA_KODEM",
        "MM_HEVROT_STATUS_KODEM",
        "MM_HEVROT_TAT_SUG_onehot",
        "MM_HEVROT_TELEPHONE1",
        "MM_HEVROT_MIS_DIRA",
        "SH_DOH_KASPIM_KSP_ZEHUT",
        "SH_DOH_KASPIM_KSP_MIS_SHUTAFIM",
        "SH_DOH_KASPIM_KSP_MATBE_DIVUCH",
        "DOH_MIS_OSEK",
    ]

    # Drop the specified columns from the dataset
    return df.drop(drop_cols, axis=1)


# Function to remove columns with all null values
def remove_null_cols(df):
    # Drop columns where all values are NaN
    df_cleaned = df.dropna(axis=1, how="all")
    return df_cleaned


# Function to remove columns with a single unique value
def remove_single_value_cols(df):
    # Retain only columns with more than one unique value
    return df.loc[:, df.nunique() != 1]


# Define model type and configuration
model_type = "neural"  # Options: "neural" or "linear_model"
recreate_df = False  # Flag to recreate the dataset from the database
rows_top_selection = 120000  # Number of rows to select from the database
logger = get_logger("my_logger2" if model_type == "linear_model" else "my_logger")


# Function to remove highly correlated features using advanced techniques
# This uses a custom MultiCollinearityEliminator class to identify and remove
# features that are highly correlated with others, based on a given threshold.
def remove_correlated_features_advanced(df, threshold=0.9):
    mce = MultiCollinearityEliminator(df, "category", threshold)
    df_cleaned = mce.autoEliminateMulticollinearity()
    return df_cleaned


# Function to remove highly correlated features using a simpler approach
# This calculates the correlation matrix and removes features with a correlation
# coefficient above the specified threshold.
def remove_correlated_features(df, threshold=0.9):
    # Convert non-numeric values to NaN to ensure compatibility with correlation calculations
    df = df.apply(pd.to_numeric, errors="coerce")
    # Drop columns that are entirely NaN
    df = df.dropna(axis=1, how="all")
    # Compute the absolute correlation matrix
    correlation_matrix = df.corr().abs()
    # Extract the upper triangle of the correlation matrix to avoid duplicate comparisons
    upper_tri = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    # Identify columns with correlation above the threshold
    correlated_features = [
        column for column in upper_tri.columns if any(upper_tri[column] > threshold)
    ]
    # Drop the identified correlated columns
    return df.drop(columns=correlated_features)


# Function to append target labels (y) to the dataset
# This merges the main dataset with an Excel file containing the target labels
# and filters the data to include only the top 50 categories.
def append_y(df):
    df_excel = pd.read_excel("./data/Book1.xlsx")
    # Identify the top 50 most frequent categories
    top_50_values = df_excel["category"].value_counts().nlargest(50).index
    # Filter the Excel data to include only rows with these top categories
    df_filtered_excel = df_excel[df_excel["category"].isin(top_50_values)]
    # Merge the filtered Excel data with the main dataset on the "TIK" column
    df = df.merge(df_filtered_excel[["TIK", "category"]], on="TIK", how="left")
    # Drop rows where the "category" column is missing
    return df.dropna(subset=["category"])


# Function to train and evaluate a linear model
# This uses logistic regression to classify the data and logs the top features
# influencing the predictions.
def linear_model(X_train, X_test, y_train, y_test):
    # Initialize a logistic regression model with a maximum of 1000 iterations
    model = LogisticRegression(max_iter=1000)
    # Create a pipeline with a standard scaler and the logistic regression model
    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", model)])
    # Train the pipeline on the training data
    pipeline.fit(X_train, y_train)
    # Predict the target variable for the test data
    y_pred_encoded = pipeline.predict(X_test)
    # Calculate and log the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred_encoded)
    logger.info(accuracy)

    # Log the top 20 features influencing the predictions
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


# Function to build a neural network model
# This creates a sequential model with multiple dense layers, batch normalization,
# and dropout for regularization.
def build_model(input_dim, num_classes):
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),  # Input layer
            layers.Dense(1024, activation="relu"),  # First hidden layer
            layers.BatchNormalization(),  # Normalize activations
            layers.Dropout(0.3),  # Dropout for regularization
            layers.Dense(512, activation="relu"),  # Second hidden layer
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation="relu"),  # Third hidden layer
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),  # Output layer
        ]
    )
    # Compile the model with Adam optimizer and sparse categorical crossentropy loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# Function to load or fetch data from the database
# If a cached CSV file exists, it loads the data from the file. Otherwise, it fetches
# the data from the database and saves it to a CSV file for future use.
def get_data_before():
    if os.path.exists("./data/full_data_before.csv") and os.path.isfile(
        "./data/full_data_before.csv"
    ):
        # Load data from the cached CSV file
        df = pd.read_csv("./data/full_data_before.csv")
        logger.info("Loaded data before")
    else:
        # Connect to the database and fetch the data
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
        # Save the fetched data to a CSV file
        df.to_csv("./data/full_data_before.csv")
        logger.info("Saved data before")
    return df


# Main script execution starts here
logger.info("starting")

if recreate_df:
    # Recreate the dataset if the flag is set
    df = get_data_before()
    df = append_y(df)
    logger.info("append_y finished")
    df = remove_irrelevant_cols(df)
    logger.info("remove_irrelevant_cols finished")
    df = remove_null_cols(df)
    logger.info("remove_null_cols finished")
    df = remove_single_value_cols(df)
    logger.info("remove_single_value_cols finished")
    df = df.fillna(0)  # Fill missing values with 0
    logger.info(f"Total rows: {len(df)}")
    logger.info(f"Total features before one-hot: {len(df.columns)}")
    df = make_onehot(df)  # Convert categorical variables to one-hot encoding
    logger.info(f"Total features after one-hot: {len(df.columns)}")
    # Save the processed dataset to CSV files
    df.head(100).to_csv("./data/full_data_after_sample.csv")
    df.to_csv("./data/full_data_after.csv")
    logger.info("Saved data after")
else:
    # Load the dataset from a saved CSV file
    df = pd.read_csv("./data/full_data_after.csv")
    logger.info("Data loaded from csv")
    logger.info(f"Total features: {len(df.columns)}")

# Ensure there are no duplicate columns in the dataset
duplicate_columns = df.columns[df.columns.duplicated()].unique()
assert len(duplicate_columns) == 0

# Load or create a label encoder for the target variable
load_existing = True
if load_existing and os.path.exists("model_label_encoder.pkl"):
    # Load the existing label encoder from a file
    encoder = joblib.load("model_label_encoder.pkl")
else:
    # Create a new label encoder and save it to a file
    encoder = LabelEncoder()
    encoder.fit(df["category"])
    joblib.dump(encoder, "model_label_encoder.pkl")

# Encode the target variable using the label encoder
df["category_encoded"] = encoder.transform(df["category"])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=["category", "category_encoded"]),  # Features
    df["category_encoded"],  # Target variable
    test_size=0.2,  # 20% of the data for testing
    stratify=df["category_encoded"],  # Ensure class distribution is maintained
    random_state=42,  # Set random seed for reproducibility
)

# Train the appropriate model based on the selected model type
if model_type == "linear_model":
    # Train and evaluate a linear model
    linear_model(X_train, X_test, y_train, y_test)
else:
    # Train and evaluate a neural network model
    num_classes = len(df["category_encoded"].unique())  # Number of unique classes
    input_dim = df.drop(columns=["category", "category_encoded"]).shape[1]  # Input size

    if load_existing and os.path.exists("model_checkpoint.keras"):
        # Load a previously saved model
        logger.info("Loading saved model...")
        model = keras.models.load_model("model_checkpoint.keras")
    else:
        # Initialize a new model
        logger.info("Initializing new model...")
        model = build_model(input_dim, num_classes)

    model.summary()  # Print the model architecture

    # Train the neural network model
    model.fit(
        X_train,
        y_train,
        epochs=1500,  # Number of training epochs
        batch_size=256,  # Batch size for training
        validation_data=(X_test, y_test),  # Validation data
        callbacks=[
            checkpoint_best_callback,  # Save the best model during training
            checkpoint_callback,  # Save model checkpoints
            LogProgressCallback(),  # Log training progress
        ],
    )
