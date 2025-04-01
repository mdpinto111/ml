from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from logger import get_logger

logger = get_logger("my_logger")

checkpoint_callback = ModelCheckpoint(
    filepath="model_checkpoint.keras",
    save_weights_only=False,  # Save full model (architecture + weights)
    save_freq="epoch",  # Save every epoch
    verbose=1,
)

checkpoint_best_callback = ModelCheckpoint(
    filepath="model_best.keras",
    monitor="accuracy",
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
)


class LogProgressCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logger.info(
            f"Epoch {epoch + 1}/{self.params['epochs']}, Loss: {logs.get('loss')}, Accuracy: {logs.get('accuracy')}, Val Loss: {logs.get('val_loss')}, Val Accuracy: {logs.get('val_accuracy')}"
        )
        if "lr" in logs:
            logger.info(f"Learning rate: {logs['lr']}")
