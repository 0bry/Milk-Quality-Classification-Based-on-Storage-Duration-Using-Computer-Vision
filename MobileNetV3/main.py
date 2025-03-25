#!/usr/bin/env python
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras import layers, models, optimizers, callbacks
import os

DATASET_PATH = "data"  
IMAGE_SIZE = (224, 224) 
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
SEED = 123
NUM_CLASSES = 9
EPOCHS = 10
LOG_FILENAME = "training_log.txt"

class LogTxtCallback(tf.keras.callbacks.Callback):
    def __init__(self, filename=LOG_FILENAME):
        super().__init__()
        self.filename = filename

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with open(self.filename, 'a') as f:
            f.write(f"Epoch {epoch+1}/{EPOCHS}: ")
            for key, value in logs.items():
                f.write(f"{key}: {value:.4f}  ")
            f.write("\n")

def main():
    if os.path.exists(LOG_FILENAME):
        os.remove(LOG_FILENAME)
    
    train_ds = image_dataset_from_directory(
        DATASET_PATH,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )
    
    val_ds = image_dataset_from_directory(
        DATASET_PATH,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    base_model = MobileNetV3Large(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = layers.Lambda(preprocess_input)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)  
    x = layers.Dropout(0.2)(x)  
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    model.summary()

    model.compile(
        optimizer=optimizers.Adam(),
        loss='sparse_categorical_crossentropy',  
        metrics=['accuracy']
    )

    checkpoint_cb = callbacks.ModelCheckpoint("milk_quality_model.h5", save_best_only=True, monitor="val_accuracy")
    earlystop_cb = callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
    logtxt_cb = LogTxtCallback(filename=LOG_FILENAME)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb, earlystop_cb, logtxt_cb]
    )

    print("Training History:")
    for key, values in history.history.items():
        print(f"{key}: {values}")

    print("Training complete. Model saved as 'milk_quality_model.h5'.")
    print(f"Training log saved in '{LOG_FILENAME}'.")

if __name__ == "__main__":
    main()
