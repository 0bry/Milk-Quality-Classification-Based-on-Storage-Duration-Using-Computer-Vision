import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import datetime
import os

DATA_DIR = "data"  
IMG_SIZE = (224, 224)             
BATCH_SIZE = 32
INITIAL_EPOCHS = 10              
FINE_TUNE_EPOCHS = 5              
INIT_LR = 1e-3                    
FINE_TUNE_LR = 1e-5               
SEED = 123                        

def load_and_split_dataset(data_dir, validation_split=0.3, test_split=0.5):
    """Load dataset and split into train/val/test sets"""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int'
    )
    
    remaining_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int',
        shuffle=False  
    )
    
    num_batches = len(remaining_ds)
    val_ds = remaining_ds.take(num_batches // 2)
    test_ds = remaining_ds.skip(num_batches // 2)
    
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = load_and_split_dataset(DATA_DIR)


preprocess_input = tf.keras.applications.efficientnet.preprocess_input

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(
    lambda x, y: (preprocess_input(x), y),
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

val_ds = val_ds.map(
    lambda x, y: (preprocess_input(x), y),
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

test_ds = test_ds.map(
    lambda x, y: (preprocess_input(x), y),
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)


def build_model(num_classes=9):
    """Build EfficientNetB0 based model"""
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=IMG_SIZE + (3,)
    )
    base_model.trainable = False  

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)

model = build_model()

model.compile(optimizer=Adam(learning_rate=INIT_LR),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir = "training_logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "training_log.txt")
csv_logger = keras.callbacks.CSVLogger(log_file)

print("\n=== Initial Training ===")
initial_history = model.fit(
    train_ds,
    epochs=INITIAL_EPOCHS,
    validation_data=val_ds,
    callbacks=[csv_logger]
)

def unfreeze_layers(model, unfreeze_after=100):
    """Unfreeze layers for fine-tuning"""
    base_model = model.layers[1]
    base_model.trainable = True
    
    for layer in base_model.layers[:unfreeze_after]:
        layer.trainable = False
        
    return model

model = unfreeze_layers(model, unfreeze_after=100)

model.compile(optimizer=Adam(learning_rate=FINE_TUNE_LR),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\n=== Fine-Tuning Phase ===")
total_epochs = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
fine_tune_history = model.fit(
    train_ds,
    initial_epoch=initial_history.epoch[-1] + 1,
    epochs=total_epochs,
    validation_data=val_ds,
    callbacks=[csv_logger]
)

def plot_training_metrics(initial_history, fine_history=None):
    """Plot training and validation metrics"""
    acc = initial_history.history['accuracy']
    loss = initial_history.history['loss']
    val_acc = initial_history.history['val_accuracy']
    val_loss = initial_history.history['val_loss']

    if fine_history:
        acc += fine_history.history['accuracy']
        loss += fine_history.history['loss']
        val_acc += fine_history.history['val_accuracy']
        val_loss += fine_history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'training_metrics.png'))
    plt.show()

plot_training_metrics(initial_history, fine_tune_history)

print("\n=== Final Evaluation ===")
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

model.save(os.path.join(log_dir, 'milk_classifier.h5'))