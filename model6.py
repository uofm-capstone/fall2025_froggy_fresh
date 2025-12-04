import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import sys

# ============================================================
# STOP SCRIPT IF NO GPU
# ============================================================
gpus = tf.config.list_physical_devices("GPU")

if len(gpus) == 0:
    print("\n❌ ERROR: No GPU detected. Training aborted.\n"
          "   This script requires at least one GPU.\n"
          "   Make sure you request a GPU in your SLURM job:\n"
          "   #SBATCH --gres=gpu:1\n")
    sys.exit(1)
else:
    print(f"✔ GPU detected: {len(gpus)} available\n")

# ============================================================
# Load MobileNetV3 (Large) without top layers
# ============================================================
base_model = MobileNetV3Large(
    weights="imagenet",
    include_top=False,
    input_shape=(336, 336, 3)
)

base_model.trainable = False   # Phase 1: freeze entire base model

# ============================================================
# Add classification head (BatchNorm added)
# ============================================================
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)     # << ADDED
x = Dropout(0.5)(x)             # updated dropout
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# ============================================================
# Compile for Phase 1 (Frozen)
# ============================================================
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# ============================================================
# Data pipeline WITH STRONG AUGMENTATION
# ============================================================
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=15,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2
)

validation_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'data/',
    target_size=(336, 336),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = validation_datagen.flow_from_directory(
    'data/',
    target_size=(336, 336),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# ============================================================
# Callbacks
# ============================================================
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-7
)

# ============================================================
# PHASE 1 — Train Frozen Model (40 epochs max)
# ============================================================
print("\n===== PHASE 1: Training frozen base (40 epochs) =====\n")

model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=40,
    callbacks=[early_stop, reduce_lr]
)

# Save model after frozen phase
model.save("frog_detector_frozen.keras")

# ============================================================
# PHASE 2 — Fine-Tuning (Unfreeze Last 30 Layers)
# ============================================================
print("\n===== PHASE 2: Fine-tuning last 30 layers (20 epochs) =====\n")

# Unfreeze last 30 layers
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Recompile for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# Fine-tune the model
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    callbacks=[early_stop, reduce_lr]
)

# Save final fine-tuned model
model.save("frog_detector_finetuned.keras")
