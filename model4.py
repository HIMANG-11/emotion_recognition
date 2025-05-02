import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, GlobalAveragePooling2D,
                                     Dense, BatchNormalization, Dropout, Input, LeakyReLU,
                                     Reshape, LSTM) # Added Reshape and LSTM
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.optimizers.schedules import ExponentialDecay # Not used in provided code
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight
# from tensorflow.keras.regularizers import l2 # Not used in provided code structure

# --- GPU Configuration (Keep as is) ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, using CPU.")

# --- Paths and Hyperparameters (Keep as is) ---
TRAIN_DIR = "/home/himang/my_project/data/balanced_affectnet_train"
TEST_DIR = "/home/himang/my_project/data/balanced_affectnet_test1"
IMG_HEIGHT, IMG_WIDTH = 48, 48
NUM_CLASSES = 7
BATCH_SIZE = 64
EPOCHS = 100
INITIAL_LR = 0.001

# --- Data Augmentation and Generators (Keep as is) ---
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,
    rotation_range=35,
    width_shift_range=0.22,
    height_shift_range=0.22,
    shear_range=0.25,
    zoom_range=[0.9, 1.1],
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode="nearest"
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    subset='training',
    shuffle=True,
    seed=42
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    subset='validation',
    shuffle=True,
    seed=42
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False
)

# --- Class Weights (Keep as is) ---
# Compute class weights only if train_generator is not empty
if train_generator.samples > 0:
    class_weights_values = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights = dict(enumerate(class_weights_values))
    print("Class weights computed:", class_weights)
else:
    print("Warning: Training generator is empty. Cannot compute class weights.")
    class_weights = None # Set to None if no data


# --- CNN-RNN Hybrid Model Architecture ---
# --- Corrected CNN-RNN Hybrid Model Architecture ---
def build_cnn_rnn_model(img_height, img_width, num_classes):
    model = Sequential(name="CNN_RNN_Facial_Emotion_Recognition")
    model.add(Input(shape=(img_height, img_width, 1)))

    # --- CNN Feature Extractor Blocks ---
    # Block 1
    model.add(Conv2D(32, (3, 3), padding='same', name='conv1a'))
    model.add(LeakyReLU(negative_slope=0.1, name='lrelu1a')) # Use negative_slope
    model.add(BatchNormalization(name='bn1a'))
    model.add(Conv2D(32, (3, 3), padding='same', name='conv1b'))
    model.add(LeakyReLU(negative_slope=0.1, name='lrelu1b')) # Use negative_slope
    model.add(BatchNormalization(name='bn1b'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool1')) # Output: (24, 24, 32)
    model.add(Dropout(0.25, name='drop1'))

    # Block 2
    model.add(Conv2D(64, (3, 3), padding='same', name='conv2a'))
    model.add(LeakyReLU(negative_slope=0.1, name='lrelu2a')) # Use negative_slope
    model.add(BatchNormalization(name='bn2a'))
    model.add(Conv2D(64, (3, 3), padding='same', name='conv2b'))
    model.add(LeakyReLU(negative_slope=0.1, name='lrelu2b')) # Use negative_slope
    model.add(BatchNormalization(name='bn2b'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool2')) # Output: (12, 12, 64)
    model.add(Dropout(0.3, name='drop2'))

    # Block 3
    model.add(Conv2D(128, (3, 3), padding='same', name='conv3a'))
    model.add(LeakyReLU(negative_slope=0.1, name='lrelu3a')) # Use negative_slope
    model.add(BatchNormalization(name='bn3a'))
    model.add(Conv2D(128, (3, 3), padding='same', name='conv3b'))
    model.add(LeakyReLU(negative_slope=0.1, name='lrelu3b')) # Use negative_slope
    model.add(BatchNormalization(name='bn3b'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool3')) # Output: (6, 6, 128)
    model.add(Dropout(0.35, name='drop3'))

    # Block 4
    model.add(Conv2D(256, (3, 3), padding='same', name='conv4a'))
    model.add(LeakyReLU(negative_slope=0.1, name='lrelu4a')) # Use negative_slope
    model.add(BatchNormalization(name='bn4a'))
    model.add(Conv2D(256, (3, 3), padding='same', name='conv4b'))
    model.add(LeakyReLU(negative_slope=0.1, name='lrelu4b')) # Use negative_slope
    model.add(BatchNormalization(name='bn4b'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool4')) # Output: (3, 3, 256)
    model.add(Dropout(0.35, name='drop4'))

    # --- Bridge from CNN to RNN --- # <<<<<<<<<<<<<<< CHECK THIS SECTION CAREFULLY
    # The output shape from the last Dropout (drop4) is (batch_size, 3, 3, 256).
    # Reshape this into a sequence: (batch_size, timesteps, features).
    # We'll use height (3) as timesteps and width*channels (3 * 256 = 768) as features.

    # ** THE INCORRECT LINES BELOW SHOULD BE REMOVED/COMMENTED OUT **
    # last_cnn_output_shape = model.layers[-2].output_shape # REMOVE THIS LINE
    # print(f"Shape before Reshape: {last_cnn_output_shape}") # REMOVE THIS LINE
    # target_shape = (last_cnn_output_shape[1], last_cnn_output_shape[2] * last_cnn_output_shape[3]) # REMOVE THIS LINE
    # print(f"Target shape for Reshape: {target_shape}") # REMOVE THIS LINE

    # ** USE THIS HARDCODED SHAPE INSTEAD **
    reshape_target_shape = (3, 3 * 256) # (timesteps, features)
    print(f"Reshaping CNN output to target shape for RNN: {reshape_target_shape}") # Optional print
    model.add(Reshape(reshape_target_shape, name='reshape_for_rnn'))
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # --- RNN Part ---
    model.add(LSTM(128, return_sequences=False, name='lstm'))
    model.add(BatchNormalization(name='bn_lstm'))
    model.add(Dropout(0.4, name='drop_lstm'))

    # --- Final Classification Layers ---
    model.add(Dense(128, name='dense1'))
    model.add(LeakyReLU(negative_slope=0.1, name='lrelu_dense1')) # Use negative_slope
    model.add(BatchNormalization(name='bn_dense1'))
    model.add(Dropout(0.5, name='drop_dense1'))

    model.add(Dense(num_classes, activation='softmax', name='output_softmax'))

    return model

# Build the hybrid model
model = build_cnn_rnn_model(IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES)

# Print model summary to check the architecture
model.summary()

# --- Optimizer (Keep as is) ---
optimizer = Adam(learning_rate=INITIAL_LR)

# --- Compile (Keep as is) ---
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy']
)

# --- Callbacks (Keep as is, consider adjusting patience if needed) ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1), # Increased patience slightly more
    ModelCheckpoint("emotion_recognition_cnn_rnn_model_best.h5", monitor="val_loss", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1) # Increased patience slightly more
]

# --- Training (Keep as is, check steps) ---
# Ensure generators are not empty before calculating steps
if train_generator.samples > 0 and val_generator.samples > 0 :
    steps_per_epoch = train_generator.samples // BATCH_SIZE
    validation_steps = val_generator.samples // BATCH_SIZE

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")


    # Check if steps are zero, which happens if batch size > samples
    if steps_per_epoch == 0:
        print("Warning: steps_per_epoch is 0. Check batch size vs training samples.")
        steps_per_epoch = 1 # Run at least one step
    if validation_steps == 0:
        print("Warning: validation_steps is 0. Check batch size vs validation samples.")
        validation_steps = 1 # Run at least one step


    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # --- Save Final Model ---
    model.save("emotion_recognition_cnn_rnn_model_final.h5")
    print("Final model saved to emotion_recognition_cnn_rnn_model_final.h5")

    # --- Evaluate Model ---
    # Ensure test generator is not empty
    if test_generator.samples > 0:
        test_steps = test_generator.samples // BATCH_SIZE
        if test_steps == 0:
             print("Warning: test_steps is 0. Check batch size vs test samples.")
             test_steps = 1 # Evaluate at least one step
        print(f"Evaluating on test data with {test_steps} steps...")
        loss, accuracy = model.evaluate(test_generator, steps=test_steps, verbose=1)
        print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    else:
        print("Test generator is empty. Skipping evaluation.")

else:
    print("Training or validation generator is empty. Skipping training and evaluation.")