import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset/sample_data')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

# Data loader
IMG_SIZE = (224, 224)
BATCH = 32
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_gen = datagen.flow_from_directory(DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH, subset='training')
val_gen = datagen.flow_from_directory(DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH, subset='validation')

# Automatically detect class names
classes = list(train_gen.class_indices.keys())

# Build model
base = tf.keras.applications.MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights=None)
model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(len(classes), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_gen, validation_data=val_gen, epochs=3)

# Save model and labels
model.save(os.path.join(MODEL_DIR, 'rice_model.h5'))
with open(os.path.join(MODEL_DIR, 'rice_labels.txt'), 'w') as f:
    for label in classes:
        f.write(label + '\n')

print("✅ Model trained and saved.")
