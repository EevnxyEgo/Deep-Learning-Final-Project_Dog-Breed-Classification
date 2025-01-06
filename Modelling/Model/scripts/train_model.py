import tensorflow as tf
import os
import matplotlib.pyplot as plt
import json

IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 25

# Load dataset
image_dir = 'dataset/Images'
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    image_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='training'
)

val_generator = datagen.flow_from_directory(
    image_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='validation'
)

# Save class indices to a JSON file
output_dir = 'models'
os.makedirs(output_dir, exist_ok=True)  # Membuat folder jika belum ada
with open(os.path.join(output_dir, 'class_indices.json'), 'w') as f:
    json.dump(train_generator.class_indices, f)

# Define the model
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=len(val_generator)
)

# Menampilkan label yang ada di dataset
label_map = train_generator.class_indices
print("Label yang ada di dataset:", label_map)

# Plot Akurasi dan Loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))

# Plot Training & Validation Accuracy
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.title('Training and Validation Accuracy')

# Plot Training & Validation Loss
plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')

# Menyimpan dalam format SavedModel di folder saved_models
model_path = "saved_models/fine_tuned_model"
try:
    model.save(model_path)  # Format SavedModel
    print(f"Model berhasil disimpan di: {model_path}")
except Exception as e:
    print("Gagal menyimpan model di folder 'saved_models':", e)

# Simpan grafik ke dalam folder 'models'
plt.savefig(os.path.join(output_dir, 'training_validation_plot.png'))

# Menampilkan plot
plt.show()