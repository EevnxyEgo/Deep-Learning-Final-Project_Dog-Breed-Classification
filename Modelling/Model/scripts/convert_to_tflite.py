import tensorflow as tf

# Load the saved model
saved_model_dir = 'saved_models/fine_tuned_model'

# Convert the model to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

# Save the TFLite model
with open('saved_models/fine_tuned_model/model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model berhasil dikonversi ke format TFLite dan disimpan sebagai model.tflite.")
