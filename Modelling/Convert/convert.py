import tensorflow as tf

# Load the SavedModel directory
saved_model_path = "./saved_model"


# Save the model in the .keras format
keras_model_path = "./model.keras"

# Load the SavedModel as a TensorFlow object
loaded_model = tf.saved_model.load(saved_model_path)

# Extract the serving function (inference function)
infer = loaded_model.signatures.get("serving_default")
if infer is None:
    raise ValueError("The SavedModel does not have a 'serving_default' signature. "
                     "Ensure it was exported for inference.")

# Create a Keras model using the input and output of the serving function
input_signature = infer.structured_input_signature[1]
inputs = [
    tf.keras.Input(shape=tensor.shape[1:], dtype=tensor.dtype)
    for tensor in input_signature.values()
]
outputs = infer(*inputs)
keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Save the Keras model in the .keras format
keras_model.save(keras_model_path, save_format="keras")