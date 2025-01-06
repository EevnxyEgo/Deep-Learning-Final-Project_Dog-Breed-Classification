import tensorflow as tf
from tensorflow.keras import layers, Model

class ConvBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, padding='same'):
        super().__init__()
        self.conv = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False
        )
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()  # Tidak ada inplace di TensorFlow, ini setara dengan ReLU(inplace=False) di PyTorch
        
    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class ResidualBlock(layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = ConvBlock(filters)
        self.conv2 = ConvBlock(filters)
        self.relu = layers.ReLU()  # ReLU tetap tanpa inplace karena TensorFlow tidak memerlukan parameter inplace
        
    def call(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual  # Pindahkan penambahan residual sebelum ReLU, sesuai dengan PyTorch
        return self.relu(out)

class CustomCNN(Model):
    def __init__(self, num_classes=10):
        super().__init__()

        # Feet: Initial feature extraction
        self.feet = tf.keras.Sequential([
            ConvBlock(32, kernel_size=7, strides=2, padding='same'),
            layers.MaxPool2D(pool_size=3, strides=2, padding='same')  # Padding diset sama seperti di PyTorch
        ])
        
        # Body: Feature processing
        self.body = tf.keras.Sequential([
            # Stage 1
            ConvBlock(64),
            ResidualBlock(64),
            layers.MaxPool2D(pool_size=2, padding='same'),  # Padding diset sama seperti di PyTorch
            layers.Dropout(0.1),
            
            # Stage 2
            ConvBlock(128),
            ResidualBlock(128),
            layers.MaxPool2D(pool_size=2, padding='same'),  # Padding diset sama seperti di PyTorch
            layers.Dropout(0.1),
            
            # Stage 3
            ConvBlock(256),
            ResidualBlock(256),
            layers.MaxPool2D(pool_size=2, padding='same'),  # Padding diset sama seperti di PyTorch
            layers.Dropout(0.1)
        ])
        
        # Head: Classification
        self.head = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(),  # Global pooling seperti di PyTorch
            layers.Flatten(),
            layers.Dense(num_classes)
        ])

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, layers.Conv2D):
                # Inisialisasi layer Conv2D dengan kaiming normal seperti di PyTorch
                fan_in = layer.kernel_size[0] * layer.kernel_size[1] * layer.input_shape[-1]
                std = tf.sqrt(2.0 / fan_in)
                layer.kernel.assign(tf.random.normal(layer.kernel.shape, mean=0.0, stddev=std))
            elif isinstance(layer, layers.BatchNormalization):
                # Inisialisasi gamma dan beta untuk BatchNorm seperti di PyTorch
                layer.gamma.assign(tf.ones_like(layer.gamma))
                layer.beta.assign(tf.zeros_like(layer.beta))
            elif isinstance(layer, layers.Dense):
                # Inisialisasi Dense dengan distribusi normal standar (std=0.01)
                std = 0.01
                layer.kernel.assign(tf.random.normal(layer.kernel.shape, mean=0.0, stddev=std))
                layer.bias.assign(tf.zeros_like(layer.bias))

    def call(self, x):
        x = self.feet(x)
        x = self.body(x)
        return self.head(x)
