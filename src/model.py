from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def build_transfer_learning_model(learning_rate=0.00001):
    base_model = VGG16(include_top=False, input_shape=(150, 150, 3))
    base_model.load_weights('path_to_VGG16_weights')  # Path to the downloaded weights
    
    # Unfreeze the top layers of the base model
    for layer in base_model.layers[-4:]:
        layer.trainable = True

    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(6, activation='softmax')  # Assuming 6 classes for chess pieces
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
