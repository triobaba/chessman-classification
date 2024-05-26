import os
import certifi
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.data_preprocessing import load_and_preprocess_data
from src.model import build_transfer_learning_model

# Set SSL certificate path
os.environ['SSL_CERT_FILE'] = certifi.where()

data_dir = '../data/Chessman'
train_data, val_data = load_and_preprocess_data(data_dir)

model = build_transfer_learning_model()

early_stopping = EarlyStopping(monitor='val_loss', patience=10)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

history = model.fit(
    train_data,
    epochs=30,
    validation_data=val_data,
    callbacks=[early_stopping, checkpoint]
)