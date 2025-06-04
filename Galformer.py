# %% [markdown]
# # Updated Galformer Model Training on CPU
#
# This notebook trains an improved Transformer-based model (Galformer) for stock price prediction.
# The improvements include data normalization, model architecture enhancements, and training optimizations.

# %%
# Install required packages if necessary
# ! pip install tensorflow==2.12

# %%
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, MultiHeadAttention, LayerNormalization
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras import regularizers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# %% [markdown]
# ## Constants and Parameters

# %%
sequence_length = 60  # Should match the value used in data preparation
prediction_horizon = 25
epochs = 100
batch_size = 512  # Increased batch size
print('Batch size:', batch_size)

# %% [markdown]
# ## Define Feature Description

# %%
# Define feature description for parsing TFRecord files
feature_description = {
    'feature': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.string),
}

# %% [markdown]
# ## Get the List of TFRecord Files

# %%
# List all TFRecord files
tfrecord_files = glob.glob('tfrecords_data/*.tfrecord')
print(f"Found {len(tfrecord_files)} TFRecord files.")

# Create a dataset from the list of TFRecord files
raw_dataset = tf.data.TFRecordDataset(tfrecord_files, compression_type='GZIP')

# %% [markdown]
# ## Determine `num_features` from the Dataset

# %%
# Extract one example to determine num_features
for raw_record in raw_dataset.take(1):
    example = tf.io.parse_single_example(raw_record, feature_description)
    feature = tf.io.parse_tensor(example['feature'], out_type=tf.float32)
    label = tf.io.parse_tensor(example['label'], out_type=tf.float32)
    sequence_length = feature.shape[0]
    num_features = feature.shape[1]
    prediction_horizon = label.shape[0]
    print(f"Sequence Length: {sequence_length}, Num Features: {num_features}, Prediction Horizon: {prediction_horizon}")
    break

# %% [markdown]
# ## Function to Parse and Normalize TFRecord Examples

# %%
def parse_tfrecord_fn(example_proto):
    example = tf.io.parse_single_example(example_proto, feature_description)

    feature = tf.io.parse_tensor(example['feature'], out_type=tf.float32)
    label = tf.io.parse_tensor(example['label'], out_type=tf.float32)

    # Set shapes for feature and label
    feature.set_shape([sequence_length, num_features])
    label.set_shape([prediction_horizon])

    # Normalize features (z-score standardization)
    feature_mean = tf.reduce_mean(feature, axis=0, keepdims=True)
    feature_std = tf.math.reduce_std(feature, axis=0, keepdims=True) + 1e-6  # Add epsilon to avoid division by zero
    feature = (feature - feature_mean) / feature_std

    # Normalize labels (z-score standardization)
    label_mean = tf.reduce_mean(label)
    label_std = tf.math.reduce_std(label) + 1e-6
    label = (label - label_mean) / label_std

    # Store label mean and std for inverse transformation (if needed)
    label_info = tf.stack([label_mean, label_std])  # Shape: (2,)

    return feature, (label, label_info)

# %% [markdown]
# ## Create `tf.data.Dataset` from TFRecord Files

# %%
# Parse the serialized data in the TFRecord files
parsed_dataset = raw_dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)

# Determine the total dataset size
total_dataset_size = sum(1 for _ in parsed_dataset)
print(f"Total number of samples in dataset: {total_dataset_size}")

# Reset the parsed_dataset iterator after counting
parsed_dataset = raw_dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)

# %% [markdown]
# ## Split Dataset into Training and Testing Sets

# %%
# Shuffle and split the dataset
train_size = int(0.8 * total_dataset_size)
test_size = total_dataset_size - train_size

# Shuffle the entire dataset and split
parsed_dataset = parsed_dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=False)

train_dataset = parsed_dataset.take(train_size)
test_dataset = parsed_dataset.skip(train_size)

# Batch and prefetch the datasets
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

print(f"Training samples: {train_size}, Testing samples: {test_size}")

# %% [markdown]
# ## Add Positional Encoding Function

# %%
def positional_encoding(sequence_length, d_model):
    position = tf.range(sequence_length, dtype=tf.float32)[:, tf.newaxis]
    i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
    angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    angle_rads = position * angle_rates
    sin_terms = tf.sin(angle_rads[:, 0::2])
    cos_terms = tf.cos(angle_rads[:, 1::2])
    pos_encoding = tf.concat([sin_terms, cos_terms], axis=-1)
    return pos_encoding  # Shape: (sequence_length, d_model)

# %% [markdown]
# ## Build and Compile the Enhanced Galformer Model

# %%
# Initialize the Galformer model
input_shape = (sequence_length, num_features)  # (sequence_length, num_features)
output_length = prediction_horizon  # prediction_horizon

def build_galformer_model(input_shape, output_length, num_layers=3, dff=512, num_heads=8):
    inputs = Input(shape=input_shape)  # input_shape = (sequence_length, num_features)

    # Create positional encodings
    pe = positional_encoding(input_shape[0], input_shape[1])
    pe = tf.expand_dims(pe, axis=0)  # Shape: (1, sequence_length, num_features)

    # Add positional encoding to inputs
    x = inputs + pe  # Broadcasting, pe shape is (1, sequence_length, num_features)

    # Stack multiple Transformer Encoder Layers
    for _ in range(num_layers):
        # Multi-Head Attention
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[1], dropout=0.1)(x, x)
        attn_output = Dropout(0.2)(attn_output)  # Increased dropout
        out1 = LayerNormalization(epsilon=1e-6)(attn_output + x)

        # Feed Forward Network with L2 Regularization
        ffn_output = Dense(dff, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(out1)
        ffn_output = Dense(input_shape[1], kernel_regularizer=regularizers.l2(1e-4))(ffn_output)
        ffn_output = Dropout(0.2)(ffn_output)  # Increased dropout
        x = LayerNormalization(epsilon=1e-6)(ffn_output + out1)

    # Flatten and Output Layer
    x = Flatten()(x)
    outputs = Dense(output_length, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)
    # Define learning rate schedule
    lr_schedule = ExponentialDecay(
        initial_learning_rate=1e-4,  # Reduced learning rate
        decay_steps=10000,
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

galformer_model = build_galformer_model(input_shape, output_length)

# %% [markdown]
# ## Training Callbacks

# %%
# Increased early stopping patience
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_scheduler = LearningRateScheduler(scheduler)

callbacks = [early_stopping, lr_scheduler]

# %% [markdown]
# ## Train the Enhanced Galformer Model

# %%
history_galformer = galformer_model.fit(
    train_dataset,
    epochs=epochs,  # Increased epochs
    validation_data=test_dataset,
    callbacks=callbacks,
    verbose=1
)

# %% [markdown]
# ## Plot Training & Validation Loss Values

# %%
plt.figure(figsize=(10, 6))
plt.plot(history_galformer.history['loss'], label='Train Loss')
plt.plot(history_galformer.history['val_loss'], label='Validation Loss')
plt.title('Enhanced Galformer Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('train_validation_loss.png')
plt.close()  # Close the plot to free memory

# %% [markdown]
# ## Evaluate the Enhanced Galformer Model

# %%
test_loss_galformer, test_mae_galformer = galformer_model.evaluate(test_dataset, verbose=1)
print(f"Enhanced Galformer Model - Test Loss: {test_loss_galformer:.4f}, Test MAE: {test_mae_galformer:.4f}")

# %% [markdown]
# ## Predict on Test Data

# %%
# Collect features, labels, and label info from test_dataset
X_test_list = []
y_test_list = []
label_info_list = []

for features, (labels, label_info) in test_dataset:
    X_test_list.append(features.numpy())
    y_test_list.append(labels.numpy())
    label_info_list.append(label_info.numpy())

# Concatenate lists to form arrays
X_test = np.concatenate(X_test_list, axis=0)
y_test = np.concatenate(y_test_list, axis=0)
label_info = np.concatenate(label_info_list, axis=0)  # Shape: (num_samples, 2)

# Predict on test data
def add_positional_encoding(inputs):
    pe = positional_encoding(inputs.shape[1], inputs.shape[2])
    pe = tf.expand_dims(pe, axis=0)  # Shape: (1, sequence_length, num_features)
    inputs_with_pe = inputs + pe.numpy()
    return inputs_with_pe

X_test_with_pe = add_positional_encoding(X_test)
y_pred_galformer = galformer_model.predict(X_test_with_pe)

# Denormalize predictions and actual labels
label_mean = label_info[:, 0]  # Shape: (num_samples,)
label_std = label_info[:, 1]   # Shape: (num_samples,)

# Reshape label_mean and label_std to (num_samples, 1) to match the predictions
label_mean = label_mean.reshape(-1, 1)
label_std = label_std.reshape(-1, 1)

# Denormalize predictions and actual labels
y_pred_galformer_denorm = y_pred_galformer * label_std + label_mean
y_test_denorm = y_test * label_std + label_mean

# Visualize predictions for the first test sample
plt.figure(figsize=(10, 6))

# Plot Actual Prices for the first test sample
plt.plot(range(1, output_length + 1), y_test_denorm[0], label="Actual Prices", marker='o')

# Plot Predicted Prices for the first test sample (Galformer)
plt.plot(range(1, output_length + 1), y_pred_galformer_denorm[0], label="Predicted Prices (Galformer)", marker='x')

plt.title("Actual vs Predicted Prices (First Test Sample - Enhanced Galformer)")
plt.xlabel("Days Ahead")
plt.ylabel("Price")
plt.legend()
plt.savefig('actual_vs_predicted_prices.png')  # Save the plot
plt.close()  # Close the plot to free memory

# %% [markdown]
# ## Save the Enhanced Galformer Model

# %%
galformer_model.save('enhanced_stock_galformer_model.keras')
print("Enhanced Galformer Model has been saved to 'enhanced_stock_galformer_model.keras'.")

# %% [markdown]
# ## Inference with New Data

# %% [markdown]
# ### Load Company Data for Inference

# %%
def load_company_data():
    all_dfs = {}
    parquet_files = glob.glob('df_*.parquet')
    for file in parquet_files:
        key = file.split('.')[0]  # e.g., 'df_AAPL'
        df = pd.read_parquet(file)
        all_dfs[key] = df
    return all_dfs

all_dfs = load_company_data()

# %% [markdown]
# ### Prepare Data for Inference

# %%
def prepare_inference_data(company_df, sequence_length=60):
    """
    Prepare input data for inference for a single company.
    Args:
        company_df (DataFrame): The DataFrame for a specific company.
        sequence_length (int): The number of past days to consider as input.

    Returns:
        numpy array: The input data ready for prediction.
    """
    # Ensure data is sorted by date
    company_df = company_df.sort_index()

    # Select relevant input features (exclude targets)
    input_features = company_df.filter(regex="^(?!.*target).*").values

    # Take the last `sequence_length` days as input for prediction
    if len(input_features) >= sequence_length:
        input_sequence = input_features[-sequence_length:]
        # Normalize features using the same method as during training
        feature_mean = np.mean(input_sequence, axis=0, keepdims=True)
        feature_std = np.std(input_sequence, axis=0, keepdims=True) + 1e-6
        input_sequence = (input_sequence - feature_mean) / feature_std
        # Ensure the input sequence has the correct shape
        input_sequence = np.expand_dims(input_sequence, axis=0)  # Add batch dimension
        return input_sequence.astype(np.float32), feature_mean, feature_std  # Return mean and std for future use
    else:
        raise ValueError("Insufficient data for inference (less than sequence length).")

# %% [markdown]
# ### Make Predictions for Each Company

# %%
def get_galformer_predictions_for_company(company_df, galformer_model, sequence_length=60):
    """
    Get Galformer predictions for a single company using the trained model.
    Args:
        company_df (DataFrame): DataFrame of the company.
        galformer_model: Trained Galformer model.
        sequence_length (int): Number of past days to consider as input.

    Returns:
        numpy array: Predicted prices.
    """
    try:
        # Prepare data for inference
        input_data, _, _ = prepare_inference_data(company_df, sequence_length=sequence_length)

        # Add positional encoding to inference data
        pe = positional_encoding(input_data.shape[1], input_data.shape[2])
        pe = tf.expand_dims(pe, axis=0)  # Shape: (1, sequence_length, num_features)
        input_data_with_pe = input_data + pe.numpy()

        # Make predictions with Galformer
        pred_galformer = galformer_model.predict(input_data_with_pe)
        # Since labels were normalized during training per sample, and during inference we don't have label mean and std,
        # these predictions are in standardized form and cannot be directly denormalized.
        # If you have global label_mean and label_std from training data, use them here to denormalize.
        # For now, we will return the standardized predictions.
        return pred_galformer.flatten()
    except ValueError as e:
        print(f"Skipping due to error: {e}")
        return None

# %% [markdown]
# ### Load the Trained Enhanced Model

# %%
# Load the trained Enhanced Galformer model
galformer_model = tf.keras.models.load_model('enhanced_stock_galformer_model.keras')

# %% [markdown]
# ### Make Predictions for All Companies

# %%
def get_galformer_predictions_for_all_companies(all_dfs, galformer_model, sequence_length=60):
    """
    Get Galformer predictions for all companies.
    """
    all_predictions = {}
    for company_key, company_df in all_dfs.items():
        print(f"Processing {company_key}...")
        predictions = get_galformer_predictions_for_company(company_df, galformer_model, sequence_length=sequence_length)
        if predictions is not None:
            all_predictions[company_key] = predictions
        else:
            print(f"Predictions not available for {company_key}.")
    return all_predictions

# %%
# Get predictions for all companies
all_company_predictions = get_galformer_predictions_for_all_companies(all_dfs, galformer_model, sequence_length=60)

# %% [markdown]
# ### Save Predictions

# %%
def predictions_to_dataframe(predictions_dict):
    """
    Convert predictions dictionary to a DataFrame.
    """
    records = []
    for company_key, pred_values in predictions_dict.items():
        for day_ahead, value in enumerate(pred_values, start=1):
            records.append({
                'Company': company_key.replace('df_', ''),
                'Day_Ahead': day_ahead,
                'Predicted_Price_Standardized': value  # Note that this is in standardized form
            })
    return pd.DataFrame(records)

predictions_df = predictions_to_dataframe(all_company_predictions)
predictions_df.head()

# %%
# Save the predictions DataFrame to a CSV file
predictions_df.to_csv('enhanced_galformer_stock_price_predictions.csv', index=False)
print("Enhanced Galformer Predictions have been saved to 'enhanced_galformer_stock_price_predictions.csv'.")
