import pandas as pd

# Parameters ---------------
annotations_file = './annotations_gif.csv' # Annotation file produced after running 'convert_avi_to_gif.py'
test_samples = 0.1 # i.e. 10% for testing
val_samples = 0.1 # i.e. 10 % for validation
# --------------------------

# Non-augmented dataset ----------------------
data = pd.read_csv(annotations_file)
data = data.sample(frac=1).reset_index(drop=True)

data_len = len(data)
test_samples = 0.1 * data_len
val_samples = 0.1 * data_len

# Testing datasets --------------------------
# Test samples
test_data = data.iloc[-test_samples:, :]

test_data.to_csv('./annotations_test.csv')

# Training datasets -------------------------
train_data = data.iloc[:-test_samples, :]

# Val samples
val_data = train_data.iloc[-val_samples:, :]

val_data.to_csv('./annotations_val.csv')

# Train samples
train_data = train_data.iloc[:-val_samples, :]
train_data.to_csv('./annotations_train.csv')

print(tuple(map(len, (train_data, val_data, test_data))))