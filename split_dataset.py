import pandas as pd

# Parameters ---------------
annotations_file = './annotations_new_3s.csv' # Annotation file produced after running 'convert_avi_to_gif.py'

# Output CSV filenames for the training, validation and testing datasets 
# that will be generated in this script
annotations_train = './annotations_train_3s.csv'
annotations_val = './annotations_val_3s.csv'
annotations_test = './annotations_test_3s.csv'

test_samples = 0.1 # i.e. 10% for testing
val_samples = 0.1 # i.e. 10 % for validation
# --------------------------

# Non-augmented dataset ----------------------
data = pd.read_csv(annotations_file)
data = data.sample(frac=1).reset_index(drop=True)

data_len = len(data)
test_samples = round(0.1 * data_len)
val_samples = round(0.1 * data_len)

# Testing datasets --------------------------
# Test samples
test_data = data.iloc[-test_samples:, :]

test_data.to_csv(annotations_test)

# Training datasets -------------------------
train_data = data.iloc[:-test_samples, :]

# Val samples
val_data = train_data.iloc[-val_samples:, :]

val_data.to_csv(annotations_val)

# Train samples
train_data = train_data.iloc[:-val_samples, :]
train_data.to_csv(annotations_train)

print(tuple(map(len, (train_data, val_data, test_data))))