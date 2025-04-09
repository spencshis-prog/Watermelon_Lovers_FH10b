import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import math

# Input directory
dataset_dir = "/Users/ericmendoza/Watermelon/backend/watermelon_acoustic/input/vis_qilin/picture"

# debugging
if not os.path.exists(dataset_dir):
    raise ValueError(f"Dataset directory does not exist: {dataset_dir}")

files = os.listdir(dataset_dir)
print("Files in dataset directory:", files)  # Debugging output

jpg_files = []
labels = []

for file_name in files:
    if file_name.endswith(".jpg"):
        image_path = os.path.join(dataset_dir, file_name)
        
        # Extract brix value from the filename
        # The filename format is: <watermelonID>_<brix>_<index>.jpg
        brix = float(file_name.split("_")[1])  # Extract the brix value

        jpg_files.append(image_path)
        labels.append(brix)  # Append the brix value as the label

if not jpg_files:
    raise ValueError("No image files found in the dataset directory.")

dataset = tf.data.Dataset.from_tensor_slices((jpg_files, labels))
dataset = dataset.shuffle(buffer_size=1000)

train_size = int(0.8 * len(jpg_files))
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

batch_size = 4  # Batch size kept at 4 for stability

# Data Augmentation (simplified to just random horizontal flip)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
])

def load_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (1080, 1080))
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image

# Apply data augmentation to training dataset
train_dataset = train_dataset.map(lambda jpg, label: (data_augmentation(load_image(jpg)), label))
val_dataset = val_dataset.map(lambda jpg, label: (load_image(jpg), label))

train_dataset = train_dataset.batch(batch_size).repeat()
val_dataset = val_dataset.batch(batch_size)

resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(1080, 1080, 3))

for layer in resnet.layers:                         # Freeze all layers of ResNet50
    layer.trainable = False


image_input = Input(shape=(1080, 1080, 3))          # Preprocessing: Resizing

resnet_output = resnet(image_input)
resnet_output = GlobalAveragePooling2D()(resnet_output)


output = Dense(64, activation='relu')(resnet_output) # Add fully connected layers for the final output
output = Dense(1, activation='linear')(output)


model = Model(inputs=image_input, outputs=output)
model.compile(optimizer='adam', loss=MeanSquaredError())
model.fit(train_dataset, epochs=20, validation_data=val_dataset, steps_per_epoch=train_size // batch_size)

y_true = []
y_pred = []

for jpg, label in val_dataset:
    y_true.extend(label.numpy())
    pred = model.predict(jpg)
    y_pred.extend(pred.flatten())

mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.4f}")

mse = mean_squared_error(y_true, y_pred)
print(f"MSE: {mse:.4f}")

rmse = math.sqrt(mse)
print(f"RMSE: {rmse:.4f}")

mape = mean_absolute_percentage_error(y_true, y_pred)
print(f"MAPE: {mape:.4f}")

r2 = r2_score(y_true, y_pred)
print(f"R2: {r2:.4f}")

# Save the model
model.save("vis_model.keras")
