import tensorflow as tf
import numpy as np
from skimage.filters import sobel
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt


def ext_sobel_features(image):
    sobel_image = sobel(image)
    return sobel_image


test_batches = ImageDataGenerator(
        rescale=1. / 255,
        # This is the name of the preprocessing function defined above which extracts the sobel features
        preprocessing_function=ext_sobel_features
    )

TESTING_IMAGES_DIRECTORY = '../data/Spectrogram/Testing'

# Create training data from train_bathes
testing_data = test_batches.flow_from_directory(
    directory=TESTING_IMAGES_DIRECTORY,
    target_size=(128, 128),
)

X = []
y = []

count = 0

# Extract features and labels from from testing data iterator
for features, labels in testing_data:
    X.append(features)
    y.append(labels)
    count += 1
    if count == 13:
        break


X = np.array(X)
y = np.array(y)

X = np.reshape(X, (-1, 128, 128, 3))
y = np.reshape(y, (-1, 3))
y_one_hot = y
y = np.argmax(y, axis=1)

# Load the model
model = tf.keras.models.load_model('model_cnn')

# Make prediction
y_pred = model.predict(X, batch_size=32, verbose=1)
# Get the boolean labels from prediction
y_pred_bool = np.argmax(y_pred, axis=1)

print(classification_report(y, y_pred_bool, output_dict=True))

auc_score = roc_auc_score(y_one_hot.ravel(), y_pred.ravel())
print("Auc Score - ", auc_score)

# Draw roc curve
fpr, tpr, thresholds = roc_curve(y_one_hot.ravel(), y_pred.ravel())

plt.plot(fpr, tpr, color="darkorange", label="ROC curve (area = %0.2f)" % auc_score)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.axis([0, 1, 0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

