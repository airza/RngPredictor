import tensorflow as tf
from extractor import get_data_from_file
IMPORT_COUNT = 1000000
TEST_COUNT = 1000000
RNG_NAME="xorshift128"

X,y=get_data_from_file(RNG_NAME+'_extra.rng',IMPORT_COUNT,4)

X_train = X[TEST_COUNT:]
X_test = X[:TEST_COUNT]
y_train = y[TEST_COUNT:]
y_test = y[:TEST_COUNT]
model = tf.keras.models.load_model(RNG_NAME)
results = model.evaluate(X_test, y_test, batch_size=128)
print("test loss: %f, test acc: %s" % tuple(results))
