import numpy as np
import tensorflow as tf
import kerastuner as kt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
import datetime
from extractor import get_data_from_file
IMPORT_COUNT = 2000000
TEST_COUNT = 20000
RNG_NAME="xorshift128"
"""
Control how many outputs back the model should look.
If you are not sure, I would suggest
(Size of the RNG state in bits)/(Bits of output from the RNG).
If your RNG produces low entropy output, you
may need more past data-but I have no tested this.
"""
X,y=get_data_from_file(RNG_NAME+'_extra.rng',IMPORT_COUNT,2)
"""
Default model assumes that you want to use an LSTM to learn underlying
state about the representation. There is some reason to beleive that
you could just input a
ll of the bits as a flat array; if so, use
np.reshape(X,[TOTAL_DATA_NUM,-1])
so for example x goes from a (TOTAL_DATA_NUM,32,4) tensor to a
(TOTAL_DATA_NUM,32*4) tensor
"""
print(X.shape)
X = np.reshape(X,[X.shape[0],-1])
print(X.shape)
X_train = X[TEST_COUNT:]
X_test = X[:TEST_COUNT]
y_train = y[TEST_COUNT:]
y_test = y[:TEST_COUNT]

"""
some notes on my experience with hyperparameters, which are not really
explained in detail in the blog:
Deeper networks don't seem to help (no surprise)
The ability of the model to learn seems to be very sensitive to the learning rate
If you are constrained for compute searching as much of the log learning rate space
as possible is probably the best bang for your bucks
I didn't have much success with non relu activations (vanishing gradient problemos)
and although it would make more sense for the final layer to be constrained to (0,1)
that didn't seem to work very well either.
"""
def fastLoss(y_true,y_pred):
	s = 3*tf.math.abs(y_true-y_pred)
	return tf.math.reduce_logsumexp(s)
def build_model(hp):
	LOSS="mse"
	model = Sequential()
	FAT_WIDTH = hp.Int("fat_width",256,512,sampling="log")
	THIN_WIDTH = 128 #hp.Choice("thin_width",(32,64,128))
	DEPTH = hp.Int("depth",1,4)
	model.add(BatchNormalization(input_shape=(X.shape[1],)))
	for i in range(DEPTH):
		model.add(Dense(FAT_WIDTH,activation="relu",))
		model.add(Dense(FAT_WIDTH,activation="relu",))
		model.add(Dense(THIN_WIDTH,activation="relu",))
	model.add(Dense(y.shape[1]))
	opt = keras.optimizers.Nadam(
		learning_rate=hp.Float("learning_rate", 10**-6,10**-2,sampling="reverse_log"),
		epsilon=1e-8,
		beta_1=.9,
		beta_2=.9,
	)
	model.compile(optimizer=opt, loss=tf.keras.losses.MSE,metrics=['binary_accuracy'])
	model.summary()
	return model
#define CB
stopEarly = tf.keras.callbacks.EarlyStopping(monitor='binary_accuracy', min_delta=.03, patience=20, verbose=0, mode='auto', restore_best_weights=False)
log_dir = "logs/"+RNG_NAME+"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,profile_batch=0)
tuner = kt.tuners.bayesian.BayesianOptimization(build_model,'binary_accuracy',100,project_name="hp_search_"+RNG_NAME)
tuner.search(X_train, y_train,batch_size=256,verbose=0,epochs=50,validation_data=(X_test,y_test),callbacks=[tensorboard_callback,stopEarly])
tuner.results_summary()
best_hps = tuner.get_best_hyperparameters(num_trials = 5)[-1]
model = tuner.hypermodel.build(best_hps)
model.summary()
"""
Annealing process: several cycles on the same model on training on a subset
of the data, then all of the data.  I didn't have any success getting it to
learn from the full dataset at once, but I didn't test it across smaller
batch sizes or different HPs than the smaller subset.  Those would be good
things to do if I had spare cloud compute.  But in the end, this works quite
well and is *much* faster than training all 2 million examples.
"""
for i in range(10):
	model.fit(X_train, y_train, epochs=50, batch_size=512,callbacks=[tensorboard_callback,],verbose=0)
	results = model.evaluate(X_test, y_test, batch_size=128)
	print("test loss: %f, test acc: %s" % tuple(results))
	model.save_weights(RNG_NAME+"_weights_annealing_pass_"+str(i))
model.save_weights(RNG_NAME+"_weights_annealing_pass_DONE")