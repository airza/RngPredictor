import numpy as np
import tensorflow as tf
import kerastuner as kt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import datetime

IMPORT_COUNT = 3990000
TEST_COUNT = 20000

"""
Control how many outputs back the model should look.
If you are not sure, I would suggest
(Size of the RNG state in bits)/(Bits of output from the RNG).
If your RNG produces low entropy output, you
may need more past data-but I have no tested this.
"""
PREVIOUS_TIMESTEP_COUNT = 8

TOTAL_DATA_NUM = IMPORT_COUNT-PREVIOUS_TIMESTEP_COUNT

def strided(a, L):
	#I don't recommend touching this code.
	shp = a.shape
	s  = a.strides
	nd0 = shp[0]-L+1
	shp_in = (nd0,L)+shp[1:]
	strd_in = (s[0],) + s
	return np.lib.stride_tricks.as_strided(a, shape=shp_in, strides=strd_in)

RNG_OUTPUT_FILENAME="xorshift128TRUNCATED.txt"
df = np.genfromtxt(RNG_OUTPUT_FILENAME,delimiter='\n',dtype='uint64')[:IMPORT_COUNT]
#calculates how many bits are in the output.
BIT_WIDTH=np.ceil(np.log2(np.amax(df))).astype(int)

#We need to transform this from a list of numbers to a list of those numbers
#binary representation, and then slice it into 1 observed output and
#PREVIOUS_TIMESTEP_COUNT previous outputs. I also do not recommend touching this code
#as it is quite fucky.
df_as_bits =(df[:,None] & (1 << np.arange(BIT_WIDTH,dtype='uint64')) > 0).astype(int)
df_as_frames = strided(df_as_bits,PREVIOUS_TIMESTEP_COUNT+1)

#normal shuffle doesn't work for some reason, oh well
indicies = np.arange(TOTAL_DATA_NUM,dtype='uint64')
np.random.shuffle(indicies)
df_as_frames=df_as_frames[indicies]

#Now is the correct time if you want to narrow the RNG prediction to specific bits
#which is probably a much easier problem (e.g.)
# y = df_as_frames[:,-1,:][:,_INSERT_MASK_HERE_]
y = df_as_frames[:,-1,:]
X = df_as_frames[:,:-1,]

"""
Default model assumes that you want to use an LSTM to learn underlying
state about the representation. There is some reason to beleive that
you could just input all of the bits as a flat array; if so, use
np.reshape(X,[TOTAL_DATA_NUM,-1])
so for example x goes from a (TOTAL_DATA_NUM,32,4) tensor to a
(TOTAL_DATA_NUM,32*4) tensor
"""

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
def build_model(hp):
	LOSS="mse"
	model = Sequential()
	width = hp.Int("network_width",64,512,sampling="log")
	model.add(LSTM(units=width*2,activation='relu',input_shape=(PREVIOUS_TIMESTEP_COUNT,BIT_WIDTH,),return_sequences=False,))
	for depth in range(hp.Int("network_depth", 4,8)):
		model.add(Dense(width,activation='relu'))
	model.add(Dense(y.shape[1],activation='sigmoid'))
	opt = keras.optimizers.Nadam(
		learning_rate=hp.Float("learning_rate", 10**(-2), 10**(-6),sampling="log"),
		epsilon=1e-7,
		beta_1=.9,
		beta_2=.9,
		)
	model.compile(optimizer=opt, loss=LOSS,metrics=['binary_accuracy'])
	return model
X_train_short= X_train[:600000]
y_train_short= y_train[:600000]
#define CBs
stopEarly = tf.keras.callbacks.EarlyStopping(
	monitor='binary_accuracy', min_delta=.001, patience=10, verbose=0, mode='auto', restore_best_weights=False
)
log_dir = "hyperparameters_for_truncated/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,profile_batch=0)

tuner = kt.tuners.bayesian.BayesianOptimization(build_model,'binary_accuracy',200,project_name="bayesfortruncatedset")
tuner.search(X_train_short, y_train_short,batch_size=256,verbose=0,epochs=50,validation_data=(X_test,y_test),callbacks=[stopEarly,tensorboard_callback])
tuner.results_summary()
best_hps = tuner.get_best_hyperparameters(num_trials = 5)[4]
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
#reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='binary_accuracy', factor=0.5,min_delta=.005,patience=5)
model.load_weights("weights_small")
for i in range(30):
	model.fit(X_train_short, y_train_short, epochs=200, batch_size=512,callbacks=[tensorboard_callback],verbose=0)
	results = model.evaluate(X_test, y_test, batch_size=128)
	print("test loss: %f, test acc: %s" % tuple(results))
	model.fit(X_train, y_train, epochs=2, batch_size=512,callbacks=[tensorboard_callback,],verbose=0)
	results = model.evaluate(X_test, y_test, batch_size=128)
	print("test loss: %f, test acc: %s" % tuple(results))
model.save_weights("weights_small2")