import numpy as np
import tensorflow as tf
import kerastuner as kt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, MultiHeadAttention,BatchNormalization
from tensorflow.keras import Input
import datetime
from extractor import get_data_from_file
IMPORT_COUNT = 600000
TEST_COUNT = 20000
RNG_NAME="xorshift128plus"
"""
Control how many outputs back the model should look.
If you are not sure, I would suggest
(Size of the RNG state in bits)/(Bits of output from the RNG).
If your RNG produces low entropy output, you
may need more past data-but I have no tested this.
"""
X,y=get_data_from_file(RNG_NAME+'.rng',IMPORT_COUNT,4)
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
def transformer(layer,num_heads):
	queries = Dense(X.shape[1])
	mha = MultiHeadAttention(num_heads=num_heads,key_dim=X.shape[1]//2,attention_axes=1)
	return mha(inputs,queries)
def build_model(hp):
	heads = hp.Int("heads",4,12,sampling="log")
	#key_dim = hp.Int("key_dim",X.shape[1]//2,X.shape[1],sampling="log")
	inputs = Input(shape=(X.shape[1],))
	queries = Dense(X.shape[1],activation="relu")(inputs)
	mha = MultiHeadAttention(num_heads=heads,key_dim=X.shape[1]//2,attention_axes=1)
	layer = mha(inputs,queries)
	layer2 =Dense(X.shape[1])
	mha = MultiHeadAttention(num_heads=heads,key_dim=key_dim,attention_axes=1)
	layer = Dense(X.shape[1],activation="relu")(layer)
	layer = Dense(X.shape[1],activation="relu")(layer)
	output= Dense(y.shape[1])(layer)
	model = keras.Model(inputs=inputs,outputs=output,name="fuckler")
	opt = keras.optimizers.Nadam(
		learning_rate=hp.Float("learning_rate", 10**(-5),10**-2,sampling="log"),
		epsilon=1e-8,
		beta_1=.9,
		beta_2=.9,
	)
	model.compile(optimizer=opt,loss=tf.keras.losses.MSE,metrics=['binary_accuracy'])
	model.summary()
	return model
X_train_short= X_train[:600000]
y_train_short= y_train[:600000]
log_dir = "logs/"+RNG_NAME+"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,write_graph=False,profile_batch=0)
tuner = kt.tuners.bayesian.BayesianOptimization(build_model,'binary_accuracy',100,project_name="hp_search_"+RNG_NAME)
tuner.search(X_train_short, y_train_short,batch_size=256,verbose=0,epochs=100,validation_data=(X_test,y_test),callbacks=[tensorboard_callback])
tuner.results_summary()
best_hps = tuner.get_best_hyperparameters(num_trials = 5)[-1]
model = tuner.hypermodel.build(best_hps)
model.summary()
"""
#define CB
model.load_weights(RNG_NAME+"_"+"DONE")
for i in range(20):
	model.fit(X_train,y_train,epochs=5, batch_size=256,callbacks=[tensorboard_callback,],verbose=0)
	results = model.evaluate(X_test, y_test, batch_size=128)
	print("test loss: %f, test acc: %s" % tuple(results))	
	model.save_weights(RNG_NAME+"_"+str(i))
model.save_weights(RNG_NAME+"_"+"DONE_2")
results = model.evaluate(X_test, y_test, batch_size=128)
print("test loss: %f, test acc: %s" % tuple(results))
"""