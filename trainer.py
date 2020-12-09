import numpy as np
import tensorflow as tf
import kerastuner as kt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, MultiHeadAttention,LayerNormalization,BatchNormalization,ReLU
from tensorflow.keras import Input
import datetime
from extractor import get_data_from_file
LOG_STEPS = 5
IMPORT_COUNT = 2**19
TEST_COUNT = 2**14
PREV_COUNT = 2
BIT=0
BATCH_SIZE = 512
RNG_NAME = "xorshift128"
if "xorshift128plus" == RNG_NAME:
	PREV_COUNT = 2
elif "xorshift128" == RNG_NAME:
	PREV_COUNT = 4
LOSS_FUNCTION ='mse'
METRIC_FUNCTION = 'binary_accuracy'
"""
Control how many outputs back the model should look.
If you are not sure, I would suggest
(Size of the RNG state in bits)/(Bits of output from the RNG).
If your RNG produces low entropy output, you
may need more past data-but I have no tested this.
"""

X,y=get_data_from_file(RNG_NAME+'.rng',IMPORT_COUNT,PREV_COUNT,bit=BIT)
"""
Default model assumes that you want to use an LSTM to learn underlying
state about the representation. There is some reason to beleive that
you could just input a
ll of the bits as a flat array; if so, use
np.reshape(X,[TOTAL_DATA_NUM,-1])
so for example x goes from a (TOTAL_DATA_NUM,32,4) tensor to a
(TOTAL_DATA_NUM,32*4) tensor
"""
X = np.reshape(X,[X.shape[0],-1])
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

def resBlock(layer,depth,width,leak=0,dropout=0):
	layer2 = layer
	lrelu = lambda x: tf.keras.activations.relu(x, alpha=leak)
	for i in range(depth):
		layer = Dense(width,activation=lrelu)(layer)
	return tf.keras.layers.ReLU(negative_slope=leak)(layer+l)

def transformer(layer,num_heads,key_dim,leak=0,dropout=0):
	lrelu = lambda x: tf.keras.activations.relu(x, alpha=leak)
	mha = MultiHeadAttention(num_heads=num_heads,key_dim=key_dim,attention_axes=1,dropout=dropout)
	res = mha(layer,layer)
	res = Dense(layer.shape[1])(res)
	residual = tf.keras.layers.ReLU(negative_slope=leak)(layer+res)
	return LayerNormalization(axis=1)(residual)
def build_model(hp):
	feature_dim = 20
	leakiness = hp.Float("leakiness",.001,.3,sampling="log")
	lrelu = lambda x: tf.keras.activations.relu(x, alpha=leakiness)
	t_count = 3
	key_dim = 10
	dropout = hp.Float("dropout",.00001,.1,sampling="log")
	heads = 3
	inputs = Input(shape=(X.shape[1],))
	i2 = inputs*2
	i2 = inputs-1
	t = Dense(feature_dim,activation=lrelu)(i2)
	t = Dense(feature_dim,activation=lrelu)(t)
	for i in range(t_count):
		t = transformer(t,heads,key_dim,leakiness,dropout)
	outputSize = 1 if len(y.shape)==1 else y.shape[1]
	outLayer= Dense(outputSize,activation='tanh')(t)  
	out = outLayer*.5
	output = out+.5
	model =keras.Model(inputs=inputs,outputs=output,name="fuckler")
	beta =hp.Float("beta", .1,.99,sampling="reverse_log"),
	opt = keras.optimizers.Nadam(
		learning_rate=hp.Float("learning_rate", 10**-5,10**-2,sampling="log"),
		epsilon= 1e-9,
		beta_1= beta,
		beta_2= beta
	)
	model.compile(optimizer=opt,loss=LOSS_FUNCTION,metrics=[METRIC_FUNCTION])
	model.summary()
	return model



class StopWhenDoneCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
    	accuracy= logs['binary_accuracy']
    	if accuracy>.99:
    		self.model.stop_training = True
log_dir = "logs/"+RNG_NAME+"/%03d/"%BATCH_SIZE+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,write_graph=False,profile_batch=0)
tuner = kt.tuners.randomsearch.RandomSearch(build_model,METRIC_FUNCTION,50,project_name="hp_"+RNG_NAME+"_%03d"%BATCH_SIZE)
tuner.search(X_train, y_train,batch_size=BATCH_SIZE,verbose=1,epochs=20,validation_data=(X_test,y_test),callbacks=[StopWhenDoneCallback(),tf.keras.callbacks.TerminateOnNaN(),tensorboard_callback])
tuner.results_summary()
training_size = np.geomspace(BATCH_SIZE, X_train.shape[0]-1, num=LOG_STEPS)
best_hps = tuner.get_best_hyperparameters(num_trials =6)[-1]
print(best_hps)
fastmode=True
log_dir = "logs/"+RNG_NAME+"/%s/"%("fastmode" if fastmode else "normal")+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,write_graph=False,profile_batch=0)
model = tuner.hypermodel.build(best_hps)
MAXEPOCHS=50
model.load_weights("128plusplusDifferent")
model.fit(X_train,y_train,batch_size=BATCH_SIZE,verbose=1,epochs=50,validation_data=(X_test,y_test),callbacks=[StopWhenDoneCallback(),tf.keras.callbacks.TerminateOnNaN(),tensorboard_callback])
results = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
print("test loss: %f, test acc: %s" % tuple(results))
model.save('128plus_model')