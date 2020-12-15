import numpy as np
import pkg_resources
import tensorflow as tf
import kerastuner as kt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, MultiHeadAttention,LayerNormalization,BatchNormalization,ReLU,Dropout
from tensorflow.keras import Input
import datetime
from extractor import get_data_from_file

LOG_STEPS = 5
IMPORT_COUNT = 2**20
TEST_COUNT = 2**14
START_BIT = 32
END_BIT = 64
BATCH_SIZE = 512
RNG_NAME = "xorshift128plus"
if "xorshift128plus" == RNG_NAME:
	PREV_COUNT = 2
elif "xorshift128" == RNG_NAME:
	PREV_COUNT = 4
LOSS_FUNCTION ='mse'
METRIC_FUNCTION = 'binary_accuracy'
def get_relu(leak):
	return lambda x: tf.keras.activations.relu(x, alpha=leak)
def residual_block(layer,depth,width,activation=tf.keras.activations.relu):
	r = layer
	for i in range(depth):
		if i==depth-1:
			r = Dense(layer.shape[1],activation=activation)(r)
		else:
			r = Dense(width,activation=activation)(r)
	return activation(r+layer)
def residual_transformer(layer,num_heads,key_dim,activation=tf.keras.activations.relu,self_attention=True):
	if self_attention:
		keys=layer
	else:
		keys = Dense(layer.shape[1],activation=activation)(layer)
	mha = MultiHeadAttention(num_heads=num_heads,key_dim=key_dim,attention_axes=1)
	res = mha(layer,keys)
	#res = Dense(layer.shape[1],activation=activation)(res)
	res = activation(layer+res)
	return res
def build_model(hp):
	alpha = hp.Float("alpha",0.001,.08,sampling="log")
	activation_function= lambda x: tf.keras.activations.relu(x, alpha=alpha)
	w = 128 #hp.Choice("w",[16,32,64])
	b = 4 # int(hp.Int("b",4,6)*(96/w))
	f_count = 0 #hp.Int("f_count",0,1)
	d_count = 1 #hp.Int("d_count",0,1)
	b -= f_count*2
	b -= d_count*2
	t_count = max(b,1)
	key_dim = 16
	heads = 8
	self_attention = False# shp.Choice("self_attention",[True,False])
	inputs = Input(shape=(X.shape[1],))
	i1 = inputs*2
	i2 = i1-1
	rate = 0#hp.Float("dropout",0.001,.05,sampling="log")
	l = i2
	l = tf.keras.layers.Dropout(rate)(l)
	l = Dense(w,activation=activation_function)(l)
	for i in range(f_count):
		l = residual_block(l,1,w,activation_function)
	for i in range(t_count):
		l = residual_transformer(l,heads,key_dim,activation=activation_function)
	for i in range(d_count):
		l = residual_block(l,2,w,activation_function)
	outputSize = 1 if len(y.shape)==1 else y.shape[1]
	outLayer= Dense(outputSize,activation='tanh')(l)  
	out = outLayer*.5
	output = out+.5
	loss = LOSS_FUNCTION
	model =keras.Model(inputs=inputs,outputs=output,name="fuckler")
	opt = keras.optimizers.Nadam(
		learning_rate=hp.Float("learning_rate", 10**-5.5,10**-4.5,sampling="log"),
		epsilon= 1e-12,
		beta_1= hp.Float("beta_1", .8,.99,sampling="log"),
		beta_2= hp.Float("beta_2", .8,.99,sampling="log"),
	)
	model.compile(optimizer=opt,loss=loss,metrics=[METRIC_FUNCTION])
	model.summary()
	return model
class StopWhenDoneCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
    	accuracy= logs['binary_accuracy']
    	if accuracy>.99:
    		self.model.stop_training = True
"""
Control how many outputs back the model should look.
If you are not sure, I would suggest
(Size of the RNG state in bits)/(Bits of output from the RNG).
If your RNG produces low entropy output, you
may need more past data-but I have no tested this.
"""
X,y=get_data_from_file(RNG_NAME+'.rng',IMPORT_COUNT,PREV_COUNT,start_bit=START_BIT,end_bit=END_BIT)
X = np.reshape(X,[X.shape[0],-1])
X_train = X[TEST_COUNT:]
X_test = X[:TEST_COUNT]
y_train = y[TEST_COUNT:]
y_test = y[:TEST_COUNT]
log_dir = "logs/"+RNG_NAME+"/START_%02d_END_%02d/"%(START_BIT,END_BIT) +datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,write_graph=False,profile_batch=0)
tuner = kt.tuners.randomsearch.RandomSearch(build_model,METRIC_FUNCTION,100,project_name="hp_"+RNG_NAME+"_START_%02d_END_%02d"%(START_BIT,END_BIT))
tuner.search(X_train, y_train,batch_size=BATCH_SIZE,verbose=0,epochs=50,validation_data=(X_test,y_test),callbacks=[StopWhenDoneCallback(),tf.keras.callbacks.TerminateOnNaN(),tensorboard_callback])
tuner.results_summary()
best_hps = tuner.get_best_hyperparameters(num_trials =3)[-1]
model = tuner.hypermodel.build(best_hps)
model.fit(X_train, y_train,batch_size=BATCH_SIZE,verbose=0,epochs=10,validation_data=(X_test,y_test),callbacks=[StopWhenDoneCallback(),tf.keras.callbacks.TerminateOnNaN(),tensorboard_callback])
results = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
model.save_weights(RNG_NAME+"_START_%02d_END_%02d"%(START_BIT,END_BIT)+"_WEIGHTS")