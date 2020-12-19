import numpy as np
import pkg_resources
import tensorflow as tf
import kerastuner as kt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, MultiHeadAttention,LayerNormalization,BatchNormalization,ReLU,Dropout,LSTM
from tensorflow.keras import Input
import datetime
from extractor import get_data_from_file

LOG_STEPS = 5
IMPORT_COUNT = 2**19
TEST_COUNT = 2**14
START_BIT = 0
END_BIT = 4
BATCH_SIZE = 1024
RNG_NAME = "xorshift128"
if "xorshift128plus" == RNG_NAME:
	PREV_COUNT = 2
elif "xorshift128" == RNG_NAME:
	PREV_COUNT = 4
LOSS_FUNCTION ='mse'
METRIC_FUNCTION = 'binary_accuracy'
def get_relu(leak):
	return lambda x: tf.keras.activations.relu(x, alpha=leak)

def residual_transformer(layer,num_heads,key_dim,attention_dim,activation=tf.keras.activations.relu,self_attention=True):
	if self_attention:
		keys=layer
	else:
		keys = Dense(layer.shape[1],activation=activation)(layer)
	mha = MultiHeadAttention(num_heads=num_heads,key_dim=key_dim,attention_axes=attention_dim)
	res = mha(layer,keys)
	return activation(layer+res)
def build_model(hp):
	alpha = hp.Float("alpha",0.01,.4,sampling="log")
	activation_function= lambda x: tf.keras.activations.relu(x, alpha=alpha)
	t_count = 2#hp.Int("depth",3,5)
	key_dim = hp.Choice("key_dim",[8])
	heads = (w)//key_dim
	inputs = Input(shape=(X.shape[1],X.shape[2]))
	i1 = inputs*2
	i2 = i1-1
	l = i2
	rate = hp.Float("noise",0.001,.1,sampling="log")
	l = tf.keras.layers.GaussianNoise(rate)(l)
	attention_dim=hp.Choice("attention_dim",["two","one","onetwo"])
	if attention_dim=="two":
		attention_dim=(2,)
	elif attention_dim=="one":
		attention_dim=(1,)
	elif attention_dim=="onetwo":
		attention_dim=(1,2)
	for i in range(t_count):
		l = residual_transformer(l,heads,key_dim,activation=activation_function,attention_dim=attention_dim)
	l = LSTM(X.shape[2])(l)
	outputSize = 1 if len(y.shape)==1 else y.shape[1]
	outLayer= Dense(outputSize)(l)  
	out = outLayer*.5
	output = out+.5
	loss = LOSS_FUNCTION
	model =keras.Model(inputs=inputs,outputs=output,name="fuckler")
	opt = keras.optimizers.Nadam(
		learning_rate=hp.Float("learning_rate", 10**-4.5,10**-2.5,sampling="log"),
		epsilon= 1e-12,
		beta_1= hp.Float("beta_1", .2,.9,sampling="reverse_log"),
		beta_2= hp.Float("beta_2", .2,.99,sampling="reverse_log"),
	)
	model.compile(optimizer=opt,loss=loss,metrics=[METRIC_FUNCTION])
	model.summary()
	return model
class StopWhenDoneCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
    	accuracy= logs['binary_accuracy']
    	if accuracy>.99:
    		self.model.stop_training = True
reduce_lr=keras.callbacks.ReduceLROnPlateau(
    monitor='loss', factor=0.8, patience=2, verbose=1,min_delta=0.1, min_lr=1e-5
)
"""
Control how many outputs back the model should look.
If you are not sure, I would suggest
(Size of the RNG state in bits)/(Bits of output from the RNG).
If your RNG produces low entropy output, you
may need more past data-but I have no tested this.
"""
X,y=get_data_from_file(RNG_NAME+'.rng',IMPORT_COUNT,PREV_COUNT,start_bit=START_BIT,end_bit=END_BIT)
print(X.shape)
X_train = X[TEST_COUNT:]
X_test = X[:TEST_COUNT]
y_train = y[TEST_COUNT:]
y_test = y[:TEST_COUNT]
log_dir = "logs/"+RNG_NAME+"/START_%02d_END_%02d/"%(START_BIT,END_BIT) +datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,write_graph=False,profile_batch=0)
tuner = kt.tuners.randomsearch.RandomSearch(build_model,METRIC_FUNCTION,100,project_name="hp_"+RNG_NAME+"_START_%02d_END_%02d"%(START_BIT,END_BIT))
tuner.search(X_train, y_train,batch_size=BATCH_SIZE,verbose=1,epochs=20,validation_data=(X_test,y_test),callbacks=[StopWhenDoneCallback(),tf.keras.callbacks.TerminateOnNaN(),tensorboard_callback])
tuner.results_summary()
best_hps = tuner.get_best_hyperparameters(num_trials =3)[-1]
model = tuner.hypermodel.build(best_hps)
model.fit(X_train, y_train,batch_size=BATCH_SIZE,verbose=0,epochs=10,validation_data=(X_test,y_test),callbacks=[StopWhenDoneCallback(),tf.keras.callbacks.TerminateOnNaN(),tensorboard_callback])
results = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
model.save_weights(RNG_NAME+"_START_%02d_END_%02d"%(START_BIT,END_BIT)+"_WEIGHTS")