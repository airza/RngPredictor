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
BIT=16
BATCH_SIZE = 512
RNG_NAME = "xorshift128"
if "xorshift128plus" == RNG_NAME:
	PREV_COUNT = 2
elif "xorshift128" == RNG_NAME:
	PREV_COUNT = 4
LOSS_FUNCTION ='mse'
METRIC_FUNCTION = 'binary_accuracy'
def transformer(layer,num_heads,key_dim,leak=0,dropout=0):
	lrelu = lambda x: tf.keras.activations.relu(x, alpha=leak)
	mha = MultiHeadAttention(num_heads=num_heads,key_dim=key_dim,attention_axes=1,dropout=dropout)
	res = mha(layer,layer)
	res = Dense(layer.shape[1],activation=lrelu)(res)
	res = tf.keras.layers.ReLU(negative_slope=leak)(layer+res)
	return res
def build_model(hp):
	size = hp.Float("size",1,1.5)*20*3*8*3
	feature_dim = hp.Int("feature_dim",16,24,sampling="log")
	size/=feature_dim
	t_count = hp.Choice("t_count",2,4)
	size/=t_count

	lrelu = lambda x: tf.keras.activations.relu(x,alpha=leakiness)
	key_dim = 8
	leakiness = .01
	dropout = 0#hp.Choice("dropout",[0.0,.001,.1])
	heads = 3
	inputs = Input(shape=(X.shape[1],))
	i1 = inputs*2
	i2 = i1-1
	t = Dense(feature_dim,activation='relu')(i2)
	t = Dense(feature_dim,activation='relu')(t)
	for i in range(t_count):
		t = transformer(t,heads,key_dim,leakiness,dropout)
	outputSize = 1 if len(y.shape)==1 else y.shape[1]
	t = Dense(feature_dim,activation='relu')(t)
	outLayer= Dense(outputSize,activation='tanh')(t)  
	out = outLayer*.5
	output = out+.5
	loss = LOSS_FUNCTION
	model =keras.Model(inputs=inputs,outputs=output,name="fuckler")
	opt = keras.optimizers.Nadam(
		learning_rate=hp.Float("learning_rate", 10**-3.5,10**-3,sampling="log"),
		epsilon= 1e-9,
		beta_1= hp.Float("beta_1", .5,.9,sampling="log"),
		beta_2= hp.Float("beta_2", .5,.99,sampling="log"),
	)
	model.compile(optimizer=opt,loss=loss,metrics=[METRIC_FUNCTION])
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
for BIT in range(3,32):
	X,y=get_data_from_file(RNG_NAME+'.rng',IMPORT_COUNT,PREV_COUNT,bit=BIT)
	X = np.reshape(X,[X.shape[0],-1])
	X_train = X[TEST_COUNT:]
	X_test = X[:TEST_COUNT]
	y_train = y[TEST_COUNT:]
	y_test = y[:TEST_COUNT]
	log_dir = "logs/"+RNG_NAME+"/BIT_NUM_%02d/"%BIT+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,write_graph=False,profile_batch=0)
	tuner = kt.tuners.randomsearch.RandomSearch(build_model,METRIC_FUNCTION,20,project_name="hp_"+RNG_NAME+"_%02d"%BIT)
	tuner.search(X_train, y_train,batch_size=BATCH_SIZE,verbose=0,epochs=15,validation_data=(X_test,y_test),callbacks=[StopWhenDoneCallback(),tf.keras.callbacks.TerminateOnNaN(),tensorboard_callback])
	tuner.results_summary()
	best_hps = tuner.get_best_hyperparameters(num_trials =1)[0]
	model = tuner.hypermodel.build(best_hps)
	model.fit(X_train, y_train,batch_size=BATCH_SIZE,verbose=0,epochs=100,validation_data=(X_test,y_test),callbacks=[StopWhenDoneCallback(),tf.keras.callbacks.TerminateOnNaN(),tensorboard_callback])
	results = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
	model.save('xorshift128_plus_bit_%2d'%BIT)
	print("BIT NUMBER:"+str(BIT)+" ------- test loss: %f, test acc: %s" % tuple(results))