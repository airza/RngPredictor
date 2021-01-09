import numpy as np
import pkg_resources
import tensorflow as tf
import kerastuner as kt
from tensorflow import keras
from tensorflow.keras.layers import Dense
from components import residual_block,residual_transformer,residual_lstm
from tensorflow.keras import Input
import datetime
from extractor import get_state_data_from_file

IMPORT_COUNT = 2**20
TEST_COUNT = 2**14
BATCH_SIZE = 512
LOSS_FUNCTION = 'mse'
RNG_NAME="xorshift128plus"
METRIC_FUNCTION = 'binary_accuracy'
def build_model(hp):
	alpha = 0#hp.Float("alpha",0.01,.4,sampling="log")
	activation_function= lambda x: keras.activations.relu(x, alpha=alpha)
	rate = 0#hp.Float("noise",.0001,.01,sampling="log")
	a_count = 4
	key_dim = 32#hp.Choice("key_dim",[128])
	heads = 4
	inputs = Input(shape=(X.shape[1]))
	i1 = inputs*2
	i2 = i1-1
	l = i2
	for i in range(a_count):
		l =  residual_transformer(l,heads,key_dim,1,activation=activation_function)
	outputSize = 1 if len(y.shape)==1 else y.shape[1]
	l= Dense(outputSize)(l)
	outLayer= Dense(outputSize)(l)
	out = outLayer*.5
	output = out+.5
	loss = LOSS_FUNCTION
	model =keras.Model(inputs=inputs,outputs=output,name="fuckler")
	beta_1=hp.Float("b1",.2,.9,sampling="log")
	beta_2=hp.Float("b2",.2,.9,sampling="log")
	opt = tf.keras.optimizers.Nadam(
		learning_rate=hp.Float("learning_rate",1e-5,1e-2,sampling="log"),
		beta_1=beta_1,
		beta_2=beta_2,
		epsilon=1e-07,
    )
	model.compile(optimizer=opt,loss=loss,metrics=[METRIC_FUNCTION])
	model.summary()
	return model
class StopWhenDoneCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
    	accuracy= logs['binary_accuracy']
    	if accuracy>.99:
    		self.model.stop_training = True
stopEarly = tf.keras.callbacks.EarlyStopping(
	monitor='binary_accuracy', min_delta=.001, patience=15, verbose=0, mode='auto', restore_best_weights=False
)
"""
Control how many outputs back the model should look.
If you are not sure, I would suggest
(Size of the RNG state in bits)/(Bits of output from the RNG).
If your RNG produces low entropy output, you
may need more past data-but I have no tested this.
"""
X,y=get_state_data_from_file(RNG_NAME,IMPORT_COUNT)
X_train = X[TEST_COUNT:]
X_test = X[:TEST_COUNT]
y_train = y[TEST_COUNT:]
y_test = y[:TEST_COUNT]
log_dir = "logs/"+RNG_NAME+"/STATE_CALCULCATOR/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,write_graph=False,profile_batch=0)
tuner = kt.tuners.bayesian.BayesianOptimization(build_model,"val_loss",20,project_name="hp_"+RNG_NAME+"_STATE")
#tuner.search(X_train, y_train,batch_size=BATCH_SIZE,verbose=0,epochs=30,validation_data=(X_test,y_test),callbacks=[tf.keras.callbacks.TerminateOnNaN(),StopWhenDoneCallback(),tensorboard_callback])
tuner.results_summary()
best_hps = tuner.get_best_hyperparameters(num_trials =2)[-1]
model = tuner.hypermodel.build(best_hps)
model.fit(X_train, y_train,batch_size=BATCH_SIZE,verbose=0,epochs=300,validation_data=(X_test,y_test),callbacks=[StopWhenDoneCallback(),tensorboard_callback])
results = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
model.save_weights(RNG_NAME+"STATE_CALC"+"_WEIGHTS")