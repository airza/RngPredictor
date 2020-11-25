import numpy as np
import tensorflow as tf
import kerastuner as kt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, MultiHeadAttention
from tensorflow.keras import Input
import datetime
from extractor import get_data_from_file
IMPORT_COUNT = 600000
TEST_COUNT = 20000
RNG_NAME="sequence"
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
LOSS="mse"
inputs = Input(shape=(X.shape[1],))
queries = Dense(X.shape[1],activation="relu")(inputs)
mha = MultiHeadAttention(num_heads=4,key_dim=X.shape[1],attention_axes=0)
layer = mha(inputs,queries)
layer2 = Dense(X.shape[1],activation="relu")(layer)
output= Dense(y.shape[1])(layer2)
model = keras.Model(inputs=inputs,outputs=output,name="fuckler")
opt = keras.optimizers.Nadam(
	learning_rate=.9,
	epsilon=1e-8,
	beta_1=.9,
	beta_2=.9,
)
model.compile(optimizer=opt,loss=tf.keras.losses.MSE,metrics=['binary_accuracy'])
model.summary()
#define CB
log_dir = "logs/"+RNG_NAME+"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,profile_batch=0)
model.fit(X_train,y_train,epochs=50, batch_size=512,callbacks=[tensorboard_callback,],verbose=0)
model.save_weights(RNG_NAME+"_DONE")