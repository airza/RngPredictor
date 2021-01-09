import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, MultiHeadAttention,LSTM

def residual_block(layer,depth,activation=tf.keras.activations.relu):
	r = layer
	for i in range(depth):
		r = Dense(layer.shape[1],activation=activation)(r)
	return activation(r+layer)
def residual_transformer(layer,num_heads,key_dim,attention_axes,activation=keras.activations.relu):
	mha = MultiHeadAttention(num_heads=num_heads,key_dim=key_dim,attention_axes=attention_axes)
	res = mha(layer,layer)
	res = activation(res+layer)
	return res
def residual_lstm(layer,activation=keras.activations.relu,num_lstm=2):
	l = layer
	for i in range(num_lstm):
		l = LSTM(X.shape[2],activation=activation,return_sequences=True)(l)
	return activation(l+layer)