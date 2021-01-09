import numpy as np
def strided(a, L):
	#I don't recommend touching this code.
	shp = a.shape
	s  = a.strides
	nd0 = shp[0]-L+1
	shp_in = (nd0,L)+shp[1:]
	strd_in = (s[0],) + s
	return np.lib.stride_tricks.as_strided(a, shape=shp_in, strides=strd_in)

def get_state_data_from_file(rngname,total_data_count):
	state_df = np.genfromtxt(rngname+"_state.rng")[:total_data_count].astype("uint64")
	BIT_WIDTH=np.ceil(np.log2(np.amax(state_df))).astype(int)
	old_shape = state_df.shape
	df2 = state_df.reshape(-1)
	df3 = (df2[:,None] & (1 << np.arange(BIT_WIDTH-1,-1,-1,dtype='uint64')) > 0).astype(int)
	print(total_data_count,old_shape[1]*BIT_WIDTH)
	X = df3.reshape(total_data_count,old_shape[1]*BIT_WIDTH)
	output_df =np.genfromtxt(rngname+".rng")[:total_data_count].astype("uint64")
	BIT_WIDTH =np.ceil(np.log2(np.amax(output_df))).astype(int)
	y =(output_df[:,None] & (1 << np.arange(BIT_WIDTH-1,-1,-1,dtype='uint64')) > 0).astype(int)
	indicies = np.arange(total_data_count,dtype='uint64')
	np.random.shuffle(indicies)
	X=X[indicies]
	y=y[indicies]
	return (X,y)
def get_data_from_file(filename,total_data_count,previous_timestep_count,start_bit=None,end_bit=None):
	TOTAL_DATA_NUM = total_data_count-previous_timestep_count
	df = np.genfromtxt(filename,delimiter='\n',dtype='uint64')[:total_data_count]
	#calculates how many bits are in the output.
	BIT_WIDTH=np.ceil(np.log2(np.amax(df))).astype(int)
	#We need to transform this from a list of numbers to a list of those numbers
	#binary representation, and then slice it into 1 observed output and
	#previous_timestep_count previous outputs. I also do not recommend touching this code
	#as it is quite fucky.
	df_as_bits =(df[:,None] & (1 << np.arange(BIT_WIDTH,dtype='uint64')) > 0).astype(int)
	df_as_frames = strided(df_as_bits,previous_timestep_count+1)
	#normal shuffle doesn't work for some reason, oh well
	indicies = np.arange(TOTAL_DATA_NUM,dtype='uint64')
	np.random.shuffle(indicies)
	df_as_frames=df_as_frames[indicies]
	#Now is the correct time if you want to narrow the RNG prediction to specific bits
	#which is probably a much easier problem (e.g.)
	#y = df_as_frames[:,-1,:]
	#will take only 2 bits.
	if start_bit==None:
		y = df_as_frames[:,-1,:]
	else:
		y = df_as_frames[:,-1,:][:,start_bit:end_bit]
	y= y.astype('float64')
	X = df_as_frames[:,:-1,]
	return (X,y)