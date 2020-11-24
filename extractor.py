import numpy as np
def strided(a, L):
	#I don't recommend touching this code.
	shp = a.shape
	s  = a.strides
	nd0 = shp[0]-L+1
	shp_in = (nd0,L)+shp[1:]
	strd_in = (s[0],) + s
	return np.lib.stride_tricks.as_strided(a, shape=shp_in, strides=strd_in)
def get_data_from_file(filename,total_data_count,previous_timestep_count):
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
	# y = df_as_frames[:,-1,:][:,_INSERT_MASK_HERE_]
	y = df_as_frames[:,-1,:]
	X = df_as_frames[:,:-1,]
	return (X,y)