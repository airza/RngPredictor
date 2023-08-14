import numpy as np
def strided(a, L):
	#I don't recommend touching this code.
	shp = a.shape
	s  = a.strides
	nd0 = shp[0]-L+1
	shp_in = (nd0,L)+shp[1:]
	strd_in = (s[0],) + s
	return np.lib.stride_tricks.as_strided(a, shape=shp_in, strides=strd_in)
def get_data_from_file(filename,total_data_count,previous_timestep_count,bit=None):
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
	#indicies = np.arange(TOTAL_DATA_NUM,dtype='uint64')
	#np.random.shuffle(indicies)
	#df_as_frames=df_as_frames[indicies]
	#Now is the correct time if you want to narrow the RNG prediction to specific bits
	#which is probably a much easier problem (e.g.)
	#y = df_as_frames[:,-1,:]
	#will take only 2 bits.
	if bit==None:
		y = df_as_frames[:,-1,:]
	else:
		y = df_as_frames[:,-1,:][:,bit]
	X = df_as_frames[:,:-1,]
	return (X,y)

def debug(State1,State2,X1,X2,X,n):
	def int2bits(n):
		return '{:064b}'.format(n)
	def bits2str(bs):
		return ''.join(map(str,bs))
	s1,s2,x,x1,x2 = State1[n], State2[n], X[n],X1[n],X2[n]
	print(int2bits(s1)+int2bits(s2))
	print(bits2str(x))
def getBits(df,BIT_WIDTH):
	return (df[:,None] & (1 << np.arange(BIT_WIDTH-1,-1,-1,dtype='uint64')) > 0).astype(int)
def get_input_and_output_from_file(filename,total_data_count,bit=None):
	data = np.loadtxt(filename, dtype=np.uint64)[:total_data_count]

	# Separate the data into State1, State2, and Output
	State1, State2, Output = data[:, 0], data[:, 1], data[:, 2]

	# Convert each state into its 64-bit binary representation
	X1 = getBits(State1,64)
	X2 = getBits(State2,64)
	Y = getBits(Output,64)
	if bit != None:
		Y = Y[:,bit].reshape(-1,1)
	# Concatenate X1 and X2 to form the X array
	X = np.concatenate((X1, X2), axis=1)
	# Correct the endianness if needed
	debug(State1, State2,X1,X2, X, 0)
	return (X,Y)