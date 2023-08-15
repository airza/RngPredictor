default_x = 1234567890
default_y = 9876543210
MAXSIZE = 0xFFFFFFFFFFFFFFFF
COUNT = 100000
import numpy as np
def xorshift128plus(x,y):
	s0,s1=y,x
	s1 ^= (s1 << 23) & MAXSIZE
	s1 ^= (s1 >> 17)
	s1 ^= s0
	s1 ^= (s0 >> 26)
	x = y
	y = s1
	generated = (x+y) & MAXSIZE
	return (x,y,generated)
f = open("xorshift128_forward_pass.rng","w")
x,y = default_x,default_y
for i in range(COUNT):
	nX,nY,generated = xorshift128plus(x,y)
	f.write(f'{x} {y} {generated}\n')
	x,y = nX,nY
print("DONE")


def xorshift128minus(arr):
	x,y = int(arr[0]),int(arr[1])
	s0,s1=y,x
	s1 ^= (s1 << 23) & MAXSIZE
	s1 ^= (s1 >> 17)
	s1 ^= s0
	#s1 ^= (s0 >> 26)
	x = y
	y = s1
	generated = (x+y) & MAXSIZE
	return (np.uint64(x), np.uint64(y), np.uint64(generated))
#generate COUNT instances of x,y,generated, using x and y as the state for the rng

f2 = open("xorshift128_minus_forward_pass.rng","w")
numbers = np.random.randint(0,MAXSIZE-1,size=(COUNT,2),dtype=np.uint64)
outputs = np.apply_along_axis(xorshift128minus,1,numbers)
all = np.concatenate((numbers,outputs),axis=1)
#write each line to f2
for line in all:
	f2.write(f'{line[0]} {line[1]} {line[2]}\n')

