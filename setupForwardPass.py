import random

MAXSIZE = 0xFFFFFFFFFFFFFFFF
COUNT = 4
def xorshift128plus(x, y):
	s0, s1 = y, x
	s1 ^= (s1 << 23) & MAXSIZE
	s1 ^= (s1 >> 17)
	s1 ^= s0
	s1 ^= (s0 >> 26)
	x = y
	y = s1
	print(oksure(x), oksure(y))
	generated = (x + y) & MAXSIZE
	return generated, x, y,

def xorinald(x, y):
	s0, s1 = y, x
	s1 ^= (s1 << 23) & MAXSIZE
	s1 ^= (s1 >> 17)
	#s1 ^= s0
	#s1 ^= (s0 >> 26)
	x = y
	y = s1
	generated = (x + y) & MAXSIZE
	return x, y, generated

oksure = lambda num: format(num, '064b')
f1 = open("xorshift128_forward_pass.rng", "w")
f2 = open("bad.rng", "w")
# generate COUNT instances of x,y,generated, using x and y as the state for the rng
fs = [f1, f2]
rngs = [xorshift128plus]
