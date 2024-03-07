import random

MAXSIZE = 0xFFFFFFFFFFFFFFFF
COUNT = 8000000


def xorshift128plus(x, y):
	s0, s1 = y, x
	s1 ^= (s1 << 23) & MAXSIZE
	s1 ^= (s1 >> 17)
	s1 ^= s0
	s1 ^= (s0 >> 26)
	x = y
	y = s1
	generated = (x + y) & MAXSIZE
	return x, y, generated

def xorinald(x, y):
	s0, s1 = y, x
	s1 ^= (s1 << 23) & MAXSIZE
	#s1 ^= (s1 >> 17)
	#s1 ^= s0
	#s1 ^= (s0 >> 26)
	x = y
	y = s1
	generated = (x + y) & MAXSIZE
	return x, y, generated


f1 = open("xorshift128_forward_pass.rng", "w")
f2 = open("bad.rng", "w")
# generate COUNT instances of x,y,generated, using x and y as the state for the rng
fs = [f1, f2]
rngs = [xorshift128plus, xorinald]
for i in range(COUNT):
	for j in range(2):
		x = random.randint(0, MAXSIZE)
		y = random.randint(0, MAXSIZE)
		_, __, out = rngs[j](x, y)
		fs[j].write(f'{x} {y} {out}\n')
