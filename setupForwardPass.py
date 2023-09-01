default_x = 1234567890
default_y = 9876543210
MAXSIZE = 0xFFFFFFFFFFFFFFFF
COUNT = 2000000
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
#generate COUNT instances of x,y,generated, using x and y as the state for the rng
for i in range(COUNT):
	nX,nY,generated = xorshift128plus(x,y)
	f.write(f'{x} {y} {generated}\n')
	x,y = nX,nY
