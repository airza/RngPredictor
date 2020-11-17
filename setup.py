ENOUGH_DATA=4000000
def xorshift128():
    '''xorshift
    https://ja.wikipedia.org/wiki/Xorshift
    '''
    x = 123456789
    y = 362436069
    z = 521288629
    w = 88675123
    def _random():
        nonlocal x, y, z, w
        t = x ^ ((x << 11) & 0xFFFFFFFF)  # 32bit
        x, y, z = y, z, w
        w = (w ^ (w >> 19)) ^ (t ^ (t >> 8))
        return w
    return _random
def xorshift128plus():
    x = 1234567890
    y = 9876543210
    MAXSIZE=0xFFFFFFFFFFFFFFFF
    def _rand():
        nonlocal y,x
        s0,s1=y,x
        s1 ^= (s1 << 23) & MAXSIZE
        s1 ^= (s1 >> 17)
        s1 ^= s0
        s1 ^= (s0 >> 26)
        x = y
        y = s1
        generated = (x+y) & MAXSIZE
        return generated
    return _rand
if __name__ == '__main__':
	for f in [xorshift128,xorshift128plus]:
		_file = open(f.__name__+".rng","w")
		rng = f()
		for i in range(ENOUGH_DATA):
			_file.write(str(rng())+"\n")
		_file.close()
		_file = open(f.__name__+"TRUNCATED.rng","w")
		rng = f()
		for i in range(ENOUGH_DATA):
			z = (rng() & 0xFFFF0000) >> 16
			_file.write(str(z)+"\n")
		_file.close()