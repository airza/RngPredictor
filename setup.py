ENOUGH_DATA=2000000
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
		xx,yy,zz,ww = x,y,z,w
		t = x ^ ((x << 11) & 0xFFFFFFFF)  # 32bit
		x, y, z = y, z, w
		w = (w ^ (w >> 19)) ^ (t ^ (t >> 8))
		return (w,xx,yy,zz,ww)
	return _random
def xorshift128plus():
	x = 1
	y = 2
	MAXSIZE=0xFFFFFFFFFFFFFFFF
	def _rand():
		nonlocal y,x
		xx,yy = x,y
		s0,s1=y,x
		s1 ^= (s1 << 23) & MAXSIZE
		s1 ^= (s1 >> 17)
		s1 ^= s0
		s1 ^= (s0 >> 26)
		x = y
		y = s1
		generated = (x+y) & MAXSIZE
		return (generated,xx,yy)
	return _rand
def sequence():
	s = 0
	def _rand():
		nonlocal s
		s+=1
		return s
	return _rand
if __name__ == '__main__':
	for f in [xorshift128plus,xorshift128]:
		_outputfile = open(f.__name__+".rng","w")
		_statefile = open(f.__name__+"_state.rng","w")
		rng = f()
		for i in range(ENOUGH_DATA):
			output, *state=rng()
			_outputfile.write(str(output)+"\n")
			_statefile.write(" ".join(map(str,state))+"\n")
		_outputfile.close()
		_statefile.close()
		"""
		for d in [31,16,8,4,2,1]:
			_file = open(f.__name__+"TRUNCATED_%d.rng"%d,"w")
			rng = f()
			for i in range(ENOUGH_DATA):
				z = (rng()) >> d
				_file.write(str(z)+"\n")
			_file.close()
        """