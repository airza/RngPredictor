from forwardPass.utils import int_to_bits_tensor, float_bits_to_int
from setupForwardPass import xorshift128plus
from XorShift128PlusNetwork import XorShift128NN
oksure = lambda num: format(num, '064b')
MAXSIZE = 0xFFFFFFFFFFFFFFFF
def xorshift128plus(x, y):
    s0, s1 = y, x
    s1 ^= (s1 << 23) & MAXSIZE
    s1 ^= (s1 >> 17)
    """
	s1 ^= s0
	s1 ^= (s0 >> 26)
	x = y
	y = s1
	print(oksure(x), oksure(y))
	generated = (x + y) & MAXSIZE
	return x, y, generated
	"""
    return s0,s1
x = int_to_bits_tensor(1)
y = int_to_bits_tensor(2)
model = XorShift128NN()
print(xorshift128plus(1,2))
z = model(x.unsqueeze(0))
print(float_bits_to_int(z))