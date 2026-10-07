from math import floor, log2

L=16
w=1<<L

def inv8bit(hy):
	z = (w * w-1) // (hy << 8)
	assert z < (1<<17)
	assert z*(hy<<8) < (1<<32)
	return z

def inv( y ):
	l = floor( log2( y ) ) + 1
	yd = y * (2 ** (L - l))
	z = inv8bit(yd>>8)
	

	z = z >> l
	assert 0 < z and z < w
	assert w/y-2 < z and z < w/y+1
	assert w-2*y < z*y and z*y < w+y
	assert isinstance(z, int)
	if ((y*z)%w) < y:
		z -= 1
	# if y*z < w-y:
	# 	z += 1
	assert w/y-1 <= z and z < w/y
	assert w-y <= z*y and z*y < w
	return z

for i in range(1,2**16):
	z = inv(i)
print("All tests passed")