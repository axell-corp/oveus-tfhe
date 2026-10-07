import math

W = 32

def check(inp, bit):
    assert inp < (1 << bit)

def f_32(y):
    # 1. Normalization
    # We now normalize to the [2^31, 2^32) range
    clz = math.floor(math.log2(y)) + 1
    yy = y << (W - clz)
    
    # 2. Initial Seed (Approx 4 bits)
    # Extract top 8 bits for the seed table look-up logic
    hy = yy >> W//2
    hhy = hy >> W//4
    # z = (12 << 1) - (hhy + 1)
    z = 2**(W//2-1)//hhy
    
    # 3. First Iteration (4 bits -> 8 bits)
    # Same as before, bringing us to ~8 bits relative to the top 8 bits of input
    z = (z << W//4) - (z * (hy * z >> W//4) >> W//4)

    # if hy*z < 2**(W-2)-hy:
    #     z += 1
    # for i in range(2):
    #     if hy*z > 2**(W-2):
    #         z -= 1
    assert 2**(W-2) - 2*hy <= hy*z <= 2**(W-2)+2*hy
    
    # 4. Second Iteration (8 bits -> 16 bits)
    z = ((z << 3) << W//2) - (z * (yy * z >> (W//2-4)) >> W//2)
    
    # 6. De-normalization
    z = z >> clz
    
    # 7. Final Check & Correction
    # Handle the potential overflow if z was 2^32 exactly
    if z >= 2**W:
        z = 2**W - 1 # or handle as overflow flag
        
    # Standard remainder check
    if z * y < 2**W - y:
        z += 1
        
    return z

for b in range(1, 2**W):
    inv = f_32(b)
    # Verify: 2^32 - b <= inv * b <= 2^32
    # This assertion ensures z = floor(2^32 / b)
    print(b, inv, inv * b)
    assert 2**W - b <= inv * b <= 2**W