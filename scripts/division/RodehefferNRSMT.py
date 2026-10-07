from pysmt.logics import QF_BV
from pysmt.shortcuts import get_env, GT, Solver, Symbol, ForAll, Bool, BV, BVOr, BVAnd, BVAdd, BVSub, BVMul, BVUDiv, BVURem, BVLShl, BVNeg, BVLShl, BVLShr, BVZExt, BVSExt, BVExtract, BVULT, BVULE, BVUGE, Implies, Iff, Ite, AtMostOne, ExactlyOne, And, Or, Xor, Not, Equals, is_sat
from pysmt.typing import BVType

env = get_env()

env.factory.add_generic_solver("bitwuzla", ["/usr/local/bin/bitwuzla","-S","kissat","--time-limit",str(1000*3600*48)], [QF_BV])

solvername = 'bitwuzla'

W = 32
numiter = 5

def count_leading_zeros_bv(bv):
    """
    Given a BitVector expression 'bv' (a PySMT BV formula),
    return a BitVector expression that represents the number of 
    leading zeros in bv.

    Bit numbering:
      - bv.bv_width() returns the width (w) of the bitvector.
      - The most-significant bit (MSB) is at index w-1.
      - The least-significant bit (LSB) is at index 0.
    
    The function returns an expression equivalent to:
      if bv[w-1] == 1 then 0 else
        if bv[w-2] == 1 then 1 else
          ... else
            if bv[0] == 1 then w-1 else w.

    The result is a BitVector with a width sufficient to encode the number
    in the range 0 to w.
    """
    width = bv.bv_width()
    # Determine the minimal bit-width needed to represent values in 0..width.
    # result_width = max(1, (width + 1).bit_length())
    result_width = W
    # Create a list of bit-indices in descending order (MSB first).
    indices = list(range(width-1, -1, -1))
    
    def build_chain(idx_list):
        # If no more bits, then all bits were 0.
        if not idx_list:
            return BV(width, result_width)  # Represent count 'width'
        j = idx_list[0]
        # The candidate count is the number of zeros before bit j,
        # i.e. candidate = (width-1) - j.
        candidate = width - 1 - j
        # Construct a BitVector constant for candidate.
        candidate_bv = BV(candidate, result_width)
        # If bit j is 1 then return candidate, else check the rest.
        return Ite(Equals(BVExtract(bv,j,j), BV(1, 1)),
                   candidate_bv,
                   build_chain(idx_list[1:]))
    
    return build_chain(indices)

def MicrosoftNRInvIter(clzy, iter):
    if iter == 0:
        return BVLShl(BV(1, W),clzy)
    else:
        # ((z * (z*my % 2**W)) >> W) % 2**W
        z = MicrosoftNRInvIter(clzy, iter-1)
        return BVAdd(z, BVExtract(BVMul(BVZExt(z,W), BVZExt(BVMul(z,BVNeg(y)),W)),W,2*W-1))

def MicrosoftNRInv(y):
    clzy = count_leading_zeros_bv(y)
    z = MicrosoftNRInvIter(clzy, numiter)
    return Ite(BVULT(BVMul(y,z),BVNeg(y)),BVAdd(z,BV(1,W)),z)

def DivWithInv(x,y,z):
    q = BVExtract(BVMul(BVZExt(x,W),BVZExt(z,W)),W,2*W-1)
    r = BVSub(x,BVMul(q,y))
    cond = BVULE(y,r)
    q = Ite(cond,BVAdd(q,BV(1,W)),q)
    r = Ite(cond,BVSub(r,y),r)
    return (q,r)

x = Symbol("x", BVType(W))
y = Symbol("y", BVType(W))

tresdiv = Ite(Equals(y,BV(0,W)),BV(1,W),BVUDiv(x, y))
tresrem = Ite(Equals(y,BV(0,W)),x,BVURem(x, y))

s = Solver(name=solvername, logic=QF_BV, incremental=False)
z = MicrosoftNRInv(y)
yz = BVMul(y,z)

s.add_assertion(Not(Or(Equals(y,BV(0,W)),And(BVULE(BVNeg(y),yz),BVULE(yz,BV(2**W-1,W))))))
if s.solve():
    print("SAT")
    m = s.get_model()
    print(m[y])
    print(m[z])
else:
    print("UNSAT")


# q,r = DivWithInv(x,y,z)
# s.add_assertion(Not(And(Equals(tresdiv, q),Equals(tresrem, r))))
# clzy = count_leading_zeros_bv(y)
# if s.solve():
#     print("SAT")
#     m = s.get_model()
#     print((m[x],m[y]))
#     print((m[tresdiv],m[tresrem]))
#     print((m[q],m[r],m[z]))
# else:
#     print("UNSAT")