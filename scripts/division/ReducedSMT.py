#!/bin/bash
from pysmt.logics import QF_BV
from pysmt.shortcuts import get_env, GT, Solver, Symbol, ForAll, Bool, BV, BVOr, BVAnd, BVAdd, BVSub, BVMul, BVUDiv, BVURem, BVLShl, BVNeg, BVLShl, BVLShr, BVZExt, BVSExt, BVExtract, BVULT, BVULE, BVUGE, Implies, Iff, Ite, AtMostOne, ExactlyOne, And, Or, Xor, Not, Equals, is_sat
from pysmt.typing import BVType

# based on IntDivisionNR5NoFlag2OneFix

env = get_env()

env.factory.add_generic_solver("bitwuzla", ["/usr/local/bin/bitwuzla","-S","kissat","--time-limit",str(1000*3600*48)], [QF_BV])

solvername = 'bitwuzla'

W = 16

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

def is_power_of_two(bv):
    """
    Returns a PySMT Boolean formula that is True if and only if the given
    bit-vector 'bv' is a power of two, according to the condition:
      - bv & (bv - 1) == 0
    
    Note: Without the non-zero check, 0 will also satisfy this condition.
    """
    width = bv.bv_width()
    one = BV(1, width)
    bv_minus_one = BVSub(bv, one)
    and_result = BVAnd(bv, bv_minus_one)
    return Equals(and_result, BV(0, width))

def ReducedInv(y):
    clz = count_leading_zeros_bv(y)
    yy = BVLShl(y,clz)
    hy = BVExtract(yy,W//2)
    hhy = BVExtract(hy,W//4)

    # 1st iteration
    
    ehhy = BVZExt(hhy,1)
    carry = BVExtract(BVAdd(ehhy,ehhy),hhy.bv_width(),hhy.bv_width())
    # z = BVSub(BV(24,W//4),BVAdd(hhy,carry))
    z = BVSub(BV(8,W//4),BVAdd(hhy,BVZExt(carry,W//4-1)))
    # z = BVExtract(BVSub(BV(24,W//4+1),BVAdd(BVZExt(hhy,1),BVZExt(carry,W//4))),0,W//4-1)

    # 2nd iteration
    z = BVSub(BVLShl(BVZExt(z,W//4),W//4), BVExtract(BVMul(BVZExt(z,W//2),BVLShr(BVMul(BVZExt(z,W//2),BVZExt(hy,W//4)),W//4)),W//4))
    # return BVZExt(z,W-z.bv_width())

    # 3rd iteration
    z = BVSub(BVLShl(BVZExt(z,20-8),3+8),BVExtract(BVMul(BVZExt(z,20),BVZExt(BVExtract(BVMul(BVZExt(yy,W//2),BVZExt(z,W)),4),W//2)),8))

    # Last
    z = BVExtract(BVLShr(z,BV(W,20)-BVZExt(clz,20-W)),0,18)
    z = BVSub(BVExtract(z,0,W-1),BVZExt(BVExtract(z,W,W),W-1))
    # return z
    return Ite(BVULT(BVMul(BVZExt(y,1),BVZExt(z,1)),BVSub(BV(2**W,W+1),BVZExt(y,1))),BVAdd(z,BV(1,W)),z)

def DivWithInv(x,y,z):
    q = BVExtract(BVMul(BVZExt(x,W),BVZExt(z,W)),W)
    r = BVSub(x,BVMul(q,y))
    cond = BVULE(y,r)
    q = Ite(cond,BVAdd(q,BV(1,W)),q)
    r = Ite(cond,BVSub(r,y),r)
    # cond = BVULE(y,r)
    # q = Ite(cond,BVAdd(q,BV(1,W)),q)
    # r = Ite(cond,BVSub(r,y),r)
    return (q,r)

# def provezineq():
#     x = Symbol("x", BVType(W))
#     y = Symbol("y", BVType(W))
#     z = Symbol("z",BVType(W))
#     s = Solver(name=solvername, logic=QF_BV, incremental=False)
#     q,r = DivWithInv(x,y,z)
#     yz = BVMul(BVZExt(y,W),BVZExt(z,W))
#     tresdiv = Ite(Equals(y,BV(0,W)),BV(2,W),BVUDiv(x, y))
#     tresrem = Ite(Equals(y,BV(0,W)),x,BVURem(x, y))
#     s.add_assertion(Not(Implies(Or(And(BVULE(BV(2**W,2*W)-BVZExt(y,W),yz),BVULE(yz,BV(2**W,2*W)+BVZExt(y,W)))),Or(And(Equals(tresdiv, q),Equals(tresrem, r)),Equals(y,BV(0,W))))))
#     if s.solve():
#         print("SAT")
#         m = s.get_model()
#         print((m[x],m[y]))
#         print((m[tresdiv],m[tresrem]))
#         print((m[q],m[r],m[z]))
#     else:
#         print("UNSAT")
# provezineq()

x = Symbol("x", BVType(W))
y = Symbol("y", BVType(W))

tresdiv = Ite(Equals(y,BV(0,W)),BV(1,W),BVUDiv(x, y))
tresrem = Ite(Equals(y,BV(0,W)),x,BVURem(x, y))

s = Solver(name=solvername, logic=QF_BV, incremental=False)

z = ReducedInv(y)

yz = BVMul(y,z)
# powerof2 -> yz = 2^W ?
s.add_assertion(Not(Or(Equals(y,BV(0,W)),Ite(And(is_power_of_two(y),Not(Equals(y,BV(1,W)))), Equals(yz,BV(0,W)),And(BVULE(BVNeg(y),yz),BVULE(yz,BV(2**W-1,W)))))))
if s.solve():
    print("SAT")
    m = s.get_model()
    clz = count_leading_zeros_bv(y)
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