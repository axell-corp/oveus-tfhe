	.file	"spqlios-ifft-fma.s"
#if !__APPLE__
	.section .note.GNU-stack,"",%progbits
#endif
	.text
	.p2align 4
#if !__APPLE__
	.globl	ifft
	.type	ifft, @function
ifft:
#else
	.globl	_ifft
_ifft:
#endif

/* void _ifft(const void *tables, double *real) */
	/* Save registers */
	pushq       %r10
	pushq       %r11
	pushq       %r12
	pushq       %r13
	pushq       %r14
	pushq       %rbx

	/* Permute registers for better variable names */
	movq        %rdi, %rax
	movq        %rsi, %rdi      /* rdi: base of the real data */

	/* Load struct FftTables fields */
	movq         0(%rax), %rdx  /* rdx: n */
	movq         8(%rax), %r8   /* r8: trig_tables base */

	movq	%rdx, %r10
	shl	$1, %r10
	add	%r10, %rsi          /* rsi: base of the imaginary data */

	shr	$3, %r10  /* r10 = n/4 = ns4 (constant) */

# ── First twist multiply + first butterfly fused for ns4 >= 16 ──────────────
# For the IFFT the first operation is the twist multiply, then the largest
# butterflies work down. We fuse the twist with the first butterfly
# (halfnn = ns4/2) when possible.
	movq	%r10,%r12         /* r12 = ns4 */
	shr	$1,%r12           /* r12 = ns4/2 = largest halfnn */
	cmpq	$8,%r12
	jb	ifft_twist_only   /* ns4 < 16: just do twist, then small passes */

	# Compute twist trig base and first butterfly trig base
	movq	%r8,%r11          /* r11 = twist trig pointer */
	# First butterfly trig base = r8 + ns4*16 (twist table size)
	leaq	(%r8,%r10,8),%rax
	leaq	(%rax,%r10,8),%rax /* rax = r8 + ns4*16 = first bf trig base */

	# For the first (largest) halfnn:
	# re0 = rdi, im0 = rsi, re1 = rdi+halfnn*8, im1 = rsi+halfnn*8
	movq	%rdi,%r13         /* r13 = re0 */
	movq	%rsi,%r14         /* r14 = im0 */
	leaq	(%rdi,%r12,8),%rbx /* rbx = re1 */
	# Can't use more GPRs, compute im1 offset on the fly
	movq	$0,%rcx
.p2align 4
ifft_fused_twist_bf_loop:
	# Load data
	vmovapd (%r13),%ymm0         /* re0 */
	vmovapd (%r14),%ymm1         /* im0 */
	leaq (%rsi,%r12,8),%rdx      /* im1 base - computed each iter */
	vmovapd (%rbx),%ymm2         /* re1 */
	vmovapd (%rdx,%rcx,8),%ymm3  /* im1 */

	# Twist on input 0 (position = off)
	vmovapd (%r11),%ymm4         /* tw cos */
	vmovapd 32(%r11),%ymm5       /* tw sin */
	vmulpd %ymm0,%ymm4,%ymm6
	vmulpd %ymm0,%ymm5,%ymm7
	vfnmadd231pd %ymm1,%ymm5,%ymm6   /* re0' = re0*cos - im0*sin */
	vfmadd231pd %ymm1,%ymm4,%ymm7    /* im0' = re0*sin + im0*cos */

	# Twist on input 1 (position = off + halfnn)
	# Twist trig for off+halfnn = r11 + halfnn*16 bytes
	leaq (%r11,%r12,8),%rdx
	leaq (%rdx,%r12,8),%rdx      /* rdx = twist trig for pos off+halfnn */
	vmovapd (%rdx),%ymm4
	vmovapd 32(%rdx),%ymm5
	vmulpd %ymm2,%ymm4,%ymm8
	vmulpd %ymm2,%ymm5,%ymm9
	vfnmadd231pd %ymm3,%ymm5,%ymm8   /* re1' */
	vfmadd231pd %ymm3,%ymm4,%ymm9    /* im1' */

	# IFFT DIF butterfly: sum = a+b, diff_twiddled = (a-b)*W
	vmovapd (%rax),%ymm4         /* bf cos */
	vmovapd 32(%rax),%ymm5       /* bf sin */
	vaddpd %ymm6,%ymm8,%ymm0    /* re0' + re1' → sum_re */
	vaddpd %ymm7,%ymm9,%ymm1    /* im0' + im1' → sum_im */
	vsubpd %ymm8,%ymm6,%ymm2    /* re0' - re1' → diff_re */
	vsubpd %ymm9,%ymm7,%ymm3    /* im0' - im1' → diff_im */
	vmovapd %ymm0,(%r13)         /* store sum_re */
	vmovapd %ymm1,(%r14)         /* store sum_im */
	# diff * W
	vmulpd %ymm2,%ymm4,%ymm0    /* diff_re*cos */
	vfnmadd231pd %ymm3,%ymm5,%ymm0   /* diff_re*cos - diff_im*sin */
	vmulpd %ymm2,%ymm5,%ymm1    /* diff_re*sin */
	vfmadd231pd %ymm3,%ymm4,%ymm1    /* diff_re*sin + diff_im*cos */
	leaq (%rsi,%r12,8),%rdx
	vmovapd %ymm0,(%rbx)         /* store diff_re at re1 */
	vmovapd %ymm1,(%rdx,%rcx,8)  /* store diff_im at im1 */

	# Advance
	leaq 32(%r13),%r13
	leaq 32(%r14),%r14
	leaq 32(%rbx),%rbx
	leaq 64(%r11),%r11       /* twist trig */
	leaq 64(%rax),%rax       /* bf trig */
	addq $4,%rcx
	cmpq %r12,%rcx
	jb ifft_fused_twist_bf_loop

	# Update r8 for subsequent passes:
	# skip twist (ns4*16) + first bf (2*ns4*8 = ns4*16) + second bf (halfnn*16)
	# Actually: after fused twist+halfnn, the next halfnn is ns4/4
	# Trig pointer: r8 + twist_size + all bf entries up to and including first
	movq %r12,%rax            /* halfnn that was just processed */
	# r8 needs to advance past twist + first bf trig
	leaq (%r8,%r10,8),%r8
	leaq (%r8,%r10,8),%r8    /* r8 past twist table */
	leaq (%r8,%rax,8),%r8
	leaq (%r8,%rax,8),%r8    /* r8 past first bf trig (halfnn entries) */
	movq %r12,%r12            /* r12 = nn for next iteration */
	shr $1,%r12               /* r12 = next halfnn */
	jmp ifft_check_nn

ifft_twist_only:
	# Small N: just do twist, fall through to small passes
	movq    $0, %rcx
	movq	%r8, %r11
.p2align 4
ifft_firstloop:
	vmovapd (%rdi,%rcx,8), %ymm0
	vmovapd (%rsi,%rcx,8), %ymm1
	vmovapd 0(%r11), %ymm2
	vmovapd 32(%r11), %ymm3
	vmulpd  %ymm0, %ymm2, %ymm4
	vmulpd  %ymm0, %ymm3, %ymm5
	vfnmadd231pd  %ymm1, %ymm3, %ymm4
	vfmadd231pd  %ymm1, %ymm2, %ymm5
	vmovapd %ymm4, (%rdi,%rcx,8)
	vmovapd %ymm5, (%rsi,%rcx,8)
	leaq  64(%r11), %r11
	addq	$4,%rcx
	cmpq	%r10,%rcx
	jb	ifft_firstloop
	movq %r10,%r12            /* r12 = nn = ns4 for the first iteration */

ifft_check_nn:
	cmpq $8,%r12
	jbe ifft_before_small

# ── General IFFT DIF butterfly loop ─────────────────────────────────────────
ifft_nnloop:
	movq %r12,%r13
	shr  $1,%r13              /* r13 = halfnn */
	leaq (%r8,%r12,8),%r8
	leaq (%r8,%r12,8),%r8    /* update cur_tt += nn*16 */
	movq $0,%r11              /* r11 (block) */
ifft_blockloop:
	leaq (%rdi,%r11,8),%rax
	leaq (%rsi,%r11,8),%rbx
	leaq (%rax,%r13,8),%rcx
	leaq (%rbx,%r13,8),%rdx
	movq $0,%r9
	movq %r8,%r14
ifft_offloop:
	vmovapd (%rax,%r9,8), %ymm0
	vmovapd (%rbx,%r9,8), %ymm1
	vmovapd (%rcx,%r9,8), %ymm2
	vmovapd (%rdx,%r9,8), %ymm3
	vaddpd %ymm0,%ymm2,%ymm4
	vaddpd %ymm1,%ymm3,%ymm5
	vsubpd %ymm2,%ymm0,%ymm6
	vsubpd %ymm3,%ymm1,%ymm7
	vmovapd %ymm4,(%rax,%r9,8)
	vmovapd %ymm5,(%rbx,%r9,8)
	vmovapd (%r14),%ymm8
	vmovapd 32(%r14),%ymm9
	vmulpd %ymm6,%ymm8,%ymm4
	vfnmadd231pd %ymm7,%ymm9,%ymm4
	vmulpd %ymm6,%ymm9,%ymm5
	vfmadd231pd %ymm7,%ymm8,%ymm5
	vmovapd %ymm4,(%rcx,%r9,8)
	vmovapd %ymm5,(%rdx,%r9,8)
	leaq 64(%r14),%r14
	addq $4,%r9
	cmpq %r13,%r9
	jb ifft_offloop
	addq %r12,%r11
	cmpq %r10,%r11
	jb ifft_blockloop
	movq %r13,%r12
	cmpq $8,%r12
	ja ifft_nnloop

ifft_before_small:
# ── Fused size-8 (halfnn=4) + size-4 + size-2 pass ───────────────────────────
# Processes 8 doubles per iteration. All 3 stages in-register.
# IFFT DIF order: size-8 → size-4 → size-2
	vmovapd     ifft_s8cos(%rip), %ymm15
	vmovapd     ifft_s8sin(%rip), %ymm14
	vmovapd     ifft_negmask_s4a(%rip), %ymm13   /* [+1,+1,+1,-1] */
	vmovapd     ifft_negmask_s4b(%rip), %ymm12   /* [+1,+1,-1,+1] */
	vmovapd     ifft_negmask_s4c(%rip), %ymm11   /* [+1,+1,-1,-1] */
	vmovapd     ifft_negmask_s2(%rip), %ymm10    /* [+1,-1,+1,-1] */

	# Advance r8 past halfnn=4 trig (needed if we entered via nnloop)
	# halfnn=4 trig: nn=8, 4/4=1 group × 64 bytes
	leaq (%r8,%r12,8),%r8
	leaq (%r8,%r12,8),%r8    /* skip nn=8 trig entry */

	movq $0,%rax
	movq %rdi,%r11
	movq %rsi,%r12
.p2align 4
ifft_fused_842:
	# Load 2 consecutive blocks
	vmovapd (%r11),%ymm0         /* re[0..3] */
	vmovapd 32(%r11),%ymm1       /* re[4..7] */
	vmovapd (%r12),%ymm2         /* im[0..3] */
	vmovapd 32(%r12),%ymm3       /* im[4..7] */

	# ── Size-8 (halfnn=4) IFFT DIF butterfly across 2 blocks ──
	# sum = lo + hi, diff = lo - hi, then diff *= W8
	vaddpd %ymm0,%ymm1,%ymm4     /* sum_re */
	vaddpd %ymm2,%ymm3,%ymm5     /* sum_im */
	vsubpd %ymm1,%ymm0,%ymm6     /* diff_re */
	vsubpd %ymm3,%ymm2,%ymm7     /* diff_im */
	# diff * W8
	vmulpd %ymm6,%ymm15,%ymm0    /* diff_re * cos */
	vfnmadd231pd %ymm7,%ymm14,%ymm0  /* diff_re*cos - diff_im*sin */
	vmulpd %ymm6,%ymm14,%ymm2    /* diff_re * sin */
	vfmadd231pd %ymm7,%ymm15,%ymm2   /* diff_re*sin + diff_im*cos */
	# Now: re_lo=ymm4, re_hi=ymm0, im_lo=ymm5, im_hi=ymm2

	# ── Size-4 IFFT DIF butterfly ──
	# Pair 0: (ymm4=re_lo, ymm5=im_lo)
	# The DIF size-4 butterfly:
	#   Output[0] = re[0]+re[2]
	#   Output[1] = re[1]+im[3]  (note: +im, not +re for IFFT)
	#   Output[2] = re[0]-re[2]
	#   Output[3] = -(im[1]) + ... it's complex
	# Use the same pattern as the original IFFT size-4 code:
	vshufpd $10,%ymm5,%ymm4,%ymm6    /* [r0, i1, r2, i3] */
	vshufpd $10,%ymm4,%ymm5,%ymm7    /* [i0, r1, i2, r3] */
	vperm2f128 $0x20,%ymm6,%ymm4,%ymm8   /* [r0, r1, r0, i1] */
	vperm2f128 $0x31,%ymm6,%ymm4,%ymm9   /* [r2, r3, r2, i3] */
	vperm2f128 $0x20,%ymm7,%ymm5,%ymm1   /* [i0, i1, i0, r1] */
	vperm2f128 $0x31,%ymm7,%ymm5,%ymm3   /* [i2, i3, i2, r3] */
	vmulpd  %ymm8,%ymm13,%ymm8           /* [r0, r1, r0, -i1] */
	vfmadd231pd %ymm9,%ymm12,%ymm8       /* + [r2, r3, -r2, i3] */
	vfmadd231pd %ymm3,%ymm11,%ymm1       /* [i0,i1,i0,r1] + [i2,i3,-i2,-r3] */
	vmovapd %ymm8,%ymm4                  /* re_lo result */
	vmovapd %ymm1,%ymm5                  /* im_lo result */

	# Pair 1: (ymm0=re_hi, ymm2=im_hi) → same pattern
	vshufpd $10,%ymm2,%ymm0,%ymm6
	vshufpd $10,%ymm0,%ymm2,%ymm7
	vperm2f128 $0x20,%ymm6,%ymm0,%ymm8
	vperm2f128 $0x31,%ymm6,%ymm0,%ymm9
	vperm2f128 $0x20,%ymm7,%ymm2,%ymm1
	vperm2f128 $0x31,%ymm7,%ymm2,%ymm3
	vmulpd  %ymm8,%ymm13,%ymm8
	vfmadd231pd %ymm9,%ymm12,%ymm8
	vfmadd231pd %ymm3,%ymm11,%ymm1
	vmovapd %ymm8,%ymm0                  /* re_hi result */
	vmovapd %ymm1,%ymm2                  /* im_hi result */

	# ── Size-2 butterfly ──
	vshufpd $0,%ymm4,%ymm4,%ymm6
	vshufpd $15,%ymm4,%ymm4,%ymm4
	vfmadd231pd %ymm4,%ymm10,%ymm6       /* re_lo after s2 */
	vshufpd $0,%ymm0,%ymm0,%ymm7
	vshufpd $15,%ymm0,%ymm0,%ymm0
	vfmadd231pd %ymm0,%ymm10,%ymm7       /* re_hi after s2 */
	vshufpd $0,%ymm5,%ymm5,%ymm8
	vshufpd $15,%ymm5,%ymm5,%ymm5
	vfmadd231pd %ymm5,%ymm10,%ymm8       /* im_lo after s2 */
	vshufpd $0,%ymm2,%ymm2,%ymm9
	vshufpd $15,%ymm2,%ymm2,%ymm2
	vfmadd231pd %ymm2,%ymm10,%ymm9       /* im_hi after s2 */

	# Store
	vmovapd %ymm6,(%r11)
	vmovapd %ymm7,32(%r11)
	vmovapd %ymm8,(%r12)
	vmovapd %ymm9,32(%r12)
	leaq 64(%r11),%r11
	leaq 64(%r12),%r12
	addq $8,%rax
	cmpq %r10,%rax
	jb ifft_fused_842

	/* Restore registers */
ifft_end:
	vzeroall
	popq        %rbx
	popq        %r14
	popq        %r13
	popq        %r12
	popq        %r11
	popq        %r10
	retq


/* Constants for YMM */
.balign 32
/* IFFT size-8 twiddle: cos/sin(+2pi*k/8) for k=0..3 */
ifft_s8cos: .double +1.0, +0.7071067811865476, +0.0, -0.7071067811865476
ifft_s8sin: .double +0.0, +0.7071067811865476, +1.0, +0.7071067811865476
/* Size-4 IFFT DIF constants */
ifft_negmask_s4a: .double +1.0, +1.0, +1.0, -1.0
ifft_negmask_s4b: .double +1.0, +1.0, -1.0, +1.0
ifft_negmask_s4c: .double +1.0, +1.0, -1.0, -1.0
/* Size-2 constant */
ifft_negmask_s2:  .double +1.0, -1.0, +1.0, -1.0

#if !__APPLE__
	.size	ifft, .-ifft
#endif
