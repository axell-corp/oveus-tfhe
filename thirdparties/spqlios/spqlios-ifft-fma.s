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
	pushq       %r10
	pushq       %r11
	pushq       %r12
	pushq       %r13
	pushq       %r14
	pushq       %rbx
	movq        %rdi, %rax
	movq        %rsi, %rdi
	movq         0(%rax), %rdx
	movq         8(%rax), %r8
	movq	%rdx, %r10
	shl	$1, %r10
	add	%r10, %rsi
	shr	$3, %r10		/* r10 = ns4 */

# ── Pass 1: Fused twist + first butterfly (largest halfnn) ────────────────────
	movq	%r10,%r12
	shr	$1,%r12			/* r12 = ns4/2 = largest halfnn */
	cmpq	$8,%r12
	jb	ifft_twist_only

	movq	%r8,%r11		/* r11 = twist trig pointer */
	leaq	(%r8,%r10,8),%rax
	leaq	(%rax,%r10,8),%rax	/* rax = first bf trig base */
	movq	%rdi,%r13
	movq	%rsi,%r14
	leaq	(%rdi,%r12,8),%rbx
	movq	$0,%rcx
.p2align 4
ifft_fused_twist_bf:
	vmovapd (%r13),%ymm0
	vmovapd (%r14),%ymm1
	vmovapd (%rbx),%ymm2
	leaq (%rsi,%r12,8),%rdx
	vmovapd (%rdx,%rcx,8),%ymm3
	# Twist input 0
	vmovapd (%r11),%ymm4
	vmovapd 32(%r11),%ymm5
	vmulpd %ymm0,%ymm4,%ymm6
	vmulpd %ymm0,%ymm5,%ymm7
	vfnmadd231pd %ymm1,%ymm5,%ymm6
	vfmadd231pd %ymm1,%ymm4,%ymm7
	# Twist input 1
	leaq (%r11,%r12,8),%rdx
	leaq (%rdx,%r12,8),%rdx
	vmovapd (%rdx),%ymm4
	vmovapd 32(%rdx),%ymm5
	vmulpd %ymm2,%ymm4,%ymm8
	vmulpd %ymm2,%ymm5,%ymm9
	vfnmadd231pd %ymm3,%ymm5,%ymm8
	vfmadd231pd %ymm3,%ymm4,%ymm9
	# IFFT DIF butterfly
	vmovapd (%rax),%ymm4
	vmovapd 32(%rax),%ymm5
	vaddpd %ymm6,%ymm8,%ymm0
	vaddpd %ymm7,%ymm9,%ymm1
	vsubpd %ymm8,%ymm6,%ymm2
	vsubpd %ymm9,%ymm7,%ymm3
	vmovapd %ymm0,(%r13)
	vmovapd %ymm1,(%r14)
	vmulpd %ymm2,%ymm4,%ymm0
	vfnmadd231pd %ymm3,%ymm5,%ymm0
	vmulpd %ymm2,%ymm5,%ymm1
	vfmadd231pd %ymm3,%ymm4,%ymm1
	leaq (%rsi,%r12,8),%rdx
	vmovapd %ymm0,(%rbx)
	vmovapd %ymm1,(%rdx,%rcx,8)
	leaq 32(%r13),%r13
	leaq 32(%r14),%r14
	leaq 32(%rbx),%rbx
	leaq 64(%r11),%r11
	leaq 64(%rax),%rax
	addq $4,%rcx
	cmpq %r12,%rcx
	jb ifft_fused_twist_bf

	# Advance r8 past twist + first bf trig
	movq %r12,%rax
	leaq (%r8,%r10,8),%r8
	leaq (%r8,%r10,8),%r8
	leaq (%r8,%rax,8),%r8
	leaq (%r8,%rax,8),%r8
	movq %r12,%r12
	shr $1,%r12
	jmp ifft_after_first

ifft_twist_only:
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
	movq %r10,%r12

ifft_after_first:
# ── Pass 2: Fused halfnn=64+32 ("Option D", DIF order) for ns4>=128 ──────────
# 4-column groups at stride 32. IFFT DIF: large halfnn first.
	cmpq	$128,%r10
	jb	ifft_after_optD
	cmpq	$64,%r12		/* current nn must be >= 128 */
	jb	ifft_after_optD
	# Advance r8 for nn=r12*2
	leaq (%r8,%r12,8),%r8
	leaq (%r8,%r12,8),%r8		/* r8 = W(nn/2) = W(halfnn=r12) trig */
	movq %r8,%r11			/* r11 = W64 trig base */
	shr  $1,%r12			/* r12 = next halfnn (now halfnn for second stage) */
	leaq (%r8,%r12,8),%r8
	leaq (%r8,%r12,8),%r8		/* r8 = W32 trig base */
	movq %r8,%r13			/* r13 = W32 trig base */
	movq $0,%rax
	leaq (,%r10,8),%r14		/* r14 = ns4*8 */
.p2align 4
ifft_optD_outer:
	movq %r11,%rcx			/* reset W64 trig */
	movq %r13,%rdx			/* reset W32 trig */
	movq %rax,%rbx
	leaq 256(%rax),%r9		/* end of inner (32 doubles = 256 bytes) */
.p2align 4
ifft_optD_inner:
	vmovapd	(%rdi,%rbx),%ymm0
	vmovapd	256(%rdi,%rbx),%ymm1
	vmovapd	512(%rdi,%rbx),%ymm2
	vmovapd	768(%rdi,%rbx),%ymm3
	vmovapd	(%rsi,%rbx),%ymm4
	vmovapd	256(%rsi,%rbx),%ymm5
	vmovapd	512(%rsi,%rbx),%ymm6
	vmovapd	768(%rsi,%rbx),%ymm7
	# W64[off] for DIF halfnn=64
	vmovapd	(%rcx),%ymm8
	vmovapd	32(%rcx),%ymm9
	# halfnn=64 DIF A: (ymm0,ymm2)
	vsubpd	%ymm2,%ymm0,%ymm10
	vsubpd	%ymm6,%ymm4,%ymm11
	vaddpd	%ymm2,%ymm0,%ymm0
	vaddpd	%ymm6,%ymm4,%ymm4
	vmulpd	%ymm10,%ymm8,%ymm2
	vfnmadd231pd %ymm11,%ymm9,%ymm2
	vmulpd	%ymm10,%ymm9,%ymm6
	vfmadd231pd  %ymm11,%ymm8,%ymm6
	# W64[off+32] for pair B
	vmovapd	512(%rcx),%ymm8
	vmovapd	544(%rcx),%ymm9
	# halfnn=64 DIF B: (ymm1,ymm3)
	vsubpd	%ymm3,%ymm1,%ymm10
	vsubpd	%ymm7,%ymm5,%ymm11
	vaddpd	%ymm3,%ymm1,%ymm1
	vaddpd	%ymm7,%ymm5,%ymm5
	vmulpd	%ymm10,%ymm8,%ymm3
	vfnmadd231pd %ymm11,%ymm9,%ymm3
	vmulpd	%ymm10,%ymm9,%ymm7
	vfmadd231pd  %ymm11,%ymm8,%ymm7
	# W32[off] for DIF halfnn=32
	vmovapd	(%rdx),%ymm8
	vmovapd	32(%rdx),%ymm9
	# halfnn=32 DIF A: (ymm0,ymm1)
	vsubpd	%ymm1,%ymm0,%ymm10
	vsubpd	%ymm5,%ymm4,%ymm11
	vaddpd	%ymm1,%ymm0,%ymm0
	vaddpd	%ymm5,%ymm4,%ymm4
	vmulpd	%ymm10,%ymm8,%ymm1
	vfnmadd231pd %ymm11,%ymm9,%ymm1
	vmulpd	%ymm10,%ymm9,%ymm5
	vfmadd231pd  %ymm11,%ymm8,%ymm5
	# halfnn=32 DIF B: (ymm2,ymm3) with same W32
	vsubpd	%ymm3,%ymm2,%ymm10
	vsubpd	%ymm7,%ymm6,%ymm11
	vaddpd	%ymm3,%ymm2,%ymm2
	vaddpd	%ymm7,%ymm6,%ymm6
	vmulpd	%ymm10,%ymm8,%ymm3
	vfnmadd231pd %ymm11,%ymm9,%ymm3
	vmulpd	%ymm10,%ymm9,%ymm7
	vfmadd231pd  %ymm11,%ymm8,%ymm7
	# Store
	vmovapd	%ymm0,(%rdi,%rbx)
	vmovapd	%ymm1,256(%rdi,%rbx)
	vmovapd	%ymm2,512(%rdi,%rbx)
	vmovapd	%ymm3,768(%rdi,%rbx)
	vmovapd	%ymm4,(%rsi,%rbx)
	vmovapd	%ymm5,256(%rsi,%rbx)
	vmovapd	%ymm6,512(%rsi,%rbx)
	vmovapd	%ymm7,768(%rsi,%rbx)
	addq	$32,%rbx
	leaq	64(%rcx),%rcx
	leaq	64(%rdx),%rdx
	cmpq	%r9,%rbx
	jb	ifft_optD_inner
	leaq	1024(%rax),%rax		/* next super-block: 128 elements × 8 bytes */
	cmpq	%r14,%rax
	jb	ifft_optD_outer
	shr $1,%r12			/* advance to next halfnn */
	movq %r10,%r9			/* restore r9 = ns4 (clobbered by inner end) */
ifft_after_optD:
	movq %r10,%r9			/* ensure r9 = ns4 */

# ── Pass 3: Fused halfnn=16+8 ("Option C", DIF order) for ns4>=32 ────────────
	cmpq	$32,%r10
	jb	ifft_after_optC
	cmpq	$16,%r12
	jb	ifft_after_optC
	leaq (%r8,%r12,8),%r8
	leaq (%r8,%r12,8),%r8		/* r8 = W16 trig base */
	movq %r8,%r11			/* r11 = W16 trig base */
	shr  $1,%r12
	leaq (%r8,%r12,8),%r8
	leaq (%r8,%r12,8),%r8		/* r8 = W8 trig base */
	movq %r8,%r13			/* r13 = W8 trig base */
	movq $0,%rax
	leaq (,%r10,8),%r14
.p2align 4
ifft_optC_outer:
	movq %r11,%rcx
	movq %r13,%rdx
	movq %rax,%rbx
	leaq 64(%rax),%r9
.p2align 4
ifft_optC_inner:
	vmovapd	(%rdi,%rbx),%ymm0
	vmovapd	64(%rdi,%rbx),%ymm1
	vmovapd	128(%rdi,%rbx),%ymm2
	vmovapd	192(%rdi,%rbx),%ymm3
	vmovapd	(%rsi,%rbx),%ymm4
	vmovapd	64(%rsi,%rbx),%ymm5
	vmovapd	128(%rsi,%rbx),%ymm6
	vmovapd	192(%rsi,%rbx),%ymm7
	# W16[off] for DIF halfnn=16 pair A
	vmovapd	(%rcx),%ymm8
	vmovapd	32(%rcx),%ymm9
	# halfnn=16 DIF A: (ymm0,ymm2)
	vsubpd	%ymm2,%ymm0,%ymm10
	vsubpd	%ymm6,%ymm4,%ymm11
	vaddpd	%ymm2,%ymm0,%ymm0
	vaddpd	%ymm6,%ymm4,%ymm4
	vmulpd	%ymm10,%ymm8,%ymm2
	vfnmadd231pd %ymm11,%ymm9,%ymm2
	vmulpd	%ymm10,%ymm9,%ymm6
	vfmadd231pd  %ymm11,%ymm8,%ymm6
	# W16[off+8] for pair B
	vmovapd	128(%rcx),%ymm8
	vmovapd	160(%rcx),%ymm9
	# halfnn=16 DIF B: (ymm1,ymm3)
	vsubpd	%ymm3,%ymm1,%ymm10
	vsubpd	%ymm7,%ymm5,%ymm11
	vaddpd	%ymm3,%ymm1,%ymm1
	vaddpd	%ymm7,%ymm5,%ymm5
	vmulpd	%ymm10,%ymm8,%ymm3
	vfnmadd231pd %ymm11,%ymm9,%ymm3
	vmulpd	%ymm10,%ymm9,%ymm7
	vfmadd231pd  %ymm11,%ymm8,%ymm7
	# W8[off] for DIF halfnn=8
	vmovapd	(%rdx),%ymm8
	vmovapd	32(%rdx),%ymm9
	# halfnn=8 DIF A: (ymm0,ymm1)
	vsubpd	%ymm1,%ymm0,%ymm10
	vsubpd	%ymm5,%ymm4,%ymm11
	vaddpd	%ymm1,%ymm0,%ymm0
	vaddpd	%ymm5,%ymm4,%ymm4
	vmulpd	%ymm10,%ymm8,%ymm1
	vfnmadd231pd %ymm11,%ymm9,%ymm1
	vmulpd	%ymm10,%ymm9,%ymm5
	vfmadd231pd  %ymm11,%ymm8,%ymm5
	# halfnn=8 DIF B: (ymm2,ymm3) with same W8
	vsubpd	%ymm3,%ymm2,%ymm10
	vsubpd	%ymm7,%ymm6,%ymm11
	vaddpd	%ymm3,%ymm2,%ymm2
	vaddpd	%ymm7,%ymm6,%ymm6
	vmulpd	%ymm10,%ymm8,%ymm3
	vfnmadd231pd %ymm11,%ymm9,%ymm3
	vmulpd	%ymm10,%ymm9,%ymm7
	vfmadd231pd  %ymm11,%ymm8,%ymm7
	# Store
	vmovapd	%ymm0,(%rdi,%rbx)
	vmovapd	%ymm1,64(%rdi,%rbx)
	vmovapd	%ymm2,128(%rdi,%rbx)
	vmovapd	%ymm3,192(%rdi,%rbx)
	vmovapd	%ymm4,(%rsi,%rbx)
	vmovapd	%ymm5,64(%rsi,%rbx)
	vmovapd	%ymm6,128(%rsi,%rbx)
	vmovapd	%ymm7,192(%rsi,%rbx)
	addq	$32,%rbx
	leaq	64(%rcx),%rcx
	leaq	64(%rdx),%rdx
	cmpq	%r9,%rbx
	jb	ifft_optC_inner
	leaq	256(%rax),%rax		/* next super-block: 32 elements × 8 bytes */
	cmpq	%r14,%rax
	jb	ifft_optC_outer
	shr $1,%r12
	movq %r10,%r9
ifft_after_optC:
	movq %r10,%r9

# ── General IFFT DIF loop for remaining halfnn ────────────────────────────────
	cmpq $8,%r12
	jbe ifft_before_small
ifft_nnloop:
	movq %r12,%r13
	shr  $1,%r13
	leaq (%r8,%r12,8),%r8
	leaq (%r8,%r12,8),%r8
	movq $0,%r11
ifft_blockloop:
	leaq (%rdi,%r11,8),%rax
	leaq (%rsi,%r11,8),%rbx
	leaq (%rax,%r13,8),%rcx
	leaq (%rbx,%r13,8),%rdx
	movq $0,%r9
	movq %r8,%r14
ifft_offloop:
	vmovapd (%rax,%r9,8),%ymm0
	vmovapd (%rbx,%r9,8),%ymm1
	vmovapd (%rcx,%r9,8),%ymm2
	vmovapd (%rdx,%r9,8),%ymm3
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
	movq %r10,%r9		/* restore r9=ns4 */
	cmpq %r9,%r11
	jb ifft_blockloop
	movq %r13,%r12
	cmpq $8,%r12
	ja ifft_nnloop

ifft_before_small:
# ── Pass N: Fused size-8 + size-4 + size-2 ───────────────────────────────────
	vmovapd     ifft_s8cos(%rip), %ymm15
	vmovapd     ifft_s8sin(%rip), %ymm14
	vmovapd     ifft_negmask_s4a(%rip), %ymm13
	vmovapd     ifft_negmask_s4b(%rip), %ymm12
	vmovapd     ifft_negmask_s4c(%rip), %ymm11
	vmovapd     ifft_negmask_s2(%rip), %ymm10
	# Advance r8 past halfnn=4 trig
	leaq (%r8,%r12,8),%r8
	leaq (%r8,%r12,8),%r8
	movq $0,%rax
	movq %rdi,%r11
	movq %rsi,%r12
.p2align 4
ifft_fused_842:
	vmovapd (%r11),%ymm0
	vmovapd 32(%r11),%ymm1
	vmovapd (%r12),%ymm2
	vmovapd 32(%r12),%ymm3
	# Size-8 DIF
	vaddpd %ymm0,%ymm1,%ymm4
	vaddpd %ymm2,%ymm3,%ymm5
	vsubpd %ymm1,%ymm0,%ymm6
	vsubpd %ymm3,%ymm2,%ymm7
	vmulpd %ymm6,%ymm15,%ymm0
	vfnmadd231pd %ymm7,%ymm14,%ymm0
	vmulpd %ymm6,%ymm14,%ymm2
	vfmadd231pd %ymm7,%ymm15,%ymm2
	# Size-4 DIF pair 0
	vshufpd $10,%ymm5,%ymm4,%ymm6
	vshufpd $10,%ymm4,%ymm5,%ymm7
	vperm2f128 $0x20,%ymm6,%ymm4,%ymm8
	vperm2f128 $0x31,%ymm6,%ymm4,%ymm9
	vperm2f128 $0x20,%ymm7,%ymm5,%ymm1
	vperm2f128 $0x31,%ymm7,%ymm5,%ymm3
	vmulpd  %ymm8,%ymm13,%ymm8
	vfmadd231pd %ymm9,%ymm12,%ymm8
	vfmadd231pd %ymm3,%ymm11,%ymm1
	vmovapd %ymm8,%ymm4
	vmovapd %ymm1,%ymm5
	# Size-4 DIF pair 1
	vshufpd $10,%ymm2,%ymm0,%ymm6
	vshufpd $10,%ymm0,%ymm2,%ymm7
	vperm2f128 $0x20,%ymm6,%ymm0,%ymm8
	vperm2f128 $0x31,%ymm6,%ymm0,%ymm9
	vperm2f128 $0x20,%ymm7,%ymm2,%ymm1
	vperm2f128 $0x31,%ymm7,%ymm2,%ymm3
	vmulpd  %ymm8,%ymm13,%ymm8
	vfmadd231pd %ymm9,%ymm12,%ymm8
	vfmadd231pd %ymm3,%ymm11,%ymm1
	vmovapd %ymm8,%ymm0
	vmovapd %ymm1,%ymm2
	# Size-2
	vshufpd $0,%ymm4,%ymm4,%ymm6
	vshufpd $15,%ymm4,%ymm4,%ymm4
	vfmadd231pd %ymm4,%ymm10,%ymm6
	vshufpd $0,%ymm0,%ymm0,%ymm7
	vshufpd $15,%ymm0,%ymm0,%ymm0
	vfmadd231pd %ymm0,%ymm10,%ymm7
	vshufpd $0,%ymm5,%ymm5,%ymm8
	vshufpd $15,%ymm5,%ymm5,%ymm5
	vfmadd231pd %ymm5,%ymm10,%ymm8
	vshufpd $0,%ymm2,%ymm2,%ymm9
	vshufpd $15,%ymm2,%ymm2,%ymm2
	vfmadd231pd %ymm2,%ymm10,%ymm9
	vmovapd %ymm6,(%r11)
	vmovapd %ymm7,32(%r11)
	vmovapd %ymm8,(%r12)
	vmovapd %ymm9,32(%r12)
	leaq 64(%r11),%r11
	leaq 64(%r12),%r12
	addq $8,%rax
	cmpq %r10,%rax
	jb ifft_fused_842

ifft_end:
	vzeroall
	popq        %rbx
	popq        %r14
	popq        %r13
	popq        %r12
	popq        %r11
	popq        %r10
	retq

.balign 32
ifft_s8cos: .double +1.0, +0.7071067811865476, +0.0, -0.7071067811865476
ifft_s8sin: .double +0.0, +0.7071067811865476, +1.0, +0.7071067811865476
ifft_negmask_s4a: .double +1.0, +1.0, +1.0, -1.0
ifft_negmask_s4b: .double +1.0, +1.0, -1.0, +1.0
ifft_negmask_s4c: .double +1.0, +1.0, -1.0, -1.0
ifft_negmask_s2:  .double +1.0, -1.0, +1.0, -1.0

#if !__APPLE__
	.size	ifft, .-ifft
#endif
