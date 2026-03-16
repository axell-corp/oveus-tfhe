	.file	"spqlios-fft-fma.s"
#if !__APPLE__
	.section .note.GNU-stack,"",%progbits
#endif
	.text
	.p2align 4
#if !__APPLE__
	.globl	fft
	.type	fft, @function
fft:
#else
	.globl	_fft
_fft:
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
	movq	%rdx, %r9
	shr	$2,%r9
	leaq	(%rdi,%r9,8),%rsi

# ── Pass 1: Fused size-2 + size-4 + size-8 (halfnn=4) ────────────────────────
	vmovapd     fft_negmask_s2(%rip), %ymm15
	vmovapd     fft_negmask_s4im(%rip), %ymm14
	vmovapd     fft_negmask_s4re(%rip), %ymm13
	vmovapd     fft_s8cos(%rip), %ymm12
	vmovapd     fft_s8sin(%rip), %ymm11
	movq	%rdi,%r10
	movq	%rsi,%r11
	movq	$0,%rax
.p2align 4
fft_fused_248:
	vmovapd (%r10),%ymm0
	vmovapd 32(%r10),%ymm1
	vmovapd (%r11),%ymm2
	vmovapd 32(%r11),%ymm3
	vshufpd $0,%ymm0,%ymm0,%ymm4
	vshufpd $15,%ymm0,%ymm0,%ymm0
	vfmadd231pd %ymm0,%ymm15,%ymm4
	vshufpd $0,%ymm1,%ymm1,%ymm5
	vshufpd $15,%ymm1,%ymm1,%ymm1
	vfmadd231pd %ymm1,%ymm15,%ymm5
	vshufpd $0,%ymm2,%ymm2,%ymm6
	vshufpd $15,%ymm2,%ymm2,%ymm2
	vfmadd231pd %ymm2,%ymm15,%ymm6
	vshufpd $0,%ymm3,%ymm3,%ymm7
	vshufpd $15,%ymm3,%ymm3,%ymm3
	vfmadd231pd %ymm3,%ymm15,%ymm7
	vperm2f128 $0x20,%ymm4,%ymm4,%ymm0
	vperm2f128 $0x20,%ymm6,%ymm6,%ymm2
	vperm2f128 $0x31,%ymm4,%ymm4,%ymm8
	vperm2f128 $0x31,%ymm6,%ymm6,%ymm9
	vshufpd $10,%ymm9,%ymm8,%ymm10
	vshufpd $10,%ymm8,%ymm9,%ymm9
	vfmadd231pd %ymm10,%ymm13,%ymm0
	vfmadd231pd %ymm9,%ymm14,%ymm2
	vperm2f128 $0x20,%ymm5,%ymm5,%ymm4
	vperm2f128 $0x20,%ymm7,%ymm7,%ymm6
	vperm2f128 $0x31,%ymm5,%ymm5,%ymm8
	vperm2f128 $0x31,%ymm7,%ymm7,%ymm9
	vshufpd $10,%ymm9,%ymm8,%ymm10
	vshufpd $10,%ymm8,%ymm9,%ymm9
	vfmadd231pd %ymm10,%ymm13,%ymm4
	vfmadd231pd %ymm9,%ymm14,%ymm6
	vmulpd	%ymm4,%ymm12,%ymm8
	vmulpd	%ymm4,%ymm11,%ymm9
	vfnmadd231pd %ymm6,%ymm11,%ymm8
	vfmadd231pd  %ymm6,%ymm12,%ymm9
	vsubpd	%ymm8,%ymm0,%ymm4
	vsubpd	%ymm9,%ymm2,%ymm6
	vaddpd	%ymm8,%ymm0,%ymm0
	vaddpd	%ymm9,%ymm2,%ymm2
	vmovapd %ymm0,(%r10)
	vmovapd %ymm4,32(%r10)
	vmovapd %ymm2,(%r11)
	vmovapd %ymm6,32(%r11)
	leaq 64(%r10),%r10
	leaq 64(%r11),%r11
	addq $8,%rax
	cmpq %r9,%rax
	jb fft_fused_248

# ── Trig pointer setup ───────────────────────────────────────────────────────
	leaq	64(%r8),%rdx		/* skip halfnn=4 trig (64 bytes) */
	movq	%r9,%rbx
	shr	$1,%rbx			/* rbx = ns4/2 = last halfnn */

# ── Pass 2: Fused halfnn=8 + halfnn=16 ("Option C") ──────────────────────────
# 4-column groups at stride 8: positions [sb+off, sb+off+8, sb+off+16, sb+off+24]
# Super-block = 32 elements. Inner loop: off=0,4 (2 iters).
# halfnn=8 butterflies: (col0,col1) & (col2,col3) with W8[off]
# halfnn=16 butterflies: (col0',col2') with W16[off] & (col1',col3') with W16[off+8]
	cmpq	$32,%r9
	jb	fft_after_optC		/* ns4 < 32: skip */
	movq	%rdx,%r10		/* r10 = W8 trig base */
	leaq	128(%rdx),%r11		/* r11 = W16 trig base (W8 is 128 bytes for halfnn=8) */
	movq	$0,%rax			/* rax = super-block byte offset */
	leaq	(,%r9,8),%r13		/* r13 = ns4*8 = total re bytes */
.p2align 4
fft_optC_outer:
	movq	%r10,%r12		/* reset W8 trig to base */
	movq	%r11,%r14		/* reset W16 trig to base */
	movq	%rax,%rcx		/* rcx = inner byte offset = sb start */
	leaq	64(%rax),%rbx		/* rbx = end of this super-block inner (sb+8 doubles=64 bytes) */
.p2align 4
fft_optC_inner:
	# Load 4 columns (stride 8 elements = 64 bytes)
	vmovapd	(%rdi,%rcx),%ymm0		/* re[off] */
	vmovapd	64(%rdi,%rcx),%ymm1		/* re[off+8] */
	vmovapd	128(%rdi,%rcx),%ymm2		/* re[off+16] */
	vmovapd	192(%rdi,%rcx),%ymm3		/* re[off+24] */
	vmovapd	(%rsi,%rcx),%ymm4		/* im[off] */
	vmovapd	64(%rsi,%rcx),%ymm5		/* im[off+8] */
	vmovapd	128(%rsi,%rcx),%ymm6		/* im[off+16] */
	vmovapd	192(%rsi,%rcx),%ymm7		/* im[off+24] */
	# W8 trig (same for both halfnn=8 pairs)
	vmovapd	(%r12),%ymm8
	vmovapd	32(%r12),%ymm9
	# halfnn=8 butterfly A: (ymm0,ymm1) × (ymm4,ymm5)
	vmulpd	%ymm1,%ymm8,%ymm10
	vmulpd	%ymm1,%ymm9,%ymm11
	vfnmadd231pd %ymm5,%ymm9,%ymm10
	vfmadd231pd  %ymm5,%ymm8,%ymm11
	vsubpd	%ymm10,%ymm0,%ymm1
	vsubpd	%ymm11,%ymm4,%ymm5
	vaddpd	%ymm10,%ymm0,%ymm0
	vaddpd	%ymm11,%ymm4,%ymm4
	# halfnn=8 butterfly B: (ymm2,ymm3) × (ymm6,ymm7)
	vmulpd	%ymm3,%ymm8,%ymm10
	vmulpd	%ymm3,%ymm9,%ymm11
	vfnmadd231pd %ymm7,%ymm9,%ymm10
	vfmadd231pd  %ymm7,%ymm8,%ymm11
	vsubpd	%ymm10,%ymm2,%ymm3
	vsubpd	%ymm11,%ymm6,%ymm7
	vaddpd	%ymm10,%ymm2,%ymm2
	vaddpd	%ymm11,%ymm6,%ymm6
	# W16 trig for pair A: W16[off]
	vmovapd	(%r14),%ymm8
	vmovapd	32(%r14),%ymm9
	# halfnn=16 butterfly A: (ymm0,ymm2) × (ymm4,ymm6)
	vmulpd	%ymm2,%ymm8,%ymm10
	vmulpd	%ymm2,%ymm9,%ymm11
	vfnmadd231pd %ymm6,%ymm9,%ymm10
	vfmadd231pd  %ymm6,%ymm8,%ymm11
	vsubpd	%ymm10,%ymm0,%ymm2
	vsubpd	%ymm11,%ymm4,%ymm6
	vaddpd	%ymm10,%ymm0,%ymm0
	vaddpd	%ymm11,%ymm4,%ymm4
	# W16 trig for pair B: W16[off+8]
	vmovapd	128(%r14),%ymm8
	vmovapd	160(%r14),%ymm9
	# halfnn=16 butterfly B: (ymm1,ymm3) × (ymm5,ymm7)
	vmulpd	%ymm3,%ymm8,%ymm10
	vmulpd	%ymm3,%ymm9,%ymm11
	vfnmadd231pd %ymm7,%ymm9,%ymm10
	vfmadd231pd  %ymm7,%ymm8,%ymm11
	vsubpd	%ymm10,%ymm1,%ymm3
	vsubpd	%ymm11,%ymm5,%ymm7
	vaddpd	%ymm10,%ymm1,%ymm1
	vaddpd	%ymm11,%ymm5,%ymm5
	# Store 4 columns
	vmovapd	%ymm0,(%rdi,%rcx)
	vmovapd	%ymm1,64(%rdi,%rcx)
	vmovapd	%ymm2,128(%rdi,%rcx)
	vmovapd	%ymm3,192(%rdi,%rcx)
	vmovapd	%ymm4,(%rsi,%rcx)
	vmovapd	%ymm5,64(%rsi,%rcx)
	vmovapd	%ymm6,128(%rsi,%rcx)
	vmovapd	%ymm7,192(%rsi,%rcx)
	# Advance inner
	addq	$32,%rcx		/* +4 doubles */
	leaq	64(%r12),%r12		/* W8 trig: +64 bytes */
	leaq	64(%r14),%r14		/* W16 trig: +64 bytes */
	cmpq	%rbx,%rcx
	jb	fft_optC_inner
	# Advance to next super-block: skip from sb+64 to sb+256 (32 elements × 8 bytes)
	leaq	256(%rax),%rax		/* next super-block: 32 elements × 8 bytes */
	cmpq	%r13,%rax
	jb	fft_optC_outer
	# Advance rdx past W8+W16 trig
	leaq	384(%rdx),%rdx		/* 128 + 256 = 384 bytes */
	movq	$32,%rax		/* next halfnn to process */
fft_after_optC:

# ── Pass 3: Fused halfnn=32 + halfnn=64 ("Option D") ─────────────────────────
# 4-column groups at stride 32: positions [sb+off, sb+off+32, sb+off+64, sb+off+96]
# Super-block = 128 elements. Inner loop: off=0..31 step 4 (8 iters).
	cmpq	$128,%r9
	jb	fft_after_optD
	movq	%rdx,%r10		/* r10 = W32 trig base */
	leaq	512(%rdx),%r11		/* r11 = W64 trig base (W32 is 512 bytes) */
	movq	$0,%rax
	leaq	(,%r9,8),%r13
.p2align 4
fft_optD_outer:
	movq	%r10,%r12		/* reset W32 trig */
	movq	%r11,%r14		/* reset W64 trig */
	movq	%rax,%rcx
	leaq	256(%rax),%rbx		/* end of inner (32 doubles = 256 bytes) */
.p2align 4
fft_optD_inner:
	vmovapd	(%rdi,%rcx),%ymm0
	vmovapd	256(%rdi,%rcx),%ymm1
	vmovapd	512(%rdi,%rcx),%ymm2
	vmovapd	768(%rdi,%rcx),%ymm3
	vmovapd	(%rsi,%rcx),%ymm4
	vmovapd	256(%rsi,%rcx),%ymm5
	vmovapd	512(%rsi,%rcx),%ymm6
	vmovapd	768(%rsi,%rcx),%ymm7
	vmovapd	(%r12),%ymm8
	vmovapd	32(%r12),%ymm9
	# halfnn=32 butterfly A: (ymm0,ymm1)
	vmulpd	%ymm1,%ymm8,%ymm10
	vmulpd	%ymm1,%ymm9,%ymm11
	vfnmadd231pd %ymm5,%ymm9,%ymm10
	vfmadd231pd  %ymm5,%ymm8,%ymm11
	vsubpd	%ymm10,%ymm0,%ymm1
	vsubpd	%ymm11,%ymm4,%ymm5
	vaddpd	%ymm10,%ymm0,%ymm0
	vaddpd	%ymm11,%ymm4,%ymm4
	# halfnn=32 butterfly B: (ymm2,ymm3)
	vmulpd	%ymm3,%ymm8,%ymm10
	vmulpd	%ymm3,%ymm9,%ymm11
	vfnmadd231pd %ymm7,%ymm9,%ymm10
	vfmadd231pd  %ymm7,%ymm8,%ymm11
	vsubpd	%ymm10,%ymm2,%ymm3
	vsubpd	%ymm11,%ymm6,%ymm7
	vaddpd	%ymm10,%ymm2,%ymm2
	vaddpd	%ymm11,%ymm6,%ymm6
	# W64[off] for pair A
	vmovapd	(%r14),%ymm8
	vmovapd	32(%r14),%ymm9
	# halfnn=64 butterfly A: (ymm0,ymm2)
	vmulpd	%ymm2,%ymm8,%ymm10
	vmulpd	%ymm2,%ymm9,%ymm11
	vfnmadd231pd %ymm6,%ymm9,%ymm10
	vfmadd231pd  %ymm6,%ymm8,%ymm11
	vsubpd	%ymm10,%ymm0,%ymm2
	vsubpd	%ymm11,%ymm4,%ymm6
	vaddpd	%ymm10,%ymm0,%ymm0
	vaddpd	%ymm11,%ymm4,%ymm4
	# W64[off+32] for pair B
	vmovapd	512(%r14),%ymm8
	vmovapd	544(%r14),%ymm9
	# halfnn=64 butterfly B: (ymm1,ymm3)
	vmulpd	%ymm3,%ymm8,%ymm10
	vmulpd	%ymm3,%ymm9,%ymm11
	vfnmadd231pd %ymm7,%ymm9,%ymm10
	vfmadd231pd  %ymm7,%ymm8,%ymm11
	vsubpd	%ymm10,%ymm1,%ymm3
	vsubpd	%ymm11,%ymm5,%ymm7
	vaddpd	%ymm10,%ymm1,%ymm1
	vaddpd	%ymm11,%ymm5,%ymm5
	# Store
	vmovapd	%ymm0,(%rdi,%rcx)
	vmovapd	%ymm1,256(%rdi,%rcx)
	vmovapd	%ymm2,512(%rdi,%rcx)
	vmovapd	%ymm3,768(%rdi,%rcx)
	vmovapd	%ymm4,(%rsi,%rcx)
	vmovapd	%ymm5,256(%rsi,%rcx)
	vmovapd	%ymm6,512(%rsi,%rcx)
	vmovapd	%ymm7,768(%rsi,%rcx)
	addq	$32,%rcx
	leaq	64(%r12),%r12
	leaq	64(%r14),%r14
	cmpq	%rbx,%rcx
	jb	fft_optD_inner
	# Advance to next super-block (128 elements = 1024 bytes, already at sb+256)
	leaq	1024(%rax),%rax		/* next super-block: 128 elements × 8 bytes */
	cmpq	%r13,%rax
	jb	fft_optD_outer
	# Advance rdx past W32+W64 trig
	leaq	1536(%rdx),%rdx		/* 512 + 1024 = 1536 bytes */
	movq	$128,%rax		/* next halfnn */
fft_after_optD:

# ── General loop for remaining halfnn values ──────────────────────────────────
	movq	%r9,%rbx
	shr	$1,%rbx			/* rbx = ns4/2 = last halfnn */
	cmpq	%rbx,%rax
	ja	fftbeforefinal
	je	fft_last_halfnn_fused
ffthalfnnloop:
	cmpq	%rbx,%rax
	je	fft_last_halfnn_fused
	movq	$0,%rcx
fftblockloop:
	leaq (%rdi,%rcx,8),%r10
	leaq (%rsi,%rcx,8),%r11
	leaq (%r10,%rax,8),%r12
	leaq (%r11,%rax,8),%r13
	movq %rdx,%r14
	pushq %rcx
	movq $0,%rcx
fftoffloop:
	vmovapd (%r10),%ymm0
	vmovapd (%r11),%ymm1
	vmovapd (%r12),%ymm2
	vmovapd (%r13),%ymm3
	vmovapd (%r14),%ymm4
	vmovapd 32(%r14),%ymm5
	vmulpd	%ymm2,%ymm4,%ymm6
	vmulpd	%ymm2,%ymm5,%ymm7
	vfnmadd231pd %ymm3,%ymm5,%ymm6
	vfmadd231pd %ymm3,%ymm4,%ymm7
	vsubpd	%ymm6,%ymm0,%ymm2
	vsubpd	%ymm7,%ymm1,%ymm3
	vaddpd	%ymm6,%ymm0,%ymm0
	vaddpd	%ymm7,%ymm1,%ymm1
	vmovapd %ymm0,(%r10)
	vmovapd %ymm1,(%r11)
	vmovapd %ymm2,(%r12)
	vmovapd %ymm3,(%r13)
	leaq 	32(%r10),%r10
	leaq	32(%r11),%r11
	leaq 	32(%r12),%r12
	leaq 	32(%r13),%r13
	leaq 	64(%r14),%r14
	addq 	$4,%rcx
	cmpq	%rax,%rcx
	jb 	fftoffloop
	popq    %rcx
	leaq	(%rcx,%rax,2),%rcx
	cmpq	%r9,%rcx
	jb 	fftblockloop
	shlq	$1,%rax
	leaq	(%rdx,%rax,8),%rdx
	cmpq	%r9,%rax
	jb ffthalfnnloop

fftbeforefinal:
	movq $0,%rax
	movq %rdi,%r10
	movq %rsi,%r11
fftfinalloop:
	vmovapd	(%r10),%ymm0
	vmovapd	(%r11),%ymm1
	vmovapd (%rdx),%ymm2
	vmovapd 32(%rdx),%ymm3
	vmulpd %ymm0,%ymm2,%ymm4
	vmulpd %ymm0,%ymm3,%ymm5
	vfnmadd231pd %ymm1,%ymm3,%ymm4
	vfmadd231pd %ymm1,%ymm2,%ymm5
	vmovapd %ymm4,(%r10)
	vmovapd %ymm5,(%r11)
	leaq	32(%r10),%r10
	leaq	32(%r11),%r11
	leaq	64(%rdx),%rdx
	addq	$4,%rax
	cmpq	%r9,%rax
	jb fftfinalloop
	jmp	fftend

# ── Fused last butterfly + final twist ────────────────────────────────────────
fft_last_halfnn_fused:
	leaq (%rdx,%rax,8),%r10
	leaq (%r10,%rax,8),%r10
	movq %rdi,%r11
	movq %rsi,%r12
	leaq (%rdi,%rax,8),%r13
	leaq (%rsi,%rax,8),%r14
	movq $0,%rcx
.p2align 4
fft_last_fused_loop:
	vmovapd (%r11),%ymm0
	vmovapd (%r12),%ymm1
	vmovapd (%r13),%ymm2
	vmovapd (%r14),%ymm3
	vmovapd (%rdx),%ymm4
	vmovapd 32(%rdx),%ymm5
	vmulpd	%ymm2,%ymm4,%ymm6
	vmulpd	%ymm2,%ymm5,%ymm7
	vfnmadd231pd %ymm3,%ymm5,%ymm6
	vfmadd231pd %ymm3,%ymm4,%ymm7
	vsubpd	%ymm6,%ymm0,%ymm2
	vsubpd	%ymm7,%ymm1,%ymm3
	vaddpd	%ymm6,%ymm0,%ymm0
	vaddpd	%ymm7,%ymm1,%ymm1
	vmovapd (%r10),%ymm4
	vmovapd 32(%r10),%ymm5
	vmulpd %ymm0,%ymm4,%ymm6
	vmulpd %ymm0,%ymm5,%ymm7
	vfnmadd231pd %ymm1,%ymm5,%ymm6
	vfmadd231pd %ymm1,%ymm4,%ymm7
	vmovapd %ymm6,(%r11)
	vmovapd %ymm7,(%r12)
	leaq (%r10,%rax,8),%rbx
	leaq (%rbx,%rax,8),%rbx
	vmovapd (%rbx),%ymm4
	vmovapd 32(%rbx),%ymm5
	vmulpd %ymm2,%ymm4,%ymm6
	vmulpd %ymm2,%ymm5,%ymm7
	vfnmadd231pd %ymm3,%ymm5,%ymm6
	vfmadd231pd %ymm3,%ymm4,%ymm7
	vmovapd %ymm6,(%r13)
	vmovapd %ymm7,(%r14)
	leaq 32(%r11),%r11
	leaq 32(%r12),%r12
	leaq 32(%r13),%r13
	leaq 32(%r14),%r14
	leaq 64(%rdx),%rdx
	leaq 64(%r10),%r10
	addq $4,%rcx
	cmpq %rax,%rcx
	jb fft_last_fused_loop

fftend:
	vzeroall
	popq        %rbx
	popq        %r14
	popq        %r13
	popq        %r12
	popq        %r11
	popq        %r10
	retq

.balign 32
fft_negmask_s2:    .double +1.0, -1.0, +1.0, -1.0
fft_negmask_s4im:  .double +1.0, -1.0, -1.0, +1.0
fft_negmask_s4re:  .double +1.0, +1.0, -1.0, -1.0
fft_s8cos: .double +1.0, +0.7071067811865476, +0.0, -0.7071067811865476
fft_s8sin: .double +0.0, -0.7071067811865476, -1.0, -0.7071067811865476

#if !__APPLE__
	.size	fft, .-fft
#endif
