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
//void fft(const void* tables, double* c) {

	/* Save registers */
	pushq       %r10
	pushq       %r11
	pushq       %r12
	pushq       %r13
	pushq       %r14
	pushq       %rbx

	/* Permute registers for better variable names */
	movq        %rdi, %rax
	movq        %rsi, %rdi      /* rdi: base of the real data CONSTANT */

	/* Load struct FftTables fields */
	movq         0(%rax), %rdx  /* rdx: n */
	movq         8(%rax), %r8   /* r8: trig_tables base (CONSTANT) */

	movq	%rdx, %r9
	shr	$2,%r9              /* r9: ns4 CONSTANT */
	leaq	(%rdi,%r9,8),%rsi   /* rsi: base of imaginary data CONSTANT */

# ── Fused size-2 + size-4 + size-8 (halfnn=4) pass ───────────────────────────
# Processes 8 doubles per iteration (2 consecutive YMM blocks).
# Does all 3 stages in-register: load once, store once.
# Saves 2 full load-store round trips vs separate passes.
#
# Register map:
#   ymm15 = [+1,-1,+1,-1]  (size-2 negation)
#   ymm14 = [+1,-1,-1,+1]  (size-4 im)
#   ymm13 = [+1,+1,-1,-1]  (size-4 re)
#   ymm12 = fftsize8cos     (size-8 twiddle cos)
#   ymm11 = fftsize8sin     (size-8 twiddle sin)
#   ymm0-10 = data and temps
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
	# Load 2 consecutive blocks (8 re + 8 im = 4 YMM)
	vmovapd (%r10),%ymm0         /* re[0..3] */
	vmovapd 32(%r10),%ymm1       /* re[4..7] */
	vmovapd (%r11),%ymm2         /* im[0..3] */
	vmovapd 32(%r11),%ymm3       /* im[4..7] */

	# ── Size-2 butterfly on all 4 YMMs ──
	# [a0,a1,a2,a3] → [a0+a1, a0-a1, a2+a3, a2-a3]
	vshufpd $0,%ymm0,%ymm0,%ymm4
	vshufpd $15,%ymm0,%ymm0,%ymm0
	vfmadd231pd %ymm0,%ymm15,%ymm4   /* re_lo after s2 */
	vshufpd $0,%ymm1,%ymm1,%ymm5
	vshufpd $15,%ymm1,%ymm1,%ymm1
	vfmadd231pd %ymm1,%ymm15,%ymm5   /* re_hi after s2 */
	vshufpd $0,%ymm2,%ymm2,%ymm6
	vshufpd $15,%ymm2,%ymm2,%ymm2
	vfmadd231pd %ymm2,%ymm15,%ymm6   /* im_lo after s2 */
	vshufpd $0,%ymm3,%ymm3,%ymm7
	vshufpd $15,%ymm3,%ymm3,%ymm3
	vfmadd231pd %ymm3,%ymm15,%ymm7   /* im_hi after s2 */

	# ── Size-4 butterfly, pair 0: (ymm4=re_lo, ymm6=im_lo) → (ymm0, ymm2) ──
	vperm2f128 $0x20,%ymm4,%ymm4,%ymm0   /* dup low re  */
	vperm2f128 $0x20,%ymm6,%ymm6,%ymm2   /* dup low im  */
	vperm2f128 $0x31,%ymm4,%ymm4,%ymm8   /* dup high re */
	vperm2f128 $0x31,%ymm6,%ymm6,%ymm9   /* dup high im */
	vshufpd $10,%ymm9,%ymm8,%ymm10       /* [re_h, im_h', re_h, im_h'] */
	vshufpd $10,%ymm8,%ymm9,%ymm9        /* [im_h, re_h', im_h, re_h'] */
	vfmadd231pd %ymm10,%ymm13,%ymm0      /* re_lo result */
	vfmadd231pd %ymm9,%ymm14,%ymm2       /* im_lo result */

	# ── Size-4 butterfly, pair 1: (ymm5=re_hi, ymm7=im_hi) → (ymm4, ymm6) ──
	vperm2f128 $0x20,%ymm5,%ymm5,%ymm4   /* dup low re  */
	vperm2f128 $0x20,%ymm7,%ymm7,%ymm6   /* dup low im  */
	vperm2f128 $0x31,%ymm5,%ymm5,%ymm8   /* dup high re */
	vperm2f128 $0x31,%ymm7,%ymm7,%ymm9   /* dup high im */
	vshufpd $10,%ymm9,%ymm8,%ymm10       /* mix re/im high */
	vshufpd $10,%ymm8,%ymm9,%ymm9        /* mix im/re high */
	vfmadd231pd %ymm10,%ymm13,%ymm4      /* re_hi result */
	vfmadd231pd %ymm9,%ymm14,%ymm6       /* im_hi result */

	# ── Size-8 butterfly across the two blocks ──
	# re2 = re_hi*cos - im_hi*sin
	# im2 = re_hi*sin + im_hi*cos
	vmulpd	%ymm4,%ymm12,%ymm8           /* re_hi * cos */
	vmulpd	%ymm4,%ymm11,%ymm9           /* re_hi * sin */
	vfnmadd231pd %ymm6,%ymm11,%ymm8     /* re2 = re_hi*cos - im_hi*sin */
	vfmadd231pd  %ymm6,%ymm12,%ymm9     /* im2 = re_hi*sin + im_hi*cos */
	vsubpd	%ymm8,%ymm0,%ymm4            /* new re_hi = re_lo - re2 */
	vsubpd	%ymm9,%ymm2,%ymm6            /* new im_hi = im_lo - im2 */
	vaddpd	%ymm8,%ymm0,%ymm0            /* new re_lo = re_lo + re2 */
	vaddpd	%ymm9,%ymm2,%ymm2            /* new im_lo = im_lo + im2 */

	# Store
	vmovapd %ymm0,(%r10)
	vmovapd %ymm4,32(%r10)
	vmovapd %ymm2,(%r11)
	vmovapd %ymm6,32(%r11)
	leaq 64(%r10),%r10
	leaq 64(%r11),%r11
	addq $8,%rax
	cmpq %r9,%rax
	jb fft_fused_248

# ── General butterfly loop, starting from halfnn=8 ───────────────────────────
# (halfnn=4 was handled by the fused pass above)
# Skip the first trig table entry (halfnn=4: 1 group × 8 doubles = 64 bytes)
	leaq	64(%r8),%rdx      /* rdx: cur_tt, skip halfnn=4 entry */
	movq	$8,%rax           /* rax: halfnn, start at 8 */

	# Check if we should fuse the last butterfly with the final twist
	# For the last halfnn iteration (halfnn == ns4/2), fuse with twist
	movq	%r9,%rbx
	shr	$1,%rbx           /* rbx = ns4/2 = last halfnn value */
	cmpq	$8,%rbx
	jb	fftbeforefinal    /* ns4 < 16: no general loop at all */

ffthalfnnloop:
	cmpq	%rbx,%rax         /* is this the last halfnn? */
	je	fft_last_halfnn_fused /* yes: fuse with final twist */
	movq $0,%rcx              /* rcx: block */
fftblockloop:
	leaq (%rdi,%rcx,8),%r10   /* re0 pointer */
	leaq (%rsi,%rcx,8),%r11   /* im0 pointer */
	leaq (%r10,%rax,8),%r12   /* re1 pointer */
	leaq (%r11,%rax,8),%r13   /* im1 pointer */
	movq %rdx,%r14            /* tcs pointer */
	movq $0,%r13              /* r13: off (reuse) */
	leaq (%r10,%rax,8),%r12   /* re1 pointer */
	leaq (%r11,%rax,8),%r13   /* im1 pointer - wait, r13 reuse conflict */

	/* Redo properly - r13 was clobbered, use different approach */
	leaq (%rdi,%rcx,8),%r10
	leaq (%rsi,%rcx,8),%r11
	leaq (%r10,%rax,8),%r12
	leaq (%r11,%rax,8),%r13
	movq %rdx,%r14
	pushq %rcx                /* save block counter */
	movq $0,%rcx
fftoffloop:
	vmovapd (%r10),%ymm0     /* re0 */
	vmovapd (%r11),%ymm1     /* im0 */
	vmovapd (%r12),%ymm2     /* re1 */
	vmovapd (%r13),%ymm3     /* im1 */
	vmovapd (%r14),%ymm4     /* cos */
	vmovapd 32(%r14),%ymm5   /* sin */
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
	popq    %rcx              /* restore block counter */
	leaq	(%rcx,%rax,2),%rcx
	cmpq	%r9,%rcx
	jb 	fftblockloop
	/* end of halfnn loop */
	shlq	$1,%rax
	leaq	(%rdx,%rax,8),%rdx
	cmpq	%r9,%rax
	jb ffthalfnnloop

fftbeforefinal:
	/* cur_tt is at rdx */
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
# Instead of doing the last halfnn butterfly, storing, reloading for twist,
# we compute both in-register, saving one full pass through the data.
fft_last_halfnn_fused:
	# rax = halfnn (the last value), rdx = butterfly trig pointer
	# After butterfly trig, the twist trig starts at rdx + halfnn*16 bytes
	leaq (%rdx,%rax,8),%r10
	leaq (%r10,%rax,8),%r10   /* r10 = twist trig base */
	# For the last halfnn: 1 block, halfnn/4 inner iterations
	# re0 = rdi, im0 = rsi, re1 = rdi+halfnn*8, im1 = rsi+halfnn*8
	movq %rdi,%r11            /* r11 = re0 */
	movq %rsi,%r12            /* r12 = im0 */
	leaq (%rdi,%rax,8),%r13   /* r13 = re1 = rdi + halfnn*8 */
	leaq (%rsi,%rax,8),%r14   /* r14 = im1 = rsi + halfnn*8 */
	movq $0,%rcx
.p2align 4
fft_last_fused_loop:
	# Butterfly
	vmovapd (%r11),%ymm0     /* re0 */
	vmovapd (%r12),%ymm1     /* im0 */
	vmovapd (%r13),%ymm2     /* re1 */
	vmovapd (%r14),%ymm3     /* im1 */
	vmovapd (%rdx),%ymm4     /* bf cos */
	vmovapd 32(%rdx),%ymm5   /* bf sin */
	vmulpd	%ymm2,%ymm4,%ymm6
	vmulpd	%ymm2,%ymm5,%ymm7
	vfnmadd231pd %ymm3,%ymm5,%ymm6   /* re2 */
	vfmadd231pd %ymm3,%ymm4,%ymm7    /* im2 */
	vsubpd	%ymm6,%ymm0,%ymm2         /* new_re1 = re0 - re2 */
	vsubpd	%ymm7,%ymm1,%ymm3         /* new_im1 = im0 - im2 */
	vaddpd	%ymm6,%ymm0,%ymm0         /* new_re0 = re0 + re2 */
	vaddpd	%ymm7,%ymm1,%ymm1         /* new_im0 = im0 + im2 */

	# Twist on output 0 (position = off)
	vmovapd (%r10),%ymm4              /* tw0 cos */
	vmovapd 32(%r10),%ymm5            /* tw0 sin */
	vmulpd %ymm0,%ymm4,%ymm6         /* re0*cos */
	vmulpd %ymm0,%ymm5,%ymm7         /* re0*sin */
	vfnmadd231pd %ymm1,%ymm5,%ymm6   /* re0*cos - im0*sin */
	vfmadd231pd %ymm1,%ymm4,%ymm7    /* re0*sin + im0*cos */
	vmovapd %ymm6,(%r11)
	vmovapd %ymm7,(%r12)

	# Twist on output 1 (position = off + halfnn)
	# Twist trig offset: halfnn*16 bytes from tw0 base
	leaq (%r10,%rax,8),%rbx
	leaq (%rbx,%rax,8),%rbx           /* rbx = tw1 = r10 + halfnn*16 */
	vmovapd (%rbx),%ymm4              /* tw1 cos */
	vmovapd 32(%rbx),%ymm5            /* tw1 sin */
	vmulpd %ymm2,%ymm4,%ymm6
	vmulpd %ymm2,%ymm5,%ymm7
	vfnmadd231pd %ymm3,%ymm5,%ymm6
	vfmadd231pd %ymm3,%ymm4,%ymm7
	vmovapd %ymm6,(%r13)
	vmovapd %ymm7,(%r14)

	# Advance
	leaq 32(%r11),%r11
	leaq 32(%r12),%r12
	leaq 32(%r13),%r13
	leaq 32(%r14),%r14
	leaq 64(%rdx),%rdx        /* butterfly trig */
	leaq 64(%r10),%r10        /* twist trig */
	addq $4,%rcx
	cmpq %rax,%rcx
	jb fft_last_fused_loop
	/* done - skip fftfinalloop */

fftend:
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
fft_negmask_s2:    .double +1.0, -1.0, +1.0, -1.0
fft_negmask_s4im:  .double +1.0, -1.0, -1.0, +1.0
fft_negmask_s4re:  .double +1.0, +1.0, -1.0, -1.0
/* FFT size-8 twiddle: cos/sin(-2pi*k/8) for k=0..3 */
fft_s8cos: .double +1.0, +0.7071067811865476, +0.0, -0.7071067811865476
fft_s8sin: .double +0.0, -0.7071067811865476, -1.0, -0.7071067811865476

#if !__APPLE__
	.size	fft, .-fft
#endif
