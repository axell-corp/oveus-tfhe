import Mathlib

namespace Reciprocal

/-- The paper uses `W = 4k`. -/
def W (k : ℕ) : ℕ := 4 * k

/-- Bit-width: `bw y = ⌊log₂ y⌋ + 1` for `y > 0`. -/
def bw (y : ℕ) : ℕ := y.log2 + 1

lemma bw_pos (y : ℕ) : 0 < bw y := by
  unfold bw
  omega

lemma bw_ge_of_pow_le {y a : ℕ} (hy : y ≠ 0) (h : 2 ^ a ≤ y) : a + 1 ≤ bw y := by
  unfold bw
  have h' : a ≤ y.log2 := (Nat.le_log2 hy).2 h
  omega

lemma bw_le_of_lt_pow {y a : ℕ} (hy : y ≠ 0) (h : y < 2 ^ a) : bw y ≤ a := by
  unfold bw
  have h' : y.log2 < a := (Nat.log2_lt hy).2 h
  omega

lemma bw_bounds {y : ℕ} (hy : y ≠ 0) : 2 ^ (bw y - 1) ≤ y ∧ y < 2 ^ (bw y) := by
  constructor
  · simpa [bw] using Nat.log2_self_le hy
  · simpa [bw] using (Nat.lt_log2_self (n := y))

lemma bw_two_pow (m : ℕ) : bw (2 ^ m) = m + 1 := by
  simp [bw, Nat.log2_two_pow]

/-- The normalized dividend `x = y * 2^(W - ℓ)`. -/
def x (k y : ℕ) : ℕ := y * 2 ^ (W k - bw y)

/-- `h = ⌊x / 2^(2k)⌋`. -/
def h (k y : ℕ) : ℕ := x k y / 2 ^ (2 * k)

/-- `g = ⌊h / 2^k⌋`. -/
def g (k y : ℕ) : ℕ := h k y / 2 ^ k

/-- `z₀ = ⌊2^(2k-1) / g⌋`. -/
def z0 (k y : ℕ) : ℕ := 2 ^ (2 * k - 1) / g k y

lemma x_two_pow {k m : ℕ} (hm : m + 1 ≤ W k) : x k (2 ^ m) = 2 ^ (W k - 1) := by
  unfold x
  rw [bw_two_pow]
  rw [← pow_add]
  congr 1
  omega

lemma h_two_pow {k m : ℕ} (hm : m + 1 ≤ W k) : h k (2 ^ m) = 2 ^ (2 * k - 1) := by
  unfold h
  rw [x_two_pow hm]
  rw [show W k - 1 = 2 * k + (2 * k - 1) by
    unfold W
    omega]
  rw [pow_add, Nat.mul_div_right]
  positivity

lemma g_two_pow {k m : ℕ} (hm : m + 1 ≤ W k) : g k (2 ^ m) = 2 ^ (k - 1) := by
  unfold g
  rw [h_two_pow hm]
  rw [show 2 * k - 1 = k + (k - 1) by omega]
  rw [pow_add, Nat.mul_div_right]
  positivity

lemma z0_two_pow {k m : ℕ} (hm : m + 1 ≤ W k) : z0 k (2 ^ m) = 2 ^ k := by
  unfold z0
  rw [g_two_pow hm]
  rw [show 2 * k - 1 = (k - 1) + k by omega]
  rw [pow_add, Nat.mul_comm]
  exact Nat.mul_div_left (2 ^ k) (by positivity : 0 < 2 ^ (k - 1))

/-- The first refinement step from the paper. -/
def z1 (k y : ℕ) : ℕ :=
  let z0v := z0 k y
  let hv := h k y
  2 ^ k * z0v - (z0v * ((hv * z0v) / 2 ^ k)) / 2 ^ k

/-- The second refinement step from the paper. -/
def z2 (k y : ℕ) : ℕ :=
  let z1v := z1 k y
  let xv := x k y
  2 ^ (2 * k + 3) * z1v - (z1v * ((xv * z1v) / 2 ^ (2 * k - 4))) / 2 ^ (2 * k)

/-- `q₀ = ⌊z₂ / 2^ℓ⌋`. -/
def q0 (k y : ℕ) : ℕ := z2 k y / 2 ^ (bw y)

/-- The clamp to `[0, 2^W - 1]`. -/
def qTilde (k y : ℕ) : ℕ := min (q0 k y) (2 ^ (W k) - 1)

/-- The final routine from the paper. -/
def Fk (k y : ℕ) : ℕ :=
  let q := qTilde k y
  q + if q * y < 2 ^ (W k) - y then 1 else 0

/-- The target correctness predicate. -/
def Good (k y : ℕ) : Prop :=
  2 ^ (W k) - y ≤ y * Fk k y ∧ y * Fk k y ≤ 2 ^ (W k)

lemma z1_two_pow {k m : ℕ} (_hk : 3 ≤ k) (hm : m + 1 ≤ W k) : z1 k (2 ^ m) = 2 ^ (2 * k - 1) := by
  unfold z1
  rw [z0_two_pow hm, h_two_pow hm]
  simp only
  have hdiv1 : 2 ^ (2 * k - 1) * 2 ^ k / 2 ^ k = 2 ^ (2 * k - 1) := by
    exact Nat.mul_div_left (2 ^ (2 * k - 1)) (by positivity : 0 < 2 ^ k)
  rw [hdiv1]
  have hdiv2 : 2 ^ k * 2 ^ (2 * k - 1) / 2 ^ k = 2 ^ (2 * k - 1) := by
    exact Nat.mul_div_right (2 ^ (2 * k - 1)) (by positivity : 0 < 2 ^ k)
  rw [hdiv2]
  have hpow : 2 ^ k * 2 ^ k = 2 ^ (2 * k - 1) * 2 := by
    calc
      2 ^ k * 2 ^ k = 2 ^ (k + k) := by rw [← pow_add]
      _ = 2 ^ (2 * k) := by congr 1; omega
      _ = 2 ^ ((2 * k - 1) + 1) := by congr 1; omega
      _ = 2 ^ (2 * k - 1) * 2 := by rw [pow_add]; norm_num
  rw [hpow]
  rw [Nat.mul_comm _ 2, two_mul]
  exact Nat.add_sub_cancel_left _ _

lemma z2_two_pow {k m : ℕ} (hk : 3 ≤ k) (hm : m + 1 ≤ W k) : z2 k (2 ^ m) = 2 ^ (W k + 1) := by
  unfold z2
  rw [z1_two_pow hk hm, x_two_pow hm]
  simp only
  have hinner : 2 ^ (W k - 1) * 2 ^ (2 * k - 1) / 2 ^ (2 * k - 4) = 2 ^ (W k + 2) := by
    have hexp : (W k - 1) + (2 * k - 1) = (2 * k - 4) + (W k + 2) := by
      unfold W
      omega
    calc
      2 ^ (W k - 1) * 2 ^ (2 * k - 1) / 2 ^ (2 * k - 4)
          = 2 ^ ((W k - 1) + (2 * k - 1)) / 2 ^ (2 * k - 4) := by rw [← pow_add]
      _ = 2 ^ ((2 * k - 4) + (W k + 2)) / 2 ^ (2 * k - 4) := by rw [hexp]
      _ = (2 ^ (2 * k - 4) * 2 ^ (W k + 2)) / 2 ^ (2 * k - 4) := by rw [pow_add]
      _ = 2 ^ (W k + 2) := by
            exact Nat.mul_div_right (2 ^ (W k + 2)) (by positivity : 0 < 2 ^ (2 * k - 4))
  rw [hinner]
  have hdiv2 : 2 ^ (2 * k - 1) * 2 ^ (W k + 2) / 2 ^ (2 * k) = 2 ^ (W k + 1) := by
    have hexp : (2 * k - 1) + (W k + 2) = (W k + 1) + 2 * k := by
      unfold W
      omega
    calc
      2 ^ (2 * k - 1) * 2 ^ (W k + 2) / 2 ^ (2 * k)
          = 2 ^ ((2 * k - 1) + (W k + 2)) / 2 ^ (2 * k) := by rw [← pow_add]
      _ = 2 ^ ((W k + 1) + 2 * k) / 2 ^ (2 * k) := by rw [hexp]
      _ = (2 ^ (W k + 1) * 2 ^ (2 * k)) / 2 ^ (2 * k) := by rw [pow_add]
      _ = 2 ^ (W k + 1) := by
            exact Nat.mul_div_left (2 ^ (W k + 1)) (by positivity : 0 < 2 ^ (2 * k))
  rw [hdiv2]
  have hpow : 2 ^ (2 * k + 3) * 2 ^ (2 * k - 1) = 2 ^ (W k + 1) * 2 := by
    calc
      2 ^ (2 * k + 3) * 2 ^ (2 * k - 1) = 2 ^ ((2 * k + 3) + (2 * k - 1)) := by rw [← pow_add]
      _ = 2 ^ (W k + 2) := by unfold W; congr 1; omega
      _ = 2 ^ ((W k + 1) + 1) := by rfl
      _ = 2 ^ (W k + 1) * 2 := by rw [pow_add]; norm_num
  rw [hpow]
  rw [Nat.mul_comm _ 2, two_mul]
  have : 2 ^ (W k + 1) + 2 ^ (W k + 1) - 2 ^ (W k + 1) = 2 ^ (W k + 1) := by
    exact Nat.add_sub_cancel_left _ _
  simp [this]

lemma q0_two_pow {k m : ℕ} (hk : 3 ≤ k) (hm : m + 1 ≤ W k) : q0 k (2 ^ m) = 2 ^ (W k - m) := by
  unfold q0
  rw [bw_two_pow, z2_two_pow hk hm]
  have hexp : W k + 1 = (m + 1) + (W k - m) := by
    omega
  rw [hexp, pow_add, Nat.mul_div_right]
  positivity

lemma qTilde_two_pow {k m : ℕ} (hk : 3 ≤ k) (hm : m + 1 ≤ W k) (hmpos : 0 < m) :
    qTilde k (2 ^ m) = 2 ^ (W k - m) := by
  unfold qTilde
  rw [q0_two_pow hk hm]
  apply min_eq_left
  have hlt : 2 ^ (W k - m) < 2 ^ W k := by
    apply Nat.pow_lt_pow_right
    decide
    omega
  omega

lemma Fk_two_pow {k m : ℕ} (hk : 3 ≤ k) (hm : m + 1 ≤ W k) (hmpos : 0 < m) :
    Fk k (2 ^ m) = 2 ^ (W k - m) := by
  unfold Fk
  rw [qTilde_two_pow hk hm hmpos]
  have hmul : 2 ^ (W k - m) * 2 ^ m = 2 ^ W k := by
    rw [← pow_add]
    congr 1
    omega
  have hfalse : ¬2 ^ (W k - m) * 2 ^ m < 2 ^ W k - 2 ^ m := by
    rw [hmul]
    exact not_lt_of_ge (Nat.le_of_lt (Nat.sub_lt (by positivity) (by positivity)))
  simp [hfalse]

lemma Good_two_pow {k m : ℕ} (hk : 3 ≤ k) (hm : m + 1 ≤ W k) (hmpos : 0 < m) :
    Good k (2 ^ m) := by
  unfold Good
  rw [Fk_two_pow hk hm hmpos]
  have hmul : 2 ^ m * 2 ^ (W k - m) = 2 ^ W k := by
    rw [← pow_add]
    congr 1
    omega
  constructor
  · rw [hmul]
    exact Nat.sub_le _ _
  · rw [hmul]

lemma Fk_one {k : ℕ} (hk : 3 ≤ k) : Fk k 1 = 2 ^ (W k) - 1 := by
  unfold Fk qTilde
  have hm : 0 + 1 ≤ W k := by
    unfold W
    omega
  rw [show (1 : ℕ) = 2 ^ 0 by norm_num, q0_two_pow hk hm]
  simp only [Nat.sub_zero, pow_zero]
  have hmin : min (2 ^ W k) (2 ^ W k - 1) = 2 ^ W k - 1 := by
    exact min_eq_right (Nat.sub_le _ _)
  rw [hmin]
  have hfalse : ¬(2 ^ W k - 1) * 1 < 2 ^ W k - 1 := by
    simp
  simp

lemma Good_one {k : ℕ} (hk : 3 ≤ k) : Good k 1 := by
  unfold Good
  rw [Fk_one hk]
  constructor <;> simp

lemma Good_power_of_two {k m : ℕ} (hk : 3 ≤ k) (hm : m + 1 ≤ W k) : Good k (2 ^ m) := by
  by_cases hm0 : m = 0
  · subst hm0
    simpa using Good_one hk
  · exact Good_two_pow hk hm (Nat.pos_of_ne_zero hm0)

/-- TeX correspondence: start of the low-bit regime (`prop:exact-low`). -/
lemma x_low {k y : ℕ} (hyk : bw y ≤ k) : x k y = y * 2 ^ (2 * k + (2 * k - bw y)) := by
  unfold x W
  congr 2
  omega

lemma h_low {k y : ℕ} (hyk : bw y ≤ k) : h k y = y * 2 ^ (2 * k - bw y) := by
  unfold h
  rw [x_low hyk]
  rw [pow_add]
  calc
    y * (2 ^ (2 * k) * 2 ^ (2 * k - bw y)) / 2 ^ (2 * k)
        = (2 ^ (2 * k) * (y * 2 ^ (2 * k - bw y))) / 2 ^ (2 * k) := by ac_rfl
    _ = y * 2 ^ (2 * k - bw y) := by
      exact Nat.mul_div_right (y * 2 ^ (2 * k - bw y)) (by positivity : 0 < 2 ^ (2 * k))

lemma g_low {k y : ℕ} (hyk : bw y ≤ k) : g k y = y * 2 ^ (k - bw y) := by
  unfold g
  rw [h_low hyk]
  rw [show 2 * k - bw y = k + (k - bw y) by omega]
  rw [pow_add]
  calc
    y * (2 ^ k * 2 ^ (k - bw y)) / 2 ^ k = (2 ^ k * (y * 2 ^ (k - bw y))) / 2 ^ k := by
      ac_rfl
    _ = y * 2 ^ (k - bw y) := by
      exact Nat.mul_div_right (y * 2 ^ (k - bw y)) (by positivity : 0 < 2 ^ k)

lemma bw_three : bw 3 = 2 := by decide

lemma pow_two_even_mod_three (n : ℕ) : 2 ^ (2 * n) % 3 = 1 := by
  induction n with
  | zero => norm_num
  | succ n ih =>
      rw [show 2 * (n + 1) = 2 * n + 2 by omega]
      rw [pow_add, pow_two]
      norm_num [Nat.mul_mod, ih]

lemma pow_two_odd_mod_three (n : ℕ) : 2 ^ (2 * n + 1) % 3 = 2 := by
  rw [show 2 * n + 1 = 2 * n + 1 by omega, pow_add]
  norm_num [pow_two_even_mod_three, Nat.mul_mod]

lemma div_pow_mod_three_even (n : ℕ) :
    2 ^ (2 * n + 1) / 3 = (2 ^ (2 * n + 1) - 2) / 3 := by
  have hdecomp := Nat.mod_add_div (2 ^ (2 * n + 1)) 3
  rw [pow_two_odd_mod_three] at hdecomp
  have hmul1 : 2 ^ (2 * n + 1) / 3 * 3 = 2 ^ (2 * n + 1) - 2 := by omega
  have hdvd : 3 ∣ 2 ^ (2 * n + 1) - 2 := ⟨2 ^ (2 * n + 1) / 3, by
    exact by simpa [Nat.mul_comm] using hmul1.symm⟩
  have hmul2 : 3 * ((2 ^ (2 * n + 1) - 2) / 3) = 2 ^ (2 * n + 1) - 2 := by
    simpa [Nat.mul_comm] using (Nat.div_mul_cancel hdvd)
  exact Nat.eq_of_mul_eq_mul_left (by decide : 0 < 3) (by simpa [Nat.mul_comm] using hmul1.trans hmul2.symm)

lemma div_pow_mod_three_odd (n : ℕ) :
    2 ^ (2 * n) / 3 = (2 ^ (2 * n) - 1) / 3 := by
  have hdecomp := Nat.mod_add_div (2 ^ (2 * n)) 3
  rw [pow_two_even_mod_three] at hdecomp
  have hmul1 : 2 ^ (2 * n) / 3 * 3 = 2 ^ (2 * n) - 1 := by omega
  have hdvd : 3 ∣ 2 ^ (2 * n) - 1 := ⟨2 ^ (2 * n) / 3, by
    exact by simpa [Nat.mul_comm] using hmul1.symm⟩
  have hmul2 : 3 * ((2 ^ (2 * n) - 1) / 3) = 2 ^ (2 * n) - 1 := by
    simpa [Nat.mul_comm] using (Nat.div_mul_cancel hdvd)
  exact Nat.eq_of_mul_eq_mul_left (by decide : 0 < 3) (by simpa [Nat.mul_comm] using hmul1.trans hmul2.symm)

lemma z0_three_even {n : ℕ} (hn : 1 ≤ n) : z0 (2 * n) 3 = (2 ^ (2 * n + 1) - 2) / 3 := by
  have hyk : bw 3 ≤ 2 * n := by
    simp [bw_three]
    omega
  unfold z0
  rw [g_low hyk, bw_three]
  rw [Nat.mul_comm 3 _, ← Nat.div_div_eq_div_mul]
  rw [show 2 * (2 * n) - 1 = 4 * n - 1 by omega]
  have hpow : 2 ^ (4 * n - 1) / 2 ^ (2 * n - 2) = 2 ^ (2 * n + 1) := by
    apply Nat.div_eq_of_eq_mul_right (by positivity : 0 < 2 ^ (2 * n - 2))
    rw [show 4 * n - 1 = (2 * n + 1) + (2 * n - 2) by omega, pow_add, Nat.mul_comm]
  rw [hpow]
  exact div_pow_mod_three_even n

lemma z0_three_even_alt {n : ℕ} (hn : 1 ≤ n) :
    z0 (2 * n) 3 = 2 * ((2 ^ (2 * n) - 1) / 3) := by
  rw [z0_three_even hn]
  have hdiv : 3 ∣ 2 ^ (2 * n) - 1 := by
    have hdecomp := Nat.mod_add_div (2 ^ (2 * n)) 3
    rw [pow_two_even_mod_three] at hdecomp
    have hmul1 : 2 ^ (2 * n) / 3 * 3 = 2 ^ (2 * n) - 1 := by omega
    exact ⟨2 ^ (2 * n) / 3, by simpa [Nat.mul_comm] using hmul1.symm⟩
  rw [show 2 ^ (2 * n + 1) - 2 = 2 * (2 ^ (2 * n) - 1) by
    rw [show 2 * n + 1 = 2 * n + 1 by omega, pow_add]
    norm_num
    omega]
  rw [Nat.mul_div_assoc _ hdiv]

lemma h_three_even {n : ℕ} (hn : 1 ≤ n) : h (2 * n) 3 = 3 * 2 ^ (4 * n - 2) := by
  have hyk : bw 3 ≤ 2 * n := by
    simp [bw_three]
    omega
  rw [show 4 * n - 2 = 2 * (2 * n) - 2 by omega]
  simpa [bw_three] using h_low (k := 2 * n) (y := 3) hyk

lemma u_three_even_mul {n : ℕ} :
    3 * ((2 ^ (2 * n) - 1) / 3) = 2 ^ (2 * n) - 1 := by
  rw [← div_pow_mod_three_odd]
  have hdecomp := Nat.mod_add_div (2 ^ (2 * n)) 3
  rw [pow_two_even_mod_three] at hdecomp
  have hmul1 : 2 ^ (2 * n) / 3 * 3 = 2 ^ (2 * n) - 1 := by omega
  simpa [Nat.mul_comm] using hmul1

lemma inner_three_even {n : ℕ} (hn : 1 ≤ n) :
    (h (2 * n) 3 * z0 (2 * n) 3) / 2 ^ (2 * n) =
      (3 * ((2 ^ (2 * n) - 1) / 3)) * 2 ^ (2 * n - 1) := by
  rw [h_three_even hn, z0_three_even_alt hn]
  have hdivpow : 2 ^ (2 * n) ∣ 2 ^ (4 * n - 1) := by
    refine ⟨2 ^ (2 * n - 1), ?_⟩
    rw [show 4 * n - 1 = 2 * n + (2 * n - 1) by omega, pow_add, Nat.mul_comm]
  have hpow :
      2 ^ (4 * n - 1) / 2 ^ (2 * n) = 2 ^ (2 * n - 1) := by
    apply Nat.div_eq_of_eq_mul_right (by positivity : 0 < 2 ^ (2 * n))
    rw [show 4 * n - 1 = (2 * n - 1) + 2 * n by omega, pow_add, Nat.mul_comm]
  calc
    (3 * 2 ^ (4 * n - 2) * (2 * ((2 ^ (2 * n) - 1) / 3))) / 2 ^ (2 * n)
        = (3 * ((2 ^ (2 * n) - 1) / 3) * 2 ^ (4 * n - 1)) / 2 ^ (2 * n) := by
            rw [show 2 ^ (4 * n - 1) = 2 * 2 ^ (4 * n - 2) by
              rw [show 4 * n - 1 = (4 * n - 2) + 1 by omega, pow_add]
              norm_num
              rw [Nat.mul_comm]]
            ac_rfl
    _ = (3 * ((2 ^ (2 * n) - 1) / 3)) * (2 ^ (4 * n - 1) / 2 ^ (2 * n)) := by
          rw [Nat.mul_div_assoc _ hdivpow]
    _ = (3 * ((2 ^ (2 * n) - 1) / 3)) * 2 ^ (2 * n - 1) := by rw [hpow]

lemma second_three_even {n : ℕ} (hn : 1 ≤ n) :
    (z0 (2 * n) 3 * ((h (2 * n) 3 * z0 (2 * n) 3) / 2 ^ (2 * n))) / 2 ^ (2 * n) =
      ((2 ^ (2 * n) - 1) / 3) * (3 * ((2 ^ (2 * n) - 1) / 3)) := by
  nth_rewrite 1 [z0_three_even_alt hn]
  rw [inner_three_even hn]
  have hpow : 2 * 2 ^ (2 * n - 1) = 2 ^ (2 * n) := by
    rw [show 2 * n = (2 * n - 1) + 1 by omega, pow_add]
    norm_num
    rw [Nat.mul_comm]
  calc
    (2 * ((2 ^ (2 * n) - 1) / 3) * ((3 * ((2 ^ (2 * n) - 1) / 3)) * 2 ^ (2 * n - 1))) /
        2 ^ (2 * n)
        = ((((2 ^ (2 * n) - 1) / 3) * (3 * ((2 ^ (2 * n) - 1) / 3))) *
            (2 * 2 ^ (2 * n - 1))) / 2 ^ (2 * n) := by
              ac_rfl
    _ = ((((2 ^ (2 * n) - 1) / 3) * (3 * ((2 ^ (2 * n) - 1) / 3))) * 2 ^ (2 * n)) /
            2 ^ (2 * n) := by rw [hpow]
    _ = ((2 ^ (2 * n) - 1) / 3) * (3 * ((2 ^ (2 * n) - 1) / 3)) := by
          rw [Nat.mul_comm]
          exact Nat.mul_div_right
            (((2 ^ (2 * n) - 1) / 3) * (3 * ((2 ^ (2 * n) - 1) / 3)))
            (by positivity : 0 < 2 ^ (2 * n))

lemma inner_three_even_explicit {n : ℕ} (hn : 1 ≤ n) :
    (3 * 2 ^ (4 * n - 2) * (2 * ((2 ^ (2 * n) - 1) / 3))) / 2 ^ (2 * n) =
      (3 * ((2 ^ (2 * n) - 1) / 3)) * 2 ^ (2 * n - 1) := by
  simpa [h_three_even hn, z0_three_even_alt hn] using inner_three_even hn

lemma second_three_even_reduced {n : ℕ} (hn : 1 ≤ n) :
    (2 * ((2 ^ (2 * n) - 1) / 3) * ((3 * ((2 ^ (2 * n) - 1) / 3)) * 2 ^ (2 * n - 1))) /
        2 ^ (2 * n) =
      ((2 ^ (2 * n) - 1) / 3) * (3 * ((2 ^ (2 * n) - 1) / 3)) := by
  have hpow : 2 * 2 ^ (2 * n - 1) = 2 ^ (2 * n) := by
    rw [show 2 * n = (2 * n - 1) + 1 by omega, pow_add]
    norm_num
    rw [Nat.mul_comm]
  calc
    (2 * ((2 ^ (2 * n) - 1) / 3) * ((3 * ((2 ^ (2 * n) - 1) / 3)) * 2 ^ (2 * n - 1))) /
        2 ^ (2 * n)
        = ((((2 ^ (2 * n) - 1) / 3) * (3 * ((2 ^ (2 * n) - 1) / 3))) *
            (2 * 2 ^ (2 * n - 1))) / 2 ^ (2 * n) := by
              ac_rfl
    _ = ((((2 ^ (2 * n) - 1) / 3) * (3 * ((2 ^ (2 * n) - 1) / 3))) * 2 ^ (2 * n)) /
            2 ^ (2 * n) := by rw [hpow]
    _ = ((2 ^ (2 * n) - 1) / 3) * (3 * ((2 ^ (2 * n) - 1) / 3)) := by
          rw [Nat.mul_comm]
          exact Nat.mul_div_right
            (((2 ^ (2 * n) - 1) / 3) * (3 * ((2 ^ (2 * n) - 1) / 3)))
            (by positivity : 0 < 2 ^ (2 * n))

lemma z1_three_even {n : ℕ} (hn : 1 ≤ n) :
    z1 (2 * n) 3 = (2 ^ (4 * n) - 1) / 3 := by
  unfold z1
  simp_rw [z0_three_even_alt hn, h_three_even hn]
  rw [inner_three_even_explicit hn, second_three_even_reduced hn]
  set u : ℕ := (2 ^ (2 * n) - 1) / 3
  have hu : 3 * u = 2 ^ (2 * n) - 1 := by
    simp [u, u_three_even_mul]
  have hmain : 2 ^ (2 * n) * (2 * u) - u * (3 * u) = u * (2 ^ (2 * n) + 1) := by
    have ha : 1 ≤ 2 ^ (2 * n) := Nat.succ_le_of_lt (by positivity)
    rw [hu]
    rw [show 2 ^ (2 * n) * (2 * u) = u * (2 * 2 ^ (2 * n)) by ac_rfl]
    rw [← Nat.mul_sub_left_distrib]
    have hcalc : 2 * 2 ^ (2 * n) - (2 ^ (2 * n) - 1) = 2 ^ (2 * n) + 1 := by omega
    rw [hcalc]
  rw [hmain]
  have hsquare : 2 ^ (4 * n) = 2 ^ (2 * n) * 2 ^ (2 * n) := by
    rw [← pow_add]
    congr 1
    omega
  have ha : 1 ≤ 2 ^ (2 * n) := Nat.succ_le_of_lt (by positivity)
  have hsq : (2 ^ (2 * n) - 1) * (2 ^ (2 * n) + 1) = 2 ^ (4 * n) - 1 := by
    have hsq_add : (2 ^ (2 * n) - 1) * (2 ^ (2 * n) + 1) + 1 = 2 ^ (4 * n) := by
      have hbig : 2 ^ (2 * n) + 1 ≤ 2 ^ (2 * n) * (2 ^ (2 * n) + 1) := by
        nlinarith [ha]
      rw [hsquare]
      calc
        (2 ^ (2 * n) - 1) * (2 ^ (2 * n) + 1) + 1
            = (2 ^ (2 * n) * (2 ^ (2 * n) + 1) - (2 ^ (2 * n) + 1)) + 1 := by
                rw [Nat.sub_mul]
                simp
        _ = 2 ^ (2 * n) * (2 ^ (2 * n) + 1) - 2 ^ (2 * n) := by omega
        _ = 2 ^ (2 * n) * 2 ^ (2 * n) := by
              rw [Nat.mul_add, Nat.mul_one, Nat.add_sub_cancel]
    omega
  have hdiv : (2 ^ (4 * n) - 1) / 3 = u * (2 ^ (2 * n) + 1) := by
    apply Nat.div_eq_of_eq_mul_right (by decide : 0 < 3)
    calc
      2 ^ (4 * n) - 1 = (2 ^ (2 * n) - 1) * (2 ^ (2 * n) + 1) := by symm; exact hsq
      _ = (3 * u) * (2 ^ (2 * n) + 1) := by rw [hu]
      _ = 3 * (u * (2 ^ (2 * n) + 1)) := by ac_rfl
  exact hdiv.symm

lemma z0_three_odd {n : ℕ} (hn : 1 ≤ n) :
    z0 (2 * n + 1) 3 = (2 ^ (2 * n + 2) - 1) / 3 := by
  have hyk : bw 3 ≤ 2 * n + 1 := by
    simp [bw_three]
    omega
  unfold z0
  rw [g_low hyk, bw_three]
  rw [Nat.mul_comm 3 _, ← Nat.div_div_eq_div_mul]
  have hpow : 2 ^ (2 * (2 * n + 1) - 1) / 2 ^ ((2 * n + 1) - 2) = 2 ^ (2 * n + 2) := by
    apply Nat.div_eq_of_eq_mul_right (by positivity : 0 < 2 ^ (2 * n - 1))
    rw [show 2 * (2 * n + 1) - 1 = (2 * n + 2) + (2 * n - 1) by omega, pow_add, Nat.mul_comm]
  rw [hpow]
  have hdecomp := Nat.mod_add_div (2 ^ (2 * n + 2)) 3
  have hmod : 2 ^ (2 * n + 2) % 3 = 1 := by
    simpa [show 2 * n + 2 = 2 * (n + 1) by ring] using pow_two_even_mod_three (n + 1)
  rw [hmod] at hdecomp
  have hmul1 : 2 ^ (2 * n + 2) / 3 * 3 = 2 ^ (2 * n + 2) - 1 := by omega
  have hdvd : 3 ∣ 2 ^ (2 * n + 2) - 1 := ⟨2 ^ (2 * n + 2) / 3, by
    simpa [Nat.mul_comm] using hmul1.symm⟩
  have hmul2 : 3 * ((2 ^ (2 * n + 2) - 1) / 3) = 2 ^ (2 * n + 2) - 1 := by
    simpa [Nat.mul_comm] using (Nat.div_mul_cancel hdvd)
  exact Nat.eq_of_mul_eq_mul_left (by decide : 0 < 3) (by
    simpa [Nat.mul_comm] using hmul1.trans hmul2.symm)

lemma z0_three_odd_alt {n : ℕ} (hn : 1 ≤ n) :
    z0 (2 * n + 1) 3 = 4 * ((2 ^ (2 * n) - 1) / 3) + 1 := by
  rw [z0_three_odd hn]
  set u : ℕ := (2 ^ (2 * n) - 1) / 3
  have hu : 3 * u = 2 ^ (2 * n) - 1 := by
    simp [u, u_three_even_mul]
  apply Nat.div_eq_of_eq_mul_right (by decide : 0 < 3)
  have hpow4 : 2 ^ (2 * n + 2) = 4 * 2 ^ (2 * n) := by
    rw [show 2 * n + 2 = 2 * n + 2 by omega, pow_add]
    norm_num
    rw [Nat.mul_comm]
  have hu' : 2 ^ (2 * n) = 3 * u + 1 := by
    have ha : 1 ≤ 2 ^ (2 * n) := Nat.succ_le_of_lt (by positivity)
    have htmp : 3 * u + 1 = 2 ^ (2 * n) := by
      calc
      3 * u + 1 = (2 ^ (2 * n) - 1) + 1 := by rw [hu]
      _ = 2 ^ (2 * n) := by omega
    exact htmp.symm
  rw [hpow4, hu']
  calc
    4 * (3 * u + 1) - 1 = 12 * u + 3 := by omega
    _ = 3 * (4 * u + 1) := by ring

lemma h_three_odd {n : ℕ} (hn : 1 ≤ n) : h (2 * n + 1) 3 = 3 * 2 ^ (4 * n) := by
  have hyk : bw 3 ≤ 2 * n + 1 := by
    simp [bw_three]
    omega
  have := h_low (k := 2 * n + 1) (y := 3) hyk
  rw [show 4 * n = 2 * (2 * n + 1) - 2 by omega]
  simpa [bw_three] using this

lemma inner_three_odd {n : ℕ} (hn : 1 ≤ n) :
    (h (2 * n + 1) 3 * z0 (2 * n + 1) 3) / 2 ^ (2 * n + 1) =
      (4 * ((2 ^ (2 * n) - 1) / 3) + 1) * (3 * 2 ^ (2 * n - 1)) := by
  rw [h_three_odd hn, z0_three_odd_alt hn]
  calc
    (3 * 2 ^ (4 * n) * (4 * ((2 ^ (2 * n) - 1) / 3) + 1)) / 2 ^ (2 * n + 1)
        = ((4 * ((2 ^ (2 * n) - 1) / 3) + 1) * (3 * 2 ^ (4 * n))) / 2 ^ (2 * n + 1) := by
            ac_rfl
    _ = (4 * ((2 ^ (2 * n) - 1) / 3) + 1) * ((3 * 2 ^ (4 * n)) / 2 ^ (2 * n + 1)) := by
          rw [Nat.mul_div_assoc]
          refine ⟨3 * 2 ^ (2 * n - 1), ?_⟩
          rw [show 4 * n = (2 * n + 1) + (2 * n - 1) by omega, pow_add]
          ac_rfl
    _ = (4 * ((2 ^ (2 * n) - 1) / 3) + 1) * (3 * 2 ^ (2 * n - 1)) := by
          apply congrArg ((4 * ((2 ^ (2 * n) - 1) / 3) + 1) * ·)
          apply Nat.div_eq_of_eq_mul_right (by positivity : 0 < 2 ^ (2 * n + 1))
          rw [show 4 * n = (2 * n - 1) + (2 * n + 1) by omega, pow_add]
          ac_rfl

lemma second_three_odd_reduced {n : ℕ} (hn : 1 ≤ n) :
    ((4 * ((2 ^ (2 * n) - 1) / 3) + 1) *
        ((4 * ((2 ^ (2 * n) - 1) / 3) + 1) * (3 * 2 ^ (2 * n - 1)))) / 2 ^ (2 * n + 1) =
      12 * ((2 ^ (2 * n) - 1) / 3) ^ 2 + 6 * ((2 ^ (2 * n) - 1) / 3) := by
  set u : ℕ := (2 ^ (2 * n) - 1) / 3
  have hpow : 2 ^ (2 * n + 1) = 4 * 2 ^ (2 * n - 1) := by
    rw [show 2 * n + 1 = (2 * n - 1) + 2 by omega, pow_add]
    norm_num
    rw [Nat.mul_comm]
  rw [hpow]
  calc
    ((4 * u + 1) * ((4 * u + 1) * (3 * 2 ^ (2 * n - 1)))) / (4 * 2 ^ (2 * n - 1))
        = ((4 * u + 1) * (3 * (4 * u + 1)) * 2 ^ (2 * n - 1)) / (4 * 2 ^ (2 * n - 1)) := by
            ac_rfl
    _ = ((4 * u + 1) * (3 * (4 * u + 1))) / 4 := by
          rw [show 4 * 2 ^ (2 * n - 1) = 2 ^ (2 * n - 1) * 4 by ac_rfl, ← Nat.div_div_eq_div_mul]
          rw [show (4 * u + 1) * (3 * (4 * u + 1)) * 2 ^ (2 * n - 1) =
              2 ^ (2 * n - 1) * ((4 * u + 1) * (3 * (4 * u + 1))) by ac_rfl]
          rw [Nat.mul_div_right
            ((4 * u + 1) * (3 * (4 * u + 1)))
            (by positivity : 0 < 2 ^ (2 * n - 1))]
    _ = 12 * u ^ 2 + 6 * u := by
          rw [show (4 * u + 1) * (3 * (4 * u + 1)) = 3 + (12 * u ^ 2 + 6 * u) * 4 by ring]
          rw [Nat.add_mul_div_right 3 (12 * u ^ 2 + 6 * u) (by decide)]
          rw [Nat.div_eq_of_lt (by decide)]
          simp

lemma inner_three_odd_explicit {n : ℕ} (hn : 1 ≤ n) :
    (3 * 2 ^ (4 * n) * (4 * ((2 ^ (2 * n) - 1) / 3) + 1)) / 2 ^ (2 * n + 1) =
      (4 * ((2 ^ (2 * n) - 1) / 3) + 1) * (3 * 2 ^ (2 * n - 1)) := by
  simpa [h_three_odd hn, z0_three_odd_alt hn] using inner_three_odd hn

lemma second_three_odd_explicit {n : ℕ} (hn : 1 ≤ n) :
    (4 * ((2 ^ (2 * n) - 1) / 3) + 1) *
        (3 * 2 ^ (4 * n) * (4 * ((2 ^ (2 * n) - 1) / 3) + 1) / 2 ^ (2 * n + 1)) /
        2 ^ (2 * n + 1) =
      12 * ((2 ^ (2 * n) - 1) / 3) ^ 2 + 6 * ((2 ^ (2 * n) - 1) / 3) := by
  rw [inner_three_odd_explicit hn]
  simpa using second_three_odd_reduced hn

lemma z1_three_odd {n : ℕ} (hn : 1 ≤ n) :
    z1 (2 * n + 1) 3 = (2 ^ (4 * n + 2) + 2) / 3 := by
  unfold z1
  simp_rw [z0_three_odd_alt hn, h_three_odd hn]
  rw [second_three_odd_explicit hn]
  set u : ℕ := (2 ^ (2 * n) - 1) / 3
  have hu : 3 * u = 2 ^ (2 * n) - 1 := by
    simp [u, u_three_even_mul]
  have hmain :
      2 ^ (2 * n + 1) * (4 * u + 1) - (12 * u ^ 2 + 6 * u) = 12 * u ^ 2 + 8 * u + 2 := by
    have hpowu : 2 ^ (2 * n + 1) = 2 * (3 * u + 1) := by
      have hu' : 2 ^ (2 * n) = 3 * u + 1 := by
        have ha : 1 ≤ 2 ^ (2 * n) := Nat.succ_le_of_lt (by positivity)
        have htmp : 3 * u + 1 = 2 ^ (2 * n) := by
          calc
          3 * u + 1 = (2 ^ (2 * n) - 1) + 1 := by rw [hu]
          _ = 2 ^ (2 * n) := by omega
        exact htmp.symm
      rw [show 2 * n + 1 = 2 * n + 1 by omega, pow_add]
      norm_num
      rw [hu']
      rw [Nat.mul_comm]
    rw [hpowu]
    have hexpand : 2 * (3 * u + 1) * (4 * u + 1) = 24 * u ^ 2 + 14 * u + 2 := by ring
    rw [hexpand]
    omega
  rw [hmain]
  have hdiv : (2 ^ (4 * n + 2) + 2) / 3 = 12 * u ^ 2 + 8 * u + 2 := by
    apply Nat.div_eq_of_eq_mul_right (by decide : 0 < 3)
    have hmul : 2 ^ (4 * n + 2) + 2 = 3 * (12 * u ^ 2 + 8 * u + 2) := by
      have hu' : 2 ^ (2 * n) = 3 * u + 1 := by
        have ha : 1 ≤ 2 ^ (2 * n) := Nat.succ_le_of_lt (by positivity)
        have htmp : 3 * u + 1 = 2 ^ (2 * n) := by
          calc
          3 * u + 1 = (2 ^ (2 * n) - 1) + 1 := by rw [hu]
          _ = 2 ^ (2 * n) := by omega
        exact htmp.symm
      rw [show 4 * n + 2 = 2 + 2 * (2 * n) by ring, pow_add]
      norm_num
      have hsq : 2 ^ (2 * (2 * n)) = (2 ^ (2 * n)) ^ 2 := by
        rw [← pow_mul]
        congr 1
        ring
      rw [hsq]
      rw [hu']
      ring
    exact hmul
  exact hdiv.symm

lemma x_three_even {n : ℕ} (hn : 1 ≤ n) : x (2 * n) 3 = 3 * 2 ^ (8 * n - 2) := by
  have hyk : bw 3 ≤ 2 * n := by
    simp [bw_three]
    omega
  rw [show 8 * n - 2 = 2 * (2 * n) + (2 * (2 * n) - 2) by omega]
  simpa [bw_three] using x_low (k := 2 * n) (y := 3) hyk

lemma x_three_odd {n : ℕ} (hn : 1 ≤ n) : x (2 * n + 1) 3 = 3 * 2 ^ (8 * n + 2) := by
  have hyk : bw 3 ≤ 2 * n + 1 := by
    simp [bw_three]
    omega
  rw [show 8 * n + 2 = 2 * (2 * n + 1) + (2 * (2 * n + 1) - 2) by omega]
  simpa [bw_three] using x_low (k := 2 * n + 1) (y := 3) hyk

lemma u_three_four_mul {n : ℕ} :
    3 * ((2 ^ (4 * n) - 1) / 3) = 2 ^ (4 * n) - 1 := by
  have hdecomp := Nat.mod_add_div (2 ^ (4 * n)) 3
  have hmod : 2 ^ (4 * n) % 3 = 1 := by
    simpa [show 4 * n = 2 * (2 * n) by ring] using pow_two_even_mod_three (2 * n)
  rw [hmod] at hdecomp
  have hmul1 : 2 ^ (4 * n) / 3 * 3 = 2 ^ (4 * n) - 1 := by omega
  have hdvd : 3 ∣ 2 ^ (4 * n) - 1 := ⟨2 ^ (4 * n) / 3, by
    simpa [Nat.mul_comm] using hmul1.symm⟩
  simpa [Nat.mul_comm] using (Nat.div_mul_cancel hdvd)

lemma u_three_odd4_mul {n : ℕ} :
    3 * ((2 ^ (4 * n + 1) + 1) / 3) = 2 ^ (4 * n + 1) + 1 := by
  have hmod : 2 ^ (4 * n + 1) % 3 = 2 := by
    simpa [show 4 * n + 1 = 2 * (2 * n) + 1 by ring] using pow_two_odd_mod_three (2 * n)
  have hdvd : 3 ∣ 2 ^ (4 * n + 1) + 1 := by
    refine ⟨2 ^ (4 * n + 1) / 3 + 1, ?_⟩
    have hdecomp := Nat.mod_add_div (2 ^ (4 * n + 1)) 3
    rw [hmod] at hdecomp
    omega
  simpa [Nat.mul_comm] using (Nat.div_mul_cancel hdvd)

lemma pred_mul_succ {a : ℕ} (ha : 1 ≤ a) : (a - 1) * (a + 1) = a * a - 1 := by
  have ha0 : 0 < a := by omega
  rw [Nat.mul_comm, Nat.mul_sub_one]
  have hle : a + 1 ≤ (a + 1) * a := by
    simpa [Nat.mul_comm] using Nat.le_mul_of_pos_left (a + 1) ha0
  have haa : 1 ≤ a * a := Nat.succ_le_of_lt (Nat.mul_pos ha0 ha0)
  apply (Nat.sub_eq_iff_eq_add hle).2
  calc
    (a + 1) * a = a * a + a := by ring
    _ = ((a * a - 1) + 1) + a := by rw [Nat.sub_add_cancel haa]
    _ = (a * a - 1) + (a + 1) := by ac_rfl

lemma z1_three_odd_alt {n : ℕ} (hn : 1 ≤ n) :
    z1 (2 * n + 1) 3 = 2 * ((2 ^ (4 * n + 1) + 1) / 3) := by
  rw [z1_three_odd hn]
  apply Nat.div_eq_of_eq_mul_right (by decide : 0 < 3)
  calc
    2 ^ (4 * n + 2) + 2 = 2 * (2 ^ (4 * n + 1) + 1) := by
      rw [show 4 * n + 2 = (4 * n + 1) + 1 by omega, pow_add]
      norm_num
      ring
    _ = 3 * (2 * ((2 ^ (4 * n + 1) + 1) / 3)) := by
      rw [show 3 * (2 * ((2 ^ (4 * n + 1) + 1) / 3)) =
          2 * (3 * ((2 ^ (4 * n + 1) + 1) / 3)) by ring]
      rw [u_three_odd4_mul]

lemma z2_three_even {n : ℕ} (hn : 1 ≤ n) :
    z2 (2 * n) 3 = 4 * ((2 ^ (8 * n) - 1) / 3) := by
  unfold z2
  simp_rw [z1_three_even hn, x_three_even hn]
  set v : ℕ := (2 ^ (4 * n) - 1) / 3
  have hv : 3 * v = 2 ^ (4 * n) - 1 := by
    simp [v, u_three_four_mul]
  rw [show 2 * (2 * n) + 3 = 4 * n + 3 by omega]
  rw [show 2 * (2 * n) - 4 = 4 * n - 4 by omega]
  rw [show 2 * (2 * n) = 4 * n by omega]
  have hinner : 3 * 2 ^ (8 * n - 2) * v / 2 ^ (4 * n - 4) = 3 * 2 ^ (4 * n + 2) * v := by
    apply Nat.div_eq_of_eq_mul_right (by positivity : 0 < 2 ^ (4 * n - 4))
    rw [show 8 * n - 2 = (4 * n + 2) + (4 * n - 4) by omega, pow_add]
    ac_rfl
  rw [hinner]
  have hcorr_raw : v * (3 * 2 ^ (4 * n + 2) * v) / 2 ^ (4 * n) = 12 * v ^ 2 := by
    calc
      v * (3 * 2 ^ (4 * n + 2) * v) / 2 ^ (4 * n)
          = (12 * v ^ 2 * 2 ^ (4 * n)) / 2 ^ (4 * n) := by
              rw [show 3 * 2 ^ (4 * n + 2) = 12 * 2 ^ (4 * n) by
                rw [show 4 * n + 2 = 2 + 4 * n by omega, pow_add]
                norm_num
                ring]
              ring_nf
      _ = 12 * v ^ 2 := by
            rw [Nat.mul_comm]
            exact Nat.mul_div_right (12 * v ^ 2) (by positivity : 0 < 2 ^ (4 * n))
  rw [hcorr_raw]
  have hcorr : 12 * v ^ 2 = 4 * v * (2 ^ (4 * n) - 1) := by
    calc
      12 * v ^ 2 = 4 * v * (3 * v) := by ring
      _ = 4 * v * (2 ^ (4 * n) - 1) := by rw [hv]
  have hpow : 2 ^ (4 * n + 3) * v = 8 * (2 ^ (4 * n) * v) := by
    rw [show 4 * n + 3 = 4 * n + 3 by omega, pow_add]
    norm_num
    ring
  have hsquare : 2 ^ (8 * n) = 2 ^ (4 * n) * 2 ^ (4 * n) := by
    rw [← pow_add]
    congr 1
    omega
  have hdiv : (2 ^ (8 * n) - 1) / 3 = v * (2 ^ (4 * n) + 1) := by
    apply Nat.div_eq_of_eq_mul_right (by decide : 0 < 3)
    calc
      2 ^ (8 * n) - 1 = (2 ^ (4 * n) - 1) * (2 ^ (4 * n) + 1) := by
        rw [hsquare]
        symm
        exact pred_mul_succ (a := 2 ^ (4 * n)) (Nat.succ_le_of_lt (by positivity))
      _ = (3 * v) * (2 ^ (4 * n) + 1) := by rw [hv]
      _ = 3 * (v * (2 ^ (4 * n) + 1)) := by ac_rfl
  have hsum : (2 ^ (4 * n) - 1) + (2 ^ (4 * n) + 1) = 2 * 2 ^ (4 * n) := by
    have ha : 1 ≤ 2 ^ (4 * n) := Nat.succ_le_of_lt (by positivity)
    omega
  have hle : 12 * v ^ 2 ≤ 2 ^ (4 * n + 3) * v := by
    rw [hcorr, hpow]
    calc
      4 * v * (2 ^ (4 * n) - 1) ≤ 4 * v * (2 ^ (4 * n) - 1) + 4 * v * (2 ^ (4 * n) + 1) := by
        exact Nat.le_add_right _ _
      _ = 8 * (2 ^ (4 * n) * v) := by
        rw [show 4 * v * (2 ^ (4 * n) - 1) + 4 * v * (2 ^ (4 * n) + 1) =
            4 * v * ((2 ^ (4 * n) - 1) + (2 ^ (4 * n) + 1)) by ring]
        rw [hsum]
        ring
  rw [hdiv]
  apply (Nat.sub_eq_iff_eq_add hle).2
  rw [hpow, hcorr]
  calc
    8 * (2 ^ (4 * n) * v) = 4 * v * (2 * 2 ^ (4 * n)) := by ring
    _ = 4 * v * ((2 ^ (4 * n) - 1) + (2 ^ (4 * n) + 1)) := by rw [hsum]
    _ = 4 * v * (2 ^ (4 * n) - 1) + 4 * (v * (2 ^ (4 * n) + 1)) := by ring
    _ = 4 * (v * (2 ^ (4 * n) + 1)) + 4 * v * (2 ^ (4 * n) - 1) := by ac_rfl

lemma z2_three_odd {n : ℕ} (hn : 1 ≤ n) :
    z2 (2 * n + 1) 3 = 4 * ((2 ^ (8 * n + 4) - 4) / 3) := by
  unfold z2
  simp_rw [z1_three_odd_alt hn, x_three_odd hn]
  set v : ℕ := (2 ^ (4 * n + 1) + 1) / 3
  have hv : 3 * v = 2 ^ (4 * n + 1) + 1 := by
    simp [v, u_three_odd4_mul]
  rw [show 2 * (2 * n + 1) + 3 = 4 * n + 5 by omega]
  rw [show 2 * (2 * n + 1) - 4 = 4 * n - 2 by omega]
  rw [show 2 * (2 * n + 1) = 4 * n + 2 by omega]
  have hinner : 3 * 2 ^ (8 * n + 2) * (2 * v) / 2 ^ (4 * n - 2) = 3 * 2 ^ (4 * n + 5) * v := by
    apply Nat.div_eq_of_eq_mul_right (by positivity : 0 < 2 ^ (4 * n - 2))
    rw [show 8 * n + 2 = (4 * n + 4) + (4 * n - 2) by omega, pow_add]
    rw [show 3 * (2 ^ (4 * n + 4) * 2 ^ (4 * n - 2)) * (2 * v) =
        2 ^ (4 * n - 2) * (3 * 2 ^ (4 * n + 5) * v) by
      rw [show 2 ^ (4 * n + 5) = 2 ^ (4 * n + 4) * 2 by
        rw [show 4 * n + 5 = (4 * n + 4) + 1 by omega, pow_add]
        norm_num]
      ac_rfl]
  rw [hinner]
  have hcorr_raw : (2 * v) * (3 * 2 ^ (4 * n + 5) * v) / 2 ^ (4 * n + 2) = 48 * v ^ 2 := by
    calc
      (2 * v) * (3 * 2 ^ (4 * n + 5) * v) / 2 ^ (4 * n + 2)
          = (48 * v ^ 2 * 2 ^ (4 * n + 2)) / 2 ^ (4 * n + 2) := by
              rw [show 3 * 2 ^ (4 * n + 5) = 24 * 2 ^ (4 * n + 2) by
                rw [show 4 * n + 5 = 3 + (4 * n + 2) by omega, pow_add]
                norm_num
                ring]
              ring_nf
      _ = 48 * v ^ 2 := by
            rw [Nat.mul_comm]
            exact Nat.mul_div_right (48 * v ^ 2) (by positivity : 0 < 2 ^ (4 * n + 2))
  rw [hcorr_raw]
  have hcorr : 48 * v ^ 2 = 16 * v * (2 ^ (4 * n + 1) + 1) := by
    calc
      48 * v ^ 2 = 16 * v * (3 * v) := by ring
      _ = 16 * v * (2 ^ (4 * n + 1) + 1) := by rw [hv]
  have hpow : 2 ^ (4 * n + 5) * (2 * v) = 16 * v * (2 * 2 ^ (4 * n + 1)) := by
    rw [show 4 * n + 5 = (4 * n + 1) + 4 by omega, pow_add]
    norm_num
    ring
  have hsquare : 2 ^ (8 * n + 2) = 2 ^ (4 * n + 1) * 2 ^ (4 * n + 1) := by
    rw [← pow_add]
    congr 1
    omega
  have hdiv : (2 ^ (8 * n + 4) - 4) / 3 = 4 * v * (2 ^ (4 * n + 1) - 1) := by
    apply Nat.div_eq_of_eq_mul_right (by decide : 0 < 3)
    calc
      2 ^ (8 * n + 4) - 4 = 4 * (2 ^ (8 * n + 2) - 1) := by
        rw [show 8 * n + 4 = (8 * n + 2) + 2 by omega, pow_add]
        norm_num
        omega
      _ = 4 * ((2 ^ (4 * n + 1) - 1) * (2 ^ (4 * n + 1) + 1)) := by
        rw [hsquare]
        congr 1
        symm
        exact pred_mul_succ (a := 2 ^ (4 * n + 1)) (Nat.succ_le_of_lt (by positivity))
      _ = 4 * ((2 ^ (4 * n + 1) - 1) * (3 * v)) := by rw [hv]
      _ = 3 * (4 * v * (2 ^ (4 * n + 1) - 1)) := by ac_rfl
  have hcomp : 2 ^ (4 * n + 1) + 1 ≤ 2 * 2 ^ (4 * n + 1) := by
    have ha : 1 ≤ 2 ^ (4 * n + 1) := Nat.succ_le_of_lt (by positivity)
    omega
  have hle : 48 * v ^ 2 ≤ 2 ^ (4 * n + 5) * (2 * v) := by
    rw [hcorr, hpow]
    exact Nat.mul_le_mul_left (16 * v) hcomp
  have hsum : (2 ^ (4 * n + 1) + 1) + (2 ^ (4 * n + 1) - 1) = 2 * 2 ^ (4 * n + 1) := by
    have ha : 1 ≤ 2 ^ (4 * n + 1) := Nat.succ_le_of_lt (by positivity)
    omega
  rw [hdiv]
  apply (Nat.sub_eq_iff_eq_add hle).2
  rw [hpow, hcorr]
  calc
    16 * v * (2 * 2 ^ (4 * n + 1))
        = 16 * v * ((2 ^ (4 * n + 1) + 1) + (2 ^ (4 * n + 1) - 1)) := by rw [hsum]
    _ = 16 * v * (2 ^ (4 * n + 1) + 1) + 4 * (4 * v * (2 ^ (4 * n + 1) - 1)) := by ring
    _ = 4 * (4 * v * (2 ^ (4 * n + 1) - 1)) + 16 * v * (2 ^ (4 * n + 1) + 1) := by ac_rfl

lemma q0_three_even {n : ℕ} (hn : 1 ≤ n) :
    q0 (2 * n) 3 = (2 ^ (8 * n) - 1) / 3 := by
  unfold q0
  rw [z2_three_even hn, bw_three]
  rw [show 2 ^ 2 = 4 by norm_num]
  exact Nat.mul_div_right (((2 ^ (8 * n) - 1) / 3)) (by decide : 0 < 4)

lemma q0_three_odd {n : ℕ} (hn : 1 ≤ n) :
    q0 (2 * n + 1) 3 = (2 ^ (8 * n + 4) - 4) / 3 := by
  unfold q0
  rw [z2_three_odd hn, bw_three]
  rw [show 2 ^ 2 = 4 by norm_num]
  exact Nat.mul_div_right (((2 ^ (8 * n + 4) - 4) / 3)) (by decide : 0 < 4)

lemma qTilde_three_even {n : ℕ} (hn : 1 ≤ n) :
    qTilde (2 * n) 3 = (2 ^ (8 * n) - 1) / 3 := by
  unfold qTilde
  rw [q0_three_even hn]
  rw [show W (2 * n) = 8 * n by unfold W; ring]
  apply min_eq_left
  exact Nat.div_le_self _ _

lemma q0_three_odd_mul {n : ℕ} :
    3 * ((2 ^ (8 * n + 4) - 4) / 3) = 2 ^ (8 * n + 4) - 4 := by
  have hmod : 2 ^ (8 * n + 4) % 3 = 1 := by
    simpa [show 8 * n + 4 = 2 * (4 * n + 2) by ring] using pow_two_even_mod_three (4 * n + 2)
  have hdecomp := Nat.mod_add_div (2 ^ (8 * n + 4)) 3
  rw [hmod] at hdecomp
  have hmul1 : 2 ^ (8 * n + 4) / 3 * 3 = 2 ^ (8 * n + 4) - 1 := by omega
  have hpow : 4 ≤ 2 ^ (8 * n + 4) := by
    have : 2 ^ 2 ≤ 2 ^ (8 * n + 4) := by
      apply Nat.pow_le_pow_right
      decide
      omega
    simpa using this
  have hdvd : 3 ∣ 2 ^ (8 * n + 4) - 4 := by
    refine ⟨2 ^ (8 * n + 4) / 3 - 1, ?_⟩
    symm
    rw [Nat.mul_sub_left_distrib, Nat.mul_one]
    rw [show 3 * (2 ^ (8 * n + 4) / 3) = 2 ^ (8 * n + 4) - 1 by
      simpa [Nat.mul_comm] using hmul1]
    omega
  simpa [Nat.mul_comm] using (Nat.div_mul_cancel hdvd)

lemma qTilde_three_odd {n : ℕ} (hn : 1 ≤ n) :
    qTilde (2 * n + 1) 3 = (2 ^ (8 * n + 4) - 4) / 3 := by
  unfold qTilde
  rw [q0_three_odd hn]
  rw [show W (2 * n + 1) = 8 * n + 4 by unfold W; ring]
  apply min_eq_left
  exact le_trans (Nat.div_le_self _ _) (by omega)

lemma Fk_three_even {n : ℕ} (hn : 1 ≤ n) :
    Fk (2 * n) 3 = (2 ^ (8 * n) - 1) / 3 := by
  unfold Fk
  rw [qTilde_three_even hn]
  rw [show W (2 * n) = 8 * n by unfold W; ring]
  have hmul : 3 * ((2 ^ (8 * n) - 1) / 3) = 2 ^ (8 * n) - 1 := by
    simpa [show 8 * n = 4 * (2 * n) by ring] using (u_three_four_mul (n := 2 * n))
  have hfalse : ¬((2 ^ (8 * n) - 1) / 3) * 3 < 2 ^ (8 * n) - 3 := by
    rw [show ((2 ^ (8 * n) - 1) / 3) * 3 = 2 ^ (8 * n) - 1 by
      simpa [Nat.mul_comm] using hmul]
    omega
  simp [hfalse]

lemma Fk_three_odd {n : ℕ} (hn : 1 ≤ n) :
    Fk (2 * n + 1) 3 = (2 ^ (8 * n + 4) - 1) / 3 := by
  unfold Fk
  rw [qTilde_three_odd hn]
  rw [show W (2 * n + 1) = 8 * n + 4 by unfold W; ring]
  have hmul0 : 3 * ((2 ^ (8 * n + 4) - 4) / 3) = 2 ^ (8 * n + 4) - 4 := q0_three_odd_mul
  have htrue : ((2 ^ (8 * n + 4) - 4) / 3) * 3 < 2 ^ (8 * n + 4) - 3 := by
    have hpow : 4 ≤ 2 ^ (8 * n + 4) := by
      have : 2 ^ 2 ≤ 2 ^ (8 * n + 4) := by
        apply Nat.pow_le_pow_right
        decide
        omega
      simpa using this
    rw [show ((2 ^ (8 * n + 4) - 4) / 3) * 3 = 2 ^ (8 * n + 4) - 4 by
      simpa [Nat.mul_comm] using hmul0]
    omega
  simp [htrue]
  apply Nat.eq_of_mul_eq_mul_left (by decide : 0 < 3)
  have hmul1 : 3 * ((2 ^ (8 * n + 4) - 1) / 3) = 2 ^ (8 * n + 4) - 1 := by
    simpa [show 8 * n + 4 = 4 * (2 * n + 1) by ring] using (u_three_four_mul (n := 2 * n + 1))
  calc
    3 * (((2 ^ (8 * n + 4) - 4) / 3) + 1)
        = 3 * ((2 ^ (8 * n + 4) - 4) / 3) + 3 := by ring
    _ = 2 ^ (8 * n + 4) - 1 := by rw [hmul0]; omega
    _ = 3 * ((2 ^ (8 * n + 4) - 1) / 3) := by symm; exact hmul1

lemma Fk_three {k : ℕ} (hk : 3 ≤ k) : Fk k 3 = 2 ^ (W k) / 3 := by
  rcases k.even_or_odd' with ⟨n, rfl | rfl⟩
  · have hn : 1 ≤ n := by omega
    rw [Fk_three_even hn]
    rw [show W (2 * n) = 8 * n by unfold W; ring]
    simpa [show 8 * n = 2 * (4 * n) by ring] using (div_pow_mod_three_odd (4 * n)).symm
  · have hn : 1 ≤ n := by omega
    rw [Fk_three_odd hn]
    rw [show W (2 * n + 1) = 8 * n + 4 by unfold W; ring]
    simpa [show 8 * n + 4 = 2 * (4 * n + 2) by ring] using (div_pow_mod_three_odd (4 * n + 2)).symm

theorem Good_three {k : ℕ} (hk : 3 ≤ k) : Good k 3 := by
  unfold Good
  rw [Fk_three hk]
  have hmul : 3 * (2 ^ (W k) / 3) = 2 ^ (W k) - 1 := by
    have hdiv : 2 ^ (W k) / 3 = (2 ^ (W k) - 1) / 3 := by
      simpa [W, show 4 * k = 2 * (2 * k) by ring] using (div_pow_mod_three_odd (2 * k))
    rw [hdiv]
    simpa [W] using (u_three_four_mul (n := k))
  constructor
  · rw [show 3 * (2 ^ (W k) / 3) = 2 ^ (W k) - 1 by simpa [Nat.mul_comm] using hmul]
    omega
  · rw [show 3 * (2 ^ (W k) / 3) = 2 ^ (W k) - 1 by simpa [Nat.mul_comm] using hmul]
    omega

lemma bw_six : bw 6 = 3 := by decide

lemma x_six {k : ℕ} (hk : 3 ≤ k) : x k 6 = x k 3 := by
  have hk6 : bw 6 ≤ k := by simpa [bw_six] using hk
  have hk3 : bw 3 ≤ k := by
    have : 2 ≤ k := by omega
    simpa [bw_three] using this
  rw [x_low hk6, x_low hk3]
  rw [bw_six, bw_three]
  rw [show 6 * 2 ^ (2 * k + (2 * k - 3)) = 3 * 2 ^ (2 * k + (2 * k - 2)) by
    rw [show 2 * k + (2 * k - 2) = (2 * k + (2 * k - 3)) + 1 by omega, pow_add]
    -- simplified by the surrounding ring arithmetic
    ring]

lemma h_six {k : ℕ} (hk : 3 ≤ k) : h k 6 = h k 3 := by
  unfold h
  rw [x_six hk]

lemma g_six {k : ℕ} (hk : 3 ≤ k) : g k 6 = g k 3 := by
  unfold g
  rw [h_six hk]

lemma z0_six {k : ℕ} (hk : 3 ≤ k) : z0 k 6 = z0 k 3 := by
  unfold z0
  rw [g_six hk]

lemma z1_six {k : ℕ} (hk : 3 ≤ k) : z1 k 6 = z1 k 3 := by
  unfold z1
  simp_rw [z0_six hk, h_six hk]

lemma z2_six {k : ℕ} (hk : 3 ≤ k) : z2 k 6 = z2 k 3 := by
  unfold z2
  simp_rw [z1_six hk, x_six hk]

lemma pow_two_even_mod_six_succ (n : ℕ) : 2 ^ (2 * (n + 1)) % 6 = 4 := by
  induction n with
  | zero =>
      norm_num
  | succ n ih =>
      rw [show 2 * (n + 1 + 1) = 2 * (n + 1) + 2 by ring]
      rw [pow_add, pow_two]
      norm_num [Nat.mul_mod, ih]

lemma div_pow_mod_six_even_succ (n : ℕ) :
    2 ^ (2 * (n + 1)) / 6 = (2 ^ (2 * (n + 1)) - 4) / 6 := by
  have hdecomp := Nat.mod_add_div (2 ^ (2 * (n + 1))) 6
  rw [pow_two_even_mod_six_succ n] at hdecomp
  have hmul1 : 2 ^ (2 * (n + 1)) / 6 * 6 = 2 ^ (2 * (n + 1)) - 4 := by omega
  have hdvd : 6 ∣ 2 ^ (2 * (n + 1)) - 4 := ⟨2 ^ (2 * (n + 1)) / 6, by
    exact by simpa [Nat.mul_comm] using hmul1.symm⟩
  have hmul2 : 6 * ((2 ^ (2 * (n + 1)) - 4) / 6) = 2 ^ (2 * (n + 1)) - 4 := by
    simpa [Nat.mul_comm] using (Nat.div_mul_cancel hdvd)
  exact Nat.eq_of_mul_eq_mul_left (by decide : 0 < 6) (by simpa [Nat.mul_comm] using hmul1.trans hmul2.symm)

lemma u_six_mul_succ (n : ℕ) :
    6 * ((2 ^ (2 * (n + 1)) - 4) / 6) = 2 ^ (2 * (n + 1)) - 4 := by
  have hdecomp := Nat.mod_add_div (2 ^ (2 * (n + 1))) 6
  rw [pow_two_even_mod_six_succ n] at hdecomp
  have hmul1 : 2 ^ (2 * (n + 1)) / 6 * 6 = 2 ^ (2 * (n + 1)) - 4 := by omega
  have hdvd : 6 ∣ 2 ^ (2 * (n + 1)) - 4 := ⟨2 ^ (2 * (n + 1)) / 6, by
    exact by simpa [Nat.mul_comm] using hmul1.symm⟩
  simpa [Nat.mul_comm] using (Nat.div_mul_cancel hdvd)

lemma q0_six_even {n : ℕ} (hn : 2 ≤ n) :
    q0 (2 * n) 6 = (2 ^ (8 * n) - 4) / 6 := by
  unfold q0
  have hk : 3 ≤ 2 * n := by omega
  have hn1 : 1 ≤ n := by omega
  rw [z2_six hk, z2_three_even hn1, bw_six]
  set q : ℕ := (2 ^ (8 * n) - 4) / 6
  have hq6 : 6 * q = 2 ^ (8 * n) - 4 := by
    simpa [q, show 8 * n = 2 * (4 * n - 1 + 1) by omega] using u_six_mul_succ (4 * n - 1)
  have hA : ((2 ^ (8 * n) - 1) / 3) = 2 * q + 1 := by
    apply Nat.eq_of_mul_eq_mul_left (by decide : 0 < 3)
    have hmulA : 3 * ((2 ^ (8 * n) - 1) / 3) = 2 ^ (8 * n) - 1 := by
      simpa [show 8 * n = 4 * (2 * n) by ring] using (u_three_four_mul (n := 2 * n))
    have hpow4 : 4 ≤ 2 ^ (8 * n) := by
      have : 2 ^ 2 ≤ 2 ^ (8 * n) := by
        apply Nat.pow_le_pow_right
        decide
        omega
      simpa using this
    calc
      3 * ((2 ^ (8 * n) - 1) / 3) = 2 ^ (8 * n) - 1 := hmulA
      _ = (2 ^ (8 * n) - 4) + 3 := by omega
      _ = 6 * q + 3 := by rw [← hq6]
      _ = 3 * (2 * q + 1) := by ring
  rw [hA]
  rw [show 4 * (2 * q + 1) = 4 + q * 8 by ring]
  norm_num
  simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using
    (Nat.add_mul_div_right 4 q (by decide : 0 < 8))

lemma q0_six_odd {n : ℕ} (hn : 1 ≤ n) :
    q0 (2 * n + 1) 6 = (2 ^ (8 * n + 4) - 4) / 6 := by
  unfold q0
  have hk : 3 ≤ 2 * n + 1 := by omega
  rw [z2_six hk, z2_three_odd hn, bw_six]
  set q : ℕ := (2 ^ (8 * n + 4) - 4) / 6
  have hq6 : 6 * q = 2 ^ (8 * n + 4) - 4 := by
    simpa [q, show 8 * n + 4 = 2 * (4 * n + 1 + 1) by ring] using u_six_mul_succ (4 * n + 1)
  have hB : ((2 ^ (8 * n + 4) - 4) / 3) = 2 * q := by
    apply Nat.eq_of_mul_eq_mul_left (by decide : 0 < 3)
    have hmulB : 3 * ((2 ^ (8 * n + 4) - 4) / 3) = 2 ^ (8 * n + 4) - 4 := by
      simpa using q0_three_odd_mul (n := n)
    calc
      3 * ((2 ^ (8 * n + 4) - 4) / 3) = 2 ^ (8 * n + 4) - 4 := hmulB
      _ = 6 * q := by rw [hq6]
      _ = 3 * (2 * q) := by ring
  rw [hB]
  rw [show 4 * (2 * q) = 8 * q by ring]
  rw [show 2 ^ 3 = 8 by norm_num]
  exact Nat.mul_div_right q (by decide : 0 < 8)

lemma Fk_six_even {n : ℕ} (hn : 2 ≤ n) :
    Fk (2 * n) 6 = (2 ^ (8 * n) - 4) / 6 := by
  unfold Fk qTilde
  rw [q0_six_even hn]
  have hlt : (2 ^ (8 * n) - 4) / 6 ≤ 2 ^ (W (2 * n)) - 1 := by
    rw [show W (2 * n) = 8 * n by unfold W; ring]
    exact le_trans (Nat.div_le_self _ _) (by omega)
  rw [min_eq_left hlt]
  have hmul : 6 * ((2 ^ (8 * n) - 4) / 6) = 2 ^ (8 * n) - 4 := by
    simpa [show 8 * n = 2 * (4 * n - 1 + 1) by omega] using u_six_mul_succ (4 * n - 1)
  have hfalse : ¬((2 ^ (8 * n) - 4) / 6) * 6 < 2 ^ (8 * n) - 6 := by
    rw [show ((2 ^ (8 * n) - 4) / 6) * 6 = 2 ^ (8 * n) - 4 by simpa [Nat.mul_comm] using hmul]
    omega
  simp [show W (2 * n) = 8 * n by unfold W; ring, hfalse]

lemma Fk_six_odd {n : ℕ} (hn : 1 ≤ n) :
    Fk (2 * n + 1) 6 = (2 ^ (8 * n + 4) - 4) / 6 := by
  unfold Fk qTilde
  rw [q0_six_odd hn]
  have hlt : (2 ^ (8 * n + 4) - 4) / 6 ≤ 2 ^ (W (2 * n + 1)) - 1 := by
    rw [show W (2 * n + 1) = 8 * n + 4 by unfold W; ring]
    exact le_trans (Nat.div_le_self _ _) (by omega)
  rw [min_eq_left hlt]
  have hmul : 6 * ((2 ^ (8 * n + 4) - 4) / 6) = 2 ^ (8 * n + 4) - 4 := by
    simpa [show 8 * n + 4 = 2 * (4 * n + 1 + 1) by ring] using u_six_mul_succ (4 * n + 1)
  have hfalse : ¬((2 ^ (8 * n + 4) - 4) / 6) * 6 < 2 ^ (8 * n + 4) - 6 := by
    rw [show ((2 ^ (8 * n + 4) - 4) / 6) * 6 = 2 ^ (8 * n + 4) - 4 by simpa [Nat.mul_comm] using hmul]
    omega
  simp [show W (2 * n + 1) = 8 * n + 4 by unfold W; ring, hfalse]

theorem Good_six {k : ℕ} (hk : 3 ≤ k) : Good k 6 := by
  unfold Good
  rcases k.even_or_odd' with ⟨n, rfl | rfl⟩
  · have hn : 2 ≤ n := by omega
    rw [Fk_six_even hn]
    rw [show W (2 * n) = 8 * n by unfold W; ring]
    have hmul : 6 * ((2 ^ (8 * n) - 4) / 6) = 2 ^ (8 * n) - 4 := by
      simpa [show 8 * n = 2 * (4 * n - 1 + 1) by omega] using u_six_mul_succ (4 * n - 1)
    constructor
    · rw [show 6 * ((2 ^ (8 * n) - 4) / 6) = 2 ^ (8 * n) - 4 by simpa [Nat.mul_comm] using hmul]
      exact Nat.sub_le_sub_left (by decide : 4 ≤ 6) _
    · rw [show 6 * ((2 ^ (8 * n) - 4) / 6) = 2 ^ (8 * n) - 4 by simpa [Nat.mul_comm] using hmul]
      exact Nat.sub_le _ _
  · have hn : 1 ≤ n := by omega
    rw [Fk_six_odd hn]
    rw [show W (2 * n + 1) = 8 * n + 4 by unfold W; ring]
    have hmul : 6 * ((2 ^ (8 * n + 4) - 4) / 6) = 2 ^ (8 * n + 4) - 4 := by
      simpa [show 8 * n + 4 = 2 * (4 * n + 1 + 1) by ring] using u_six_mul_succ (4 * n + 1)
    constructor
    · rw [show 6 * ((2 ^ (8 * n + 4) - 4) / 6) = 2 ^ (8 * n + 4) - 4 by simpa [Nat.mul_comm] using hmul]
      exact Nat.sub_le_sub_left (by decide : 4 ≤ 6) _
    · rw [show 6 * ((2 ^ (8 * n + 4) - 4) / 6) = 2 ^ (8 * n + 4) - 4 by simpa [Nat.mul_comm] using hmul]
      exact Nat.sub_le _ _

instance instDecidableGood (k y : ℕ) : Decidable (Good k y) := by
  unfold Good Fk qTilde q0 z2 z1 z0 g h x W bw
  infer_instance

/--
The finite residual region from Lemma `residual` in the TeX proof.
-/
def ResidualRegion (k y : ℕ) : Prop :=
  (k = 3 ∧ ((8 ≤ y ∧ y < 32) ∨ (64 ≤ y ∧ y < 128))) ∨
  (k = 4 ∧ 16 ≤ y ∧ y < 32)

/--
Exact verification of the 104 residual cases.
This matches the paper's "Residual exact verification" lemma.
-/
theorem residual_exact {k y : ℕ} (h : ResidualRegion k y) : Good k y := by
  rcases h with ⟨rfl, h⟩ | ⟨rfl, hy16, hy32⟩
  · rcases h with ⟨hy8, hy32⟩ | ⟨hy64, hy128⟩
    · interval_cases y <;> decide
    · interval_cases y <;> decide
  · interval_cases y <;> decide

/-!
`thm_kge5` and `thm_all_k` below formalize the final reduction argument of the paper.
To keep this file honest, the three regime theorems are parameters of a section rather than
postulated as axioms.

To finish a full end-to-end formalization, instantiate the three hypotheses in `PaperCore` with
formal proofs of:

* the remaining tiny-input result for non-powers of two (`y = 5, 7`),
* the low-bit theorem for non-powers of two (`bw y ≤ k` and `4 ≤ bw y`),
* the middle/high-bit theorem.

Once those are available, the theorems at the end of the file give the final correctness result.
-/
section PaperCore

variable
  (tiny :
    ∀ {k y : ℕ}, 3 ≤ k → 1 ≤ y → y ≤ 7 → y ≠ 3 → y ≠ 6 → (¬ ∃ m : ℕ, y = 2 ^ m) → Good k y)
  (low_exact :
    ∀ {k y : ℕ}, 3 ≤ k → 1 < y → bw y ≤ k → 4 ≤ bw y → (¬ ∃ m : ℕ, y = 2 ^ m) → Good k y)
  (mid_high :
    ∀ {k y : ℕ},
      3 ≤ k →
      y < 2 ^ (W k) →
      (((k < bw y) ∧ (bw y ≤ 2 * k) ∧ 6 ≤ bw y) ∨ ((2 * k < bw y) ∧ 8 ≤ bw y)) →
      Good k y)

include tiny low_exact mid_high

/-- The paper's theorem `k ≥ 5`. -/
theorem thm_kge5 {k y : ℕ} (hk : 5 ≤ k) (hy1 : 1 ≤ y) (hy2 : y < 2 ^ (W k)) : Good k y := by
  have hk3 : 3 ≤ k := by omega
  by_cases hsmall : y ≤ 7
  · by_cases hpow : ∃ m : ℕ, y = 2 ^ m
    · rcases hpow with ⟨m, rfl⟩
      have hm3 : m + 1 ≤ 3 := by
        have hlt : 2 ^ m < 2 ^ 3 := by
          exact lt_of_le_of_lt hsmall (by norm_num)
        have hb : bw (2 ^ m) ≤ 3 := bw_le_of_lt_pow (by positivity) hlt
        simpa [bw_two_pow] using hb
      exact Good_power_of_two hk3 (by
        unfold W
        omega)
    · by_cases hy3 : y = 3
      · subst hy3
        exact Good_three hk3
      · by_cases hy6 : y = 6
        · subst hy6
          exact Good_six hk3
        · exact tiny hk3 hy1 hsmall hy3 hy6 hpow
  · have hy8 : 8 ≤ y := by omega
    have hy0 : y ≠ 0 := by omega
    have hb4 : 4 ≤ bw y := by
      have hpow : 2 ^ 3 ≤ y := by simpa using hy8
      have : 3 + 1 ≤ bw y := bw_ge_of_pow_le hy0 hpow
      simpa using this
    by_cases hbw : bw y ≤ k
    · by_cases hpow : ∃ m : ℕ, y = 2 ^ m
      · rcases hpow with ⟨m, rfl⟩
        have hm : m + 1 ≤ k := by simpa [bw_two_pow] using hbw
        exact Good_power_of_two hk3 (by
          unfold W
          omega)
      · have hygt1 : 1 < y := by omega
        exact low_exact hk3 hygt1 hbw hb4 hpow
    · have hlt : k < bw y := by omega
      by_cases h2k : bw y ≤ 2 * k
      · exact mid_high hk3 hy2 (Or.inl ⟨hlt, h2k, by omega⟩)
      · have hgt : 2 * k < bw y := by omega
        exact mid_high hk3 hy2 (Or.inr ⟨hgt, by omega⟩)

/-- The paper's final theorem for all `k ≥ 3`. -/
theorem thm_all_k {k y : ℕ} (hk : 3 ≤ k) (hy1 : 1 ≤ y) (hy2 : y < 2 ^ (W k)) : Good k y := by
  by_cases hk5 : 5 ≤ k
  · exact thm_kge5 tiny low_exact mid_high hk5 hy1 hy2
  · have hk34 : k = 3 ∨ k = 4 := by omega
    rcases hk34 with rfl | rfl
    · -- `k = 3`
      by_cases hsmall : y ≤ 7
      · by_cases hpow : ∃ m : ℕ, y = 2 ^ m
        · rcases hpow with ⟨m, rfl⟩
          have hm3 : m + 1 ≤ 3 := by
            have hlt : 2 ^ m < 2 ^ 3 := by
              exact lt_of_le_of_lt hsmall (by norm_num)
            have hb : bw (2 ^ m) ≤ 3 := bw_le_of_lt_pow (by positivity) hlt
            simpa [bw_two_pow] using hb
          exact Good_power_of_two (by omega) (by
            unfold W
            omega)
        · by_cases hy3 : y = 3
          · subst hy3
            exact Good_three (by omega)
          · by_cases hy6 : y = 6
            · subst hy6
              exact Good_six (by omega)
            · exact tiny (by omega) hy1 hsmall hy3 hy6 hpow
      · have hy8 : 8 ≤ y := by omega
        have hy0 : y ≠ 0 := by omega
        by_cases hy32 : y < 32
        · exact residual_exact (Or.inl ⟨rfl, Or.inl ⟨hy8, hy32⟩⟩)
        · have hy32le : 32 ≤ y := by omega
          by_cases hy64 : y < 64
          · have hb6 : 6 ≤ bw y := by
              have hpow : 2 ^ 5 ≤ y := by simpa using hy32le
              have : 5 + 1 ≤ bw y := bw_ge_of_pow_le hy0 hpow
              simpa using this
            have hb6' : bw y ≤ 6 := by
              have hpow : y < 2 ^ 6 := by simpa using hy64
              exact bw_le_of_lt_pow hy0 hpow
            exact mid_high (by omega) hy2 (Or.inl ⟨by omega, by omega, hb6⟩)
          · have hy64le : 64 ≤ y := by omega
            by_cases hy128 : y < 128
            · exact residual_exact (Or.inl ⟨rfl, Or.inr ⟨hy64le, hy128⟩⟩)
            · have hy128le : 128 ≤ y := by omega
              have hb8 : 8 ≤ bw y := by
                have hpow : 2 ^ 7 ≤ y := by simpa using hy128le
                have : 7 + 1 ≤ bw y := bw_ge_of_pow_le hy0 hpow
                simpa using this
              exact mid_high (by omega) hy2 (Or.inr ⟨by omega, hb8⟩)
    · -- `k = 4`
      by_cases hsmall : y ≤ 7
      · by_cases hpow : ∃ m : ℕ, y = 2 ^ m
        · rcases hpow with ⟨m, rfl⟩
          have hm3 : m + 1 ≤ 3 := by
            have hlt : 2 ^ m < 2 ^ 3 := by
              exact lt_of_le_of_lt hsmall (by norm_num)
            have hb : bw (2 ^ m) ≤ 3 := bw_le_of_lt_pow (by positivity) hlt
            simpa [bw_two_pow] using hb
          have hm4 : m + 1 ≤ 4 := by omega
          exact Good_power_of_two (by omega) (by
            unfold W
            omega)
        · by_cases hy3 : y = 3
          · subst hy3
            exact Good_three (by omega)
          · by_cases hy6 : y = 6
            · subst hy6
              exact Good_six (by omega)
            · exact tiny (by omega) hy1 hsmall hy3 hy6 hpow
      · have hy8 : 8 ≤ y := by omega
        have hy0 : y ≠ 0 := by omega
        by_cases hy16 : y < 16
        · have hb4 : 4 ≤ bw y := by
            have hpow : 2 ^ 3 ≤ y := by simpa using hy8
            have : 3 + 1 ≤ bw y := bw_ge_of_pow_le hy0 hpow
            simpa using this
          have hb4' : bw y ≤ 4 := by
            have hpow : y < 2 ^ 4 := by simpa using hy16
            exact bw_le_of_lt_pow hy0 hpow
          by_cases hpow : ∃ m : ℕ, y = 2 ^ m
          · rcases hpow with ⟨m, rfl⟩
            have hm : m + 1 ≤ 4 := by simpa [bw_two_pow] using hb4'
            exact Good_power_of_two (by omega) (by
              unfold W
              omega)
          · have hygt1 : 1 < y := by omega
            exact low_exact (by omega) hygt1 hb4' hb4 hpow
        · have hy16le : 16 ≤ y := by omega
          by_cases hy32 : y < 32
          · exact residual_exact (Or.inr ⟨rfl, hy16le, hy32⟩)
          · have hy32le : 32 ≤ y := by omega
            by_cases hy256 : y < 256
            · have hb6 : 6 ≤ bw y := by
                have hpow : 2 ^ 5 ≤ y := by simpa using hy32le
                have : 5 + 1 ≤ bw y := bw_ge_of_pow_le hy0 hpow
                simpa using this
              have hb8 : bw y ≤ 8 := by
                have hpow : y < 2 ^ 8 := by simpa using hy256
                exact bw_le_of_lt_pow hy0 hpow
              exact mid_high (by omega) hy2 (Or.inl ⟨by omega, hb8, hb6⟩)
            · have hy256le : 256 ≤ y := by omega
              have hb9 : 9 ≤ bw y := by
                have hpow : 2 ^ 8 ≤ y := by simpa using hy256le
                have : 8 + 1 ≤ bw y := bw_ge_of_pow_le hy0 hpow
                simpa using this
              exact mid_high (by omega) hy2 (Or.inr ⟨by omega, by omega⟩)


/-- The paper's `W = 32` corollary (`k = 8`). -/
theorem cor_w32 {y : ℕ} (hy1 : 1 ≤ y) (hy2 : y < 2 ^ 32) : Good 8 y := by
  have hy2' : y < 2 ^ (W 8) := by simpa [W] using hy2
  exact thm_all_k tiny low_exact mid_high (k := 8) (y := y) (by omega) hy1 hy2'

end PaperCore

end Reciprocal



open Reciprocal

namespace BW3

lemma x_bw3 {k y : ℕ} (hbw : bw y = 3) : x k y = y * 2 ^ (4 * k - 3) := by
  unfold x W
  rw [hbw]

lemma z0_bw3 {k y : ℕ} (hk : 3 ≤ k) (hbw : bw y = 3) :
    z0 k y = 2 ^ (k + 2) / y := by
  have hyk : bw y ≤ k := by
    rw [hbw]
    omega
  unfold z0
  rw [g_low hyk, hbw]
  rw [Nat.mul_comm y _, ← Nat.div_div_eq_div_mul]
  have hpow : 2 ^ (2 * k - 1) / 2 ^ (k - 3) = 2 ^ (k + 2) := by
    apply Nat.div_eq_of_eq_mul_right (by positivity : 0 < 2 ^ (k - 3))
    rw [show 2 * k - 1 = (k + 2) + (k - 3) by omega, pow_add, Nat.mul_comm]
  rw [hpow]

lemma z1_bw3 {k y : ℕ} (hk : 3 ≤ k) (hbw : bw y = 3) :
    z1 k y = 2 ^ k * z0 k y - (y * (z0 k y) ^ 2) / 8 := by
  have hyk : bw y ≤ k := by
    rw [hbw]
    omega
  unfold z1
  rw [h_low hyk, hbw]
  simp only
  have hinner : (y * 2 ^ (2 * k - 3) * z0 k y) / 2 ^ k = y * z0 k y * 2 ^ (k - 3) := by
    apply Nat.div_eq_of_eq_mul_right (by positivity : 0 < 2 ^ k)
    rw [show 2 * k - 3 = k + (k - 3) by omega, pow_add]
    ac_rfl
  rw [hinner]
  have hsecond : (z0 k y * (y * z0 k y * 2 ^ (k - 3))) / 2 ^ k = (y * (z0 k y) ^ 2) / 8 := by
    rw [show z0 k y * (y * z0 k y * 2 ^ (k - 3)) = (y * (z0 k y) ^ 2) * 2 ^ (k - 3) by
      rw [pow_two]
      ac_rfl]
    rw [show 2 ^ k = 2 ^ (k - 3) * 8 by
      rw [show k = (k - 3) + 3 by omega, pow_add]
      norm_num]
    rw [← Nat.div_div_eq_div_mul]
    have htmp : (y * (z0 k y) ^ 2 * 2 ^ (k - 3)) / 2 ^ (k - 3) = y * (z0 k y) ^ 2 := by
      apply Nat.div_eq_of_eq_mul_right (by positivity : 0 < 2 ^ (k - 3))
      ac_rfl
    exact congrArg (fun t => t / 8) htmp
  rw [hsecond]

lemma z2_bw3_core {k y : ℕ} (hk : 3 ≤ k) (hbw : bw y = 3) :
    z2 k y = 2 ^ (2 * k + 3) * z1 k y - 2 * y * (z1 k y) ^ 2 := by
  unfold z2
  simp_rw [x_bw3 hbw]
  have hinner : (y * 2 ^ (4 * k - 3) * z1 k y) / 2 ^ (2 * k - 4) = y * z1 k y * 2 ^ (2 * k + 1) := by
    apply Nat.div_eq_of_eq_mul_right (by positivity : 0 < 2 ^ (2 * k - 4))
    rw [show 4 * k - 3 = (2 * k - 4) + (2 * k + 1) by omega, pow_add]
    ac_rfl
  rw [hinner]
  have hcorr : (z1 k y * (y * z1 k y * 2 ^ (2 * k + 1))) / 2 ^ (2 * k) = 2 * y * (z1 k y) ^ 2 := by
    apply Nat.div_eq_of_eq_mul_right (by positivity : 0 < 2 ^ (2 * k))
    rw [show z1 k y * (y * z1 k y * 2 ^ (2 * k + 1)) = 2 ^ (2 * k) * (2 * y * (z1 k y) ^ 2) by
      rw [show 2 ^ (2 * k + 1) = 2 * 2 ^ (2 * k) by
        rw [show 2 * k + 1 = 2 * k + 1 by omega, pow_add]
        norm_num
        rw [Nat.mul_comm]]
      rw [pow_two]
      ac_rfl]
  rw [hcorr]

lemma z2_bw3_from_sub {k y c : ℕ} (hk : 3 ≤ k) (hbw : bw y = 3) (hy : 0 < y)
    (hc : 2 ^ (2 * k + 1) = y * z1 k y + c) :
    z2 k y = (2 ^ (W k + 3) - 2 * c ^ 2) / y := by
  set A : ℕ := 2 ^ (2 * k + 1)
  set t : ℕ := y * z1 k y
  have hAt : A = t + c := by
    simpa [A, t] using hc
  have hz2a : y * z2 k y = 4 * (t + c) * t - 2 * t ^ 2 := by
    rw [z2_bw3_core hk hbw]
    rw [show 2 ^ (2 * k + 3) = 4 * A by
      dsimp [A]
      rw [show 2 * k + 3 = (2 * k + 1) + 2 by omega, pow_add]
      norm_num
      ac_rfl]
    rw [hAt]
    rw [Nat.mul_sub_left_distrib]
    congr 1
    · dsimp [t]
      ring
    · dsimp [t]
      rw [pow_two]
      ring
  have hz2 : y * z2 k y = 2 * t ^ 2 + 4 * c * t := by
    rw [hz2a]
    have hprod : 4 * (t + c) * t = 4 * t ^ 2 + 4 * c * t := by
      ring
    rw [hprod]
    omega
  have hnum : 2 ^ (W k + 3) - 2 * c ^ 2 = 2 * t ^ 2 + 4 * c * t := by
    unfold W
    rw [show 2 ^ (4 * k + 3) = 2 * A ^ 2 by
      dsimp [A]
      rw [show 4 * k + 3 = 1 + 2 * (2 * k + 1) by ring, pow_add]
      norm_num
      rw [← pow_mul]
      congr 1
      ring]
    rw [hAt]
    have hsq : 2 * (t + c) ^ 2 = 2 * t ^ 2 + 4 * c * t + 2 * c ^ 2 := by
      ring
    rw [hsq]
    omega
  symm
  apply Nat.div_eq_of_eq_mul_right hy
  calc
    2 ^ (W k + 3) - 2 * c ^ 2 = 2 * t ^ 2 + 4 * c * t := hnum
    _ = y * z2 k y := by rw [hz2]

lemma z2_bw3_from_add {k y c : ℕ} (hk : 3 ≤ k) (hbw : bw y = 3) (hy : 0 < y)
    (hc : y * z1 k y = 2 ^ (2 * k + 1) + c) :
    z2 k y = (2 ^ (W k + 3) - 2 * c ^ 2) / y := by
  set A : ℕ := 2 ^ (2 * k + 1)
  set t : ℕ := y * z1 k y
  have htA : t = A + c := by
    simpa [A, t] using hc
  have hz2 : y * z2 k y = 4 * A * t - 2 * t ^ 2 := by
    rw [z2_bw3_core hk hbw]
    rw [show 2 ^ (2 * k + 3) = 4 * A by
      dsimp [A]
      rw [show 2 * k + 3 = (2 * k + 1) + 2 by omega, pow_add]
      norm_num
      ac_rfl]
    rw [Nat.mul_sub_left_distrib]
    congr 1
    · dsimp [t]
      ring
    · dsimp [t]
      rw [pow_two]
      ring
  have hnum : 2 ^ (W k + 3) - 2 * c ^ 2 = 4 * A * t - 2 * t ^ 2 := by
    unfold W
    rw [show 2 ^ (4 * k + 3) = 2 * A ^ 2 by
      dsimp [A]
      rw [show 4 * k + 3 = 1 + 2 * (2 * k + 1) by ring, pow_add]
      norm_num
      rw [← pow_mul]
      congr 1
      ring]
    have hprod : 4 * A * t = 4 * A ^ 2 + 4 * c * A := by
      rw [htA]
      ring
    have hsq : 2 * t ^ 2 = 2 * A ^ 2 + 4 * c * A + 2 * c ^ 2 := by
      rw [htA]
      ring
    rw [hprod, hsq]
    omega
  symm
  apply Nat.div_eq_of_eq_mul_right hy
  calc
    2 ^ (W k + 3) - 2 * c ^ 2 = 4 * A * t - 2 * t ^ 2 := hnum
    _ = y * z2 k y := by rw [hz2]

end BW3



open Reciprocal

namespace Five

lemma bw_five : bw 5 = 3 := by decide

lemma pow_two_four_mod_five (n : ℕ) : 2 ^ (4 * n) % 5 = 1 := by
  induction n with
  | zero =>
      norm_num
  | succ n ih =>
      rw [show 4 * (n + 1) = 4 * n + 4 by ring]
      rw [pow_add]
      norm_num [Nat.mul_mod, ih]

lemma u_five_four_mul (n : ℕ) :
    5 * ((2 ^ (4 * n) - 1) / 5) = 2 ^ (4 * n) - 1 := by
  have hdecomp := Nat.mod_add_div (2 ^ (4 * n)) 5
  rw [pow_two_four_mod_five] at hdecomp
  have hmul1 : 2 ^ (4 * n) / 5 * 5 = 2 ^ (4 * n) - 1 := by omega
  have hdvd : 5 ∣ 2 ^ (4 * n) - 1 := ⟨2 ^ (4 * n) / 5, by
    simpa [Nat.mul_comm] using hmul1.symm⟩
  simpa [Nat.mul_comm] using (Nat.div_mul_cancel hdvd)

lemma z0_five_mod0 {n : ℕ} (hn : 1 ≤ n) :
    z0 (4 * n) 5 = 4 * ((2 ^ (4 * n) - 1) / 5) := by
  have hyk : bw 5 ≤ 4 * n := by
    simp [bw_five]
    omega
  unfold z0
  rw [g_low hyk, bw_five]
  rw [Nat.mul_comm 5 _, ← Nat.div_div_eq_div_mul]
  have hpow : 2 ^ (2 * (4 * n) - 1) / 2 ^ (4 * n - 3) = 2 ^ (4 * n + 2) := by
    apply Nat.div_eq_of_eq_mul_right (by positivity : 0 < 2 ^ (4 * n - 3))
    rw [show 2 * (4 * n) - 1 = (4 * n + 2) + (4 * n - 3) by omega, pow_add, Nat.mul_comm]
  rw [hpow]
  set u : ℕ := (2 ^ (4 * n) - 1) / 5
  have hu : 5 * u = 2 ^ (4 * n) - 1 := by simp [u, u_five_four_mul]
  have hu' : 2 ^ (4 * n) = 5 * u + 1 := by
    calc
      2 ^ (4 * n) = (2 ^ (4 * n) - 1) + 1 := by
        have : 1 ≤ 2 ^ (4 * n) := Nat.succ_le_of_lt (by positivity)
        omega
      _ = 5 * u + 1 := by rw [← hu]
  rw [show 2 ^ (4 * n + 2) = 4 + 5 * (4 * u) by
    rw [show 4 * n + 2 = 4 * n + 2 by omega, pow_add]
    norm_num
    rw [hu']
    ring]
  -- simplified by the following division identity
  simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using
    (Nat.add_mul_div_right 4 (4 * u) (by decide : 0 < 5))

lemma d_five_mod0 {n : ℕ} (hn : 1 ≤ n) :
    2 ^ (8 * n + 1) - 5 * z1 (4 * n) 5 = 2 := by
  have hyk : bw 5 ≤ 4 * n := by
    simp [bw_five]
    omega
  unfold z1
  simp_rw [z0_five_mod0 hn, h_low hyk, bw_five]
  set u : ℕ := (2 ^ (4 * n) - 1) / 5
  have hu : 5 * u = 2 ^ (4 * n) - 1 := by simp [u, u_five_four_mul]
  have hu' : 2 ^ (4 * n) = 5 * u + 1 := by
    calc
      2 ^ (4 * n) = (2 ^ (4 * n) - 1) + 1 := by
        have : 1 ≤ 2 ^ (4 * n) := Nat.succ_le_of_lt (by positivity)
        omega
      _ = 5 * u + 1 := by rw [← hu]
  have hinner : (5 * 2 ^ (8 * n - 3) * (4 * u)) / 2 ^ (4 * n) = 5 * u * 2 ^ (4 * n - 1) := by
    apply Nat.div_eq_of_eq_mul_right (by positivity : 0 < 2 ^ (4 * n))
    rw [show 5 * 2 ^ (8 * n - 3) * (4 * u) = 2 ^ (4 * n) * (5 * u * 2 ^ (4 * n - 1)) by
      rw [show 8 * n - 3 = 4 * n + (4 * n - 3) by omega, pow_add]
      rw [show 2 ^ (4 * n - 1) = 4 * 2 ^ (4 * n - 3) by
        rw [show 4 * n - 1 = (4 * n - 3) + 2 by omega, pow_add]
        norm_num
        rw [Nat.mul_comm]]
      ac_rfl]
  have hinner' : 5 * 2 ^ (2 * (4 * n) - 3) * (4 * u) / 2 ^ (4 * n) = 5 * u * 2 ^ (4 * n - 1) := by
    simpa [show 2 * (4 * n) - 3 = 8 * n - 3 by omega] using hinner
  rw [hinner']
  have hsecond : (4 * u * (5 * u * 2 ^ (4 * n - 1))) / 2 ^ (4 * n) = 10 * u ^ 2 := by
    calc
      (4 * u * (5 * u * 2 ^ (4 * n - 1))) / 2 ^ (4 * n)
          = (10 * u ^ 2 * 2 ^ (4 * n)) / 2 ^ (4 * n) := by
              rw [show 2 ^ (4 * n) = 2 * 2 ^ (4 * n - 1) by
                rw [show 4 * n = (4 * n - 1) + 1 by omega, pow_add]
                norm_num
                rw [Nat.mul_comm]]
              ring_nf
      _ = 10 * u ^ 2 := by
            rw [Nat.mul_comm]
            exact Nat.mul_div_right (10 * u ^ 2) (by positivity : 0 < 2 ^ (4 * n))
  rw [hsecond]
  rw [show 2 ^ (4 * n) * (4 * u) = 4 * u * (5 * u + 1) by rw [hu']; ring]
  rw [show 2 ^ (8 * n + 1) = 2 * (5 * u + 1) ^ 2 by
    rw [show 8 * n + 1 = 1 + 2 * (4 * n) by ring, pow_add]
    norm_num
    have hsq : 2 ^ (2 * (4 * n)) = (2 ^ (4 * n)) ^ 2 := by
      rw [← pow_mul]
      congr 1
      ring
    rw [hsq, hu']
  ]
  rw [show 2 * (5 * u + 1) ^ 2 = 2 + u * 20 + u ^ 2 * 50 by ring]
  have hred0 : 4 * u * (5 * u + 1) = u * 4 + u ^ 2 * 20 := by
    ring
  have hred : 4 * u * (5 * u + 1) - 10 * u ^ 2 = u * 4 + u ^ 2 * 10 := by
    rw [hred0]
    omega
  rw [hred]
  rw [show 5 * (u * 4 + u ^ 2 * 10) = u * 20 + u ^ 2 * 50 by ring]
  omega

end Five



open Reciprocal
open BW3

namespace Five

lemma z0_five_mod1 {n : ℕ} (hn : 1 ≤ n) :
    z0 (4 * n + 1) 5 = 8 * ((2 ^ (4 * n) - 1) / 5) + 1 := by
  set u : ℕ := (2 ^ (4 * n) - 1) / 5
  have hu : 5 * u = 2 ^ (4 * n) - 1 := by simp [u, u_five_four_mul]
  have hu' : 2 ^ (4 * n) = 5 * u + 1 := by
    calc
      2 ^ (4 * n) = (2 ^ (4 * n) - 1) + 1 := by
        have : 1 ≤ 2 ^ (4 * n) := Nat.succ_le_of_lt (by positivity)
        omega
      _ = 5 * u + 1 := by rw [← hu]
  rw [z0_bw3 (k := 4 * n + 1) (y := 5) (by omega) bw_five]
  rw [show 2 ^ (4 * n + 3) = 5 * (8 * u + 1) + 3 by
    rw [show 4 * n + 3 = 4 * n + 3 by omega, pow_add]
    norm_num
    rw [hu']
    ring]
  -- simplified by the following division identity
  simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using
    (Nat.add_mul_div_right 3 (8 * u + 1) (by decide : 0 < 5))

lemma d_five_mod1 {n : ℕ} (hn : 1 ≤ n) :
    5 * z1 (4 * n + 1) 5 = 2 ^ (8 * n + 3) + 2 := by
  have hz1 := z1_bw3 (k := 4 * n + 1) (y := 5) (by omega) bw_five
  rw [hz1, z0_five_mod1 hn]
  set u : ℕ := (2 ^ (4 * n) - 1) / 5
  have hu : 5 * u = 2 ^ (4 * n) - 1 := by simp [u, u_five_four_mul]
  have hu' : 2 ^ (4 * n) = 5 * u + 1 := by
    calc
      2 ^ (4 * n) = (2 ^ (4 * n) - 1) + 1 := by
        have : 1 ≤ 2 ^ (4 * n) := Nat.succ_le_of_lt (by positivity)
        omega
      _ = 5 * u + 1 := by rw [← hu]
  have hfloor : (5 * (8 * u + 1) ^ 2) / 8 = u * 10 + u ^ 2 * 40 := by
    rw [show 5 * (8 * u + 1) ^ 2 = 5 + 8 * (u * 10 + u ^ 2 * 40) by ring]
    -- simplified by the following division identity
    simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc, Nat.mul_comm, Nat.mul_left_comm] using
      (Nat.add_mul_div_right 5 (u * 10 + u ^ 2 * 40) (by decide : 0 < 8))
  rw [hfloor]
  rw [show 2 ^ (4 * n + 1) = 2 * (5 * u + 1) by
    rw [show 4 * n + 1 = 4 * n + 1 by omega, pow_add]
    norm_num
    rw [hu', Nat.mul_comm]]
  rw [show 2 ^ (8 * n + 3) = 8 * (5 * u + 1) ^ 2 by
    rw [show 8 * n + 3 = 3 + 2 * (4 * n) by ring, pow_add]
    norm_num
    have hsq : 2 ^ (2 * (4 * n)) = (2 ^ (4 * n)) ^ 2 := by
      rw [← pow_mul]
      congr 1
      ring
    rw [hsq, hu']
  ]
  have hprod : 2 * (5 * u + 1) * (8 * u + 1) = 2 + u * 26 + u ^ 2 * 80 := by
    ring
  have hred : 2 * (5 * u + 1) * (8 * u + 1) - (u * 10 + u ^ 2 * 40) = 2 + u * 16 + u ^ 2 * 40 := by
    rw [hprod]
    omega
  rw [hred]
  ring_nf

lemma z0_five_mod2 {n : ℕ} (hn : 1 ≤ n) :
    z0 (4 * n + 2) 5 = 16 * ((2 ^ (4 * n) - 1) / 5) + 3 := by
  set u : ℕ := (2 ^ (4 * n) - 1) / 5
  have hu : 5 * u = 2 ^ (4 * n) - 1 := by simp [u, u_five_four_mul]
  have hu' : 2 ^ (4 * n) = 5 * u + 1 := by
    calc
      2 ^ (4 * n) = (2 ^ (4 * n) - 1) + 1 := by
        have : 1 ≤ 2 ^ (4 * n) := Nat.succ_le_of_lt (by positivity)
        omega
      _ = 5 * u + 1 := by rw [← hu]
  rw [z0_bw3 (k := 4 * n + 2) (y := 5) (by omega) bw_five]
  rw [show 2 ^ (4 * n + 4) = 5 * (16 * u + 3) + 1 by
    rw [show 4 * n + 4 = 4 * n + 4 by omega, pow_add]
    norm_num
    rw [hu']
    ring]
  -- simplified by the following division identity
  simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using
    (Nat.add_mul_div_right 1 (16 * u + 3) (by decide : 0 < 5))

lemma d_five_mod2 {n : ℕ} (hn : 1 ≤ n) :
    5 * z1 (4 * n + 2) 5 = 2 ^ (8 * n + 5) + 3 := by
  have hz1 := z1_bw3 (k := 4 * n + 2) (y := 5) (by omega) bw_five
  rw [hz1, z0_five_mod2 hn]
  set u : ℕ := (2 ^ (4 * n) - 1) / 5
  have hu : 5 * u = 2 ^ (4 * n) - 1 := by simp [u, u_five_four_mul]
  have hu' : 2 ^ (4 * n) = 5 * u + 1 := by
    calc
      2 ^ (4 * n) = (2 ^ (4 * n) - 1) + 1 := by
        have : 1 ≤ 2 ^ (4 * n) := Nat.succ_le_of_lt (by positivity)
        omega
      _ = 5 * u + 1 := by rw [← hu]
  have hfloor : (5 * (16 * u + 3) ^ 2) / 8 = 5 + (u * 60 + u ^ 2 * 160) := by
    rw [show 5 * (16 * u + 3) ^ 2 = 5 + 8 * (5 + (u * 60 + u ^ 2 * 160)) by ring]
    -- simplified by the following division identity
    simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc, Nat.mul_comm, Nat.mul_left_comm] using
      (Nat.add_mul_div_right 5 (5 + (u * 60 + u ^ 2 * 160)) (by decide : 0 < 8))
  rw [hfloor]
  rw [show 2 ^ (4 * n + 2) = 4 * (5 * u + 1) by
    rw [show 4 * n + 2 = 4 * n + 2 by omega, pow_add]
    norm_num
    rw [hu', Nat.mul_comm]]
  rw [show 2 ^ (8 * n + 5) = 32 * (5 * u + 1) ^ 2 by
    rw [show 8 * n + 5 = 5 + 2 * (4 * n) by ring, pow_add]
    norm_num
    have hsq : 2 ^ (2 * (4 * n)) = (2 ^ (4 * n)) ^ 2 := by
      rw [← pow_mul]
      congr 1
      ring
    rw [hsq, hu']
  ]
  have hprod : 4 * (5 * u + 1) * (16 * u + 3) = 12 + u * 124 + u ^ 2 * 320 := by
    ring
  have hred :
      4 * (5 * u + 1) * (16 * u + 3) - (5 + (u * 60 + u ^ 2 * 160)) = 7 + u * 64 + u ^ 2 * 160 := by
    rw [hprod]
    omega
  rw [hred]
  ring_nf

lemma z0_five_mod3 (n : ℕ) :
    z0 (4 * n + 3) 5 = 32 * ((2 ^ (4 * n) - 1) / 5) + 6 := by
  set u : ℕ := (2 ^ (4 * n) - 1) / 5
  have hu : 5 * u = 2 ^ (4 * n) - 1 := by simp [u, u_five_four_mul]
  have hu' : 2 ^ (4 * n) = 5 * u + 1 := by
    calc
      2 ^ (4 * n) = (2 ^ (4 * n) - 1) + 1 := by
        have : 1 ≤ 2 ^ (4 * n) := Nat.succ_le_of_lt (by positivity)
        omega
      _ = 5 * u + 1 := by rw [← hu]
  rw [z0_bw3 (k := 4 * n + 3) (y := 5) (by omega) bw_five]
  rw [show 2 ^ (4 * n + 5) = 5 * (32 * u + 6) + 2 by
    rw [show 4 * n + 5 = 4 * n + 5 by omega, pow_add]
    norm_num
    rw [hu']
    ring]
  -- simplified by the following division identity
  simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using
    (Nat.add_mul_div_right 2 (32 * u + 6) (by decide : 0 < 5))

lemma d_five_mod3 (n : ℕ) :
    5 * z1 (4 * n + 3) 5 = 2 ^ (8 * n + 7) + 2 := by
  have hz1 := z1_bw3 (k := 4 * n + 3) (y := 5) (by omega) bw_five
  rw [hz1, z0_five_mod3 n]
  set u : ℕ := (2 ^ (4 * n) - 1) / 5
  have hu : 5 * u = 2 ^ (4 * n) - 1 := by simp [u, u_five_four_mul]
  have hu' : 2 ^ (4 * n) = 5 * u + 1 := by
    calc
      2 ^ (4 * n) = (2 ^ (4 * n) - 1) + 1 := by
        have : 1 ≤ 2 ^ (4 * n) := Nat.succ_le_of_lt (by positivity)
        omega
      _ = 5 * u + 1 := by rw [← hu]
  have hfloor : (5 * (32 * u + 6) ^ 2) / 8 = 22 + (u * 240 + u ^ 2 * 640) := by
    rw [show 5 * (32 * u + 6) ^ 2 = 4 + 8 * (22 + (u * 240 + u ^ 2 * 640)) by ring]
    -- simplified by the following division identity
    simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc, Nat.mul_comm, Nat.mul_left_comm] using
      (Nat.add_mul_div_right 4 (22 + (u * 240 + u ^ 2 * 640)) (by decide : 0 < 8))
  rw [hfloor]
  rw [show 2 ^ (4 * n + 3) = 8 * (5 * u + 1) by
    rw [show 4 * n + 3 = 4 * n + 3 by omega, pow_add]
    norm_num
    rw [hu', Nat.mul_comm]]
  rw [show 2 ^ (8 * n + 7) = 128 * (5 * u + 1) ^ 2 by
    rw [show 8 * n + 7 = 7 + 2 * (4 * n) by ring, pow_add]
    norm_num
    have hsq : 2 ^ (2 * (4 * n)) = (2 ^ (4 * n)) ^ 2 := by
      rw [← pow_mul]
      congr 1
      ring
    rw [hsq, hu']
  ]
  have hprod : 8 * (5 * u + 1) * (32 * u + 6) = 48 + u * 496 + u ^ 2 * 1280 := by
    ring
  have hred :
      8 * (5 * u + 1) * (32 * u + 6) - (22 + (u * 240 + u ^ 2 * 640)) = 26 + u * 256 + u ^ 2 * 640 := by
    rw [hprod]
    omega
  rw [hred]
  ring_nf

lemma q_five_mul (k : ℕ) :
    5 * ((2 ^ (W k) - 1) / 5) = 2 ^ (W k) - 1 := by
  simpa [W] using u_five_four_mul k

lemma q0_five_c2 (t : ℕ) :
    ((2 ^ (4 * t + 3) - 8) / 5) / 8 = (2 ^ (4 * t) - 1) / 5 := by
  set q : ℕ := (2 ^ (4 * t) - 1) / 5
  have hq : 5 * q = 2 ^ (4 * t) - 1 := by
    simp [q, u_five_four_mul]
  have hA : (2 ^ (4 * t + 3) - 8) / 5 = 8 * q := by
    apply Nat.div_eq_of_eq_mul_right (by decide : 0 < 5)
    rw [show 2 ^ (4 * t + 3) - 8 = 8 * (2 ^ (4 * t) - 1) by
      rw [show 4 * t + 3 = 3 + 4 * t by ring, pow_add]
      norm_num
      omega]
    rw [← hq]
    ring
  rw [hA]
  exact Nat.mul_div_right q (by decide : 0 < 8)

lemma q0_five_c3 {t : ℕ} (ht : 1 ≤ t) :
    ((2 ^ (4 * t + 3) - 18) / 5) / 8 = (2 ^ (4 * t) - 1) / 5 - 1 := by
  set q : ℕ := (2 ^ (4 * t) - 1) / 5
  have hq : 5 * q = 2 ^ (4 * t) - 1 := by
    simp [q, u_five_four_mul]
  have hq1 : 1 ≤ q := by
    have h16 : 16 ≤ 2 ^ (4 * t) := by
      have : 2 ^ 4 ≤ 2 ^ (4 * t) := by
        apply Nat.pow_le_pow_right
        decide
        omega
      simpa using this
    have h15 : 15 ≤ 5 * q := by
      rw [hq]
      omega
    have h5 : 5 ≤ 5 * q := by omega
    omega
  have hA : (2 ^ (4 * t + 3) - 18) / 5 = 8 * q - 2 := by
    apply Nat.div_eq_of_eq_mul_right (by decide : 0 < 5)
    rw [show 2 ^ (4 * t + 3) - 18 = 5 * (8 * q - 2) by
      rw [show 2 ^ (4 * t + 3) - 18 = 8 * (2 ^ (4 * t) - 1) - 10 by
        rw [show 4 * t + 3 = 3 + 4 * t by ring, pow_add]
        norm_num
        omega]
      rw [← hq]
      omega]
  rw [hA]
  rw [show 8 * q - 2 = 6 + 8 * (q - 1) by omega]
  -- simplified by the following division identity
  simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc, Nat.mul_comm, Nat.mul_left_comm] using
    (Nat.add_mul_div_right 6 (q - 1) (by decide : 0 < 8))

lemma q0_five_mod0 {n : ℕ} (hn : 1 ≤ n) :
    q0 (4 * n) 5 = (2 ^ (W (4 * n)) - 1) / 5 := by
  unfold q0
  have hc : 2 ^ (8 * n + 1) = 5 * z1 (4 * n) 5 + 2 := by
    have h := Five.d_five_mod0 hn
    omega
  rw [z2_bw3_from_sub (k := 4 * n) (y := 5) (c := 2) (by omega) bw_five (by decide)
    (by simpa [show 2 * (4 * n) + 1 = 8 * n + 1 by ring] using hc), bw_five]
  rw [show W (4 * n) = 4 * (4 * n) by unfold W; ring]
  exact q0_five_c2 (4 * n)

lemma q0_five_mod1 {n : ℕ} (hn : 1 ≤ n) :
    q0 (4 * n + 1) 5 = (2 ^ (W (4 * n + 1)) - 1) / 5 := by
  unfold q0
  rw [z2_bw3_from_add (k := 4 * n + 1) (y := 5) (c := 2) (by omega) bw_five (by decide)
    (by simpa [show 2 * (4 * n + 1) + 1 = 8 * n + 3 by ring] using d_five_mod1 hn), bw_five]
  rw [show W (4 * n + 1) = 4 * (4 * n + 1) by unfold W; ring]
  exact q0_five_c2 (4 * n + 1)

lemma q0_five_mod2 {n : ℕ} (hn : 1 ≤ n) :
    q0 (4 * n + 2) 5 = (2 ^ (W (4 * n + 2)) - 1) / 5 - 1 := by
  unfold q0
  rw [z2_bw3_from_add (k := 4 * n + 2) (y := 5) (c := 3) (by omega) bw_five (by decide)
    (by simpa [show 2 * (4 * n + 2) + 1 = 8 * n + 5 by ring] using d_five_mod2 hn), bw_five]
  rw [show W (4 * n + 2) = 4 * (4 * n + 2) by unfold W; ring]
  exact q0_five_c3 (by omega : 1 ≤ 4 * n + 2)

lemma q0_five_mod3 (n : ℕ) :
    q0 (4 * n + 3) 5 = (2 ^ (W (4 * n + 3)) - 1) / 5 := by
  unfold q0
  rw [z2_bw3_from_add (k := 4 * n + 3) (y := 5) (c := 2) (by omega) bw_five (by decide)
    (by simpa [show 2 * (4 * n + 3) + 1 = 8 * n + 7 by ring] using d_five_mod3 n), bw_five]
  rw [show W (4 * n + 3) = 4 * (4 * n + 3) by unfold W; ring]
  exact q0_five_c2 (4 * n + 3)

lemma Fk_five_of_q0 {k : ℕ} (hq0 : q0 k 5 = (2 ^ (W k) - 1) / 5) :
    Fk k 5 = (2 ^ (W k) - 1) / 5 := by
  unfold Fk qTilde
  rw [hq0]
  have hclamp : (2 ^ (W k) - 1) / 5 ≤ 2 ^ (W k) - 1 := by
    exact Nat.div_le_self _ _
  rw [min_eq_left hclamp]
  have hmul : 5 * ((2 ^ (W k) - 1) / 5) = 2 ^ (W k) - 1 := q_five_mul k
  have hfalse : ¬((2 ^ (W k) - 1) / 5) * 5 < 2 ^ (W k) - 5 := by
    rw [show ((2 ^ (W k) - 1) / 5) * 5 = 2 ^ (W k) - 1 by simpa [Nat.mul_comm] using hmul]
    omega
  simp [hfalse]

lemma Fk_five_mod2 {n : ℕ} (hn : 1 ≤ n) :
    Fk (4 * n + 2) 5 = (2 ^ (W (4 * n + 2)) - 1) / 5 := by
  unfold Fk qTilde
  rw [q0_five_mod2 hn]
  set q : ℕ := (2 ^ (W (4 * n + 2)) - 1) / 5
  have hclamp : q - 1 ≤ 2 ^ (W (4 * n + 2)) - 1 := by
    exact le_trans (Nat.sub_le _ _) (Nat.div_le_self _ _)
  rw [min_eq_left hclamp]
  have hmul : 5 * q = 2 ^ (W (4 * n + 2)) - 1 := by
    simp [q, q_five_mul]
  have hq1 : 1 ≤ q := by
    have h16 : 16 ≤ 2 ^ (W (4 * n + 2)) := by
      have hexp : 4 ≤ W (4 * n + 2) := by
        unfold W
        omega
      have : 2 ^ 4 ≤ 2 ^ (W (4 * n + 2)) := by
        exact Nat.pow_le_pow_right (by decide) hexp
      simpa using this
    have h15 : 15 ≤ 5 * q := by
      rw [hmul]
      omega
    have h5 : 5 ≤ 5 * q := by omega
    omega
  have htrue : (q - 1) * 5 < 2 ^ (W (4 * n + 2)) - 5 := by
    rw [show (q - 1) * 5 = 5 * q - 5 by omega]
    rw [hmul]
    omega
  have hqsucc : q - 1 + 1 = q := by omega
  simp [q, htrue, hqsucc]

lemma Fk_five {k : ℕ} (hk : 3 ≤ k) : Fk k 5 = (2 ^ (W k) - 1) / 5 := by
  have hmod : k % 4 < 4 := Nat.mod_lt _ (by decide)
  interval_cases hkmod : k % 4
  · have hk0 : k = 4 * (k / 4) := by
      simpa [hkmod, Nat.mul_comm] using (Nat.div_add_mod k 4).symm
    rw [hk0]
    exact Fk_five_of_q0 (q0_five_mod0 (by omega))
  · have hk1 : k = 4 * (k / 4) + 1 := by
      simpa [hkmod, Nat.mul_comm] using (Nat.div_add_mod k 4).symm
    rw [hk1]
    exact Fk_five_of_q0 (q0_five_mod1 (by omega))
  · have hk2 : k = 4 * (k / 4) + 2 := by
      simpa [hkmod, Nat.mul_comm] using (Nat.div_add_mod k 4).symm
    rw [hk2]
    exact Fk_five_mod2 (by omega)
  · have hk3 : k = 4 * (k / 4) + 3 := by
      simpa [hkmod, Nat.mul_comm] using (Nat.div_add_mod k 4).symm
    rw [hk3]
    exact Fk_five_of_q0 (q0_five_mod3 _)

theorem Good_five {k : ℕ} (hk : 3 ≤ k) : Good k 5 := by
  unfold Good
  rw [Fk_five hk]
  have hmul : 5 * ((2 ^ (W k) - 1) / 5) = 2 ^ (W k) - 1 := q_five_mul k
  constructor
  · rw [show 5 * ((2 ^ (W k) - 1) / 5) = 2 ^ (W k) - 1 by simpa [Nat.mul_comm] using hmul]
    exact Nat.sub_le_sub_left (by decide : 1 ≤ 5) _
  · rw [show 5 * ((2 ^ (W k) - 1) / 5) = 2 ^ (W k) - 1 by simpa [Nat.mul_comm] using hmul]
    exact Nat.sub_le _ _

end Five



open Reciprocal
open BW3

namespace Seven

lemma bw_seven : bw 7 = 3 := by decide

lemma pow_two_three_mod_seven (n : ℕ) : 2 ^ (3 * n) % 7 = 1 := by
  induction n with
  | zero =>
      norm_num
  | succ n ih =>
      rw [show 3 * (n + 1) = 3 * n + 3 by ring]
      rw [pow_add]
      norm_num [Nat.mul_mod, ih]

lemma u_seven_three_mul (n : ℕ) :
    7 * ((2 ^ (3 * n) - 1) / 7) = 2 ^ (3 * n) - 1 := by
  have hdecomp := Nat.mod_add_div (2 ^ (3 * n)) 7
  rw [pow_two_three_mod_seven] at hdecomp
  have hmul1 : 2 ^ (3 * n) / 7 * 7 = 2 ^ (3 * n) - 1 := by omega
  have hdvd : 7 ∣ 2 ^ (3 * n) - 1 := ⟨2 ^ (3 * n) / 7, by
    simpa [Nat.mul_comm] using hmul1.symm⟩
  simpa [Nat.mul_comm] using (Nat.div_mul_cancel hdvd)

lemma z0_seven_mod0 {n : ℕ} (hn : 1 ≤ n) :
    z0 (3 * n) 7 = 4 * ((2 ^ (3 * n) - 1) / 7) := by
  set u : ℕ := (2 ^ (3 * n) - 1) / 7
  have hu : 7 * u = 2 ^ (3 * n) - 1 := by simp [u, u_seven_three_mul]
  have hu' : 2 ^ (3 * n) = 7 * u + 1 := by
    calc
      2 ^ (3 * n) = (2 ^ (3 * n) - 1) + 1 := by
        have : 1 ≤ 2 ^ (3 * n) := Nat.succ_le_of_lt (by positivity)
        omega
      _ = 7 * u + 1 := by rw [← hu]
  rw [z0_bw3 (k := 3 * n) (y := 7) (by omega) bw_seven]
  rw [show 2 ^ (3 * n + 2) = 7 * (4 * u) + 4 by
    rw [show 3 * n + 2 = 2 + 3 * n by ring, pow_add]
    norm_num
    rw [hu']
    ring]
  -- simplified by the following division identity
  simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using
    (Nat.add_mul_div_right 4 (4 * u) (by decide : 0 < 7))

lemma d_seven_mod0 {n : ℕ} (hn : 1 ≤ n) :
    2 ^ (6 * n + 1) = 7 * z1 (3 * n) 7 + 2 := by
  have hz1 := z1_bw3 (k := 3 * n) (y := 7) (by omega) bw_seven
  rw [hz1, z0_seven_mod0 hn]
  set u : ℕ := (2 ^ (3 * n) - 1) / 7
  have hu : 7 * u = 2 ^ (3 * n) - 1 := by simp [u, u_seven_three_mul]
  have hu' : 2 ^ (3 * n) = 7 * u + 1 := by
    calc
      2 ^ (3 * n) = (2 ^ (3 * n) - 1) + 1 := by
        have : 1 ≤ 2 ^ (3 * n) := Nat.succ_le_of_lt (by positivity)
        omega
      _ = 7 * u + 1 := by rw [← hu]
  rw [show (7 * (4 * u) ^ 2) / 8 = 14 * u ^ 2 by
    rw [show 7 * (4 * u) ^ 2 = 8 * (14 * u ^ 2) by ring]
    norm_num]
  rw [show 2 ^ (3 * n) = 7 * u + 1 by exact hu']
  rw [show 2 ^ (6 * n + 1) = 2 * (7 * u + 1) ^ 2 by
    rw [show 6 * n + 1 = 1 + 2 * (3 * n) by ring, pow_add]
    norm_num
    have hsq : 2 ^ (2 * (3 * n)) = (2 ^ (3 * n)) ^ 2 := by
      rw [← pow_mul]
      congr 1
      ring
    rw [hsq, hu']
  ]
  have hprod : (7 * u + 1) * (4 * u) = u * 4 + u ^ 2 * 28 := by
    ring
  rw [hprod]
  rw [show 14 * u ^ 2 = u ^ 2 * 14 by ring]
  have hred : u * 4 + u ^ 2 * 28 - u ^ 2 * 14 = u * 4 + u ^ 2 * 14 := by
    omega
  rw [hred]
  ring_nf

lemma z0_seven_mod1 {n : ℕ} (hn : 1 ≤ n) :
    z0 (3 * n + 1) 7 = 8 * ((2 ^ (3 * n) - 1) / 7) + 1 := by
  set u : ℕ := (2 ^ (3 * n) - 1) / 7
  have hu : 7 * u = 2 ^ (3 * n) - 1 := by simp [u, u_seven_three_mul]
  have hu' : 2 ^ (3 * n) = 7 * u + 1 := by
    calc
      2 ^ (3 * n) = (2 ^ (3 * n) - 1) + 1 := by
        have : 1 ≤ 2 ^ (3 * n) := Nat.succ_le_of_lt (by positivity)
        omega
      _ = 7 * u + 1 := by rw [← hu]
  rw [z0_bw3 (k := 3 * n + 1) (y := 7) (by omega) bw_seven]
  rw [show 2 ^ (3 * n + 3) = 7 * (8 * u + 1) + 1 by
    rw [show 3 * n + 3 = 3 + 3 * n by ring, pow_add]
    norm_num
    rw [hu']
    ring]
  -- simplified by the following division identity
  simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using
    (Nat.add_mul_div_right 1 (8 * u + 1) (by decide : 0 < 7))

lemma d_seven_mod1 {n : ℕ} (hn : 1 ≤ n) :
    7 * z1 (3 * n + 1) 7 = 2 ^ (6 * n + 3) + 6 := by
  have hz1 := z1_bw3 (k := 3 * n + 1) (y := 7) (by omega) bw_seven
  rw [hz1, z0_seven_mod1 hn]
  set u : ℕ := (2 ^ (3 * n) - 1) / 7
  have hu : 7 * u = 2 ^ (3 * n) - 1 := by simp [u, u_seven_three_mul]
  have hu' : 2 ^ (3 * n) = 7 * u + 1 := by
    calc
      2 ^ (3 * n) = (2 ^ (3 * n) - 1) + 1 := by
        have : 1 ≤ 2 ^ (3 * n) := Nat.succ_le_of_lt (by positivity)
        omega
      _ = 7 * u + 1 := by rw [← hu]
  have hfloor : (7 * (8 * u + 1) ^ 2) / 8 = u * 14 + u ^ 2 * 56 := by
    rw [show 7 * (8 * u + 1) ^ 2 = 7 + 8 * (u * 14 + u ^ 2 * 56) by ring]
    -- simplified by the following division identity
    simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc, Nat.mul_comm, Nat.mul_left_comm] using
      (Nat.add_mul_div_right 7 (u * 14 + u ^ 2 * 56) (by decide : 0 < 8))
  rw [hfloor]
  rw [show 2 ^ (3 * n + 1) = 2 * (7 * u + 1) by
    rw [show 3 * n + 1 = 1 + 3 * n by ring, pow_add]
    norm_num
    rw [hu', Nat.mul_comm]]
  rw [show 2 ^ (6 * n + 3) = 8 * (7 * u + 1) ^ 2 by
    rw [show 6 * n + 3 = 3 + 2 * (3 * n) by ring, pow_add]
    norm_num
    have hsq : 2 ^ (2 * (3 * n)) = (2 ^ (3 * n)) ^ 2 := by
      rw [← pow_mul]
      congr 1
      ring
    rw [hsq, hu']
  ]
  have hprod : 2 * (7 * u + 1) * (8 * u + 1) = 2 + u * 30 + u ^ 2 * 112 := by
    ring
  have hred : 2 * (7 * u + 1) * (8 * u + 1) - (u * 14 + u ^ 2 * 56) = 2 + u * 16 + u ^ 2 * 56 := by
    rw [hprod]
    omega
  rw [hred]
  ring_nf

lemma z0_seven_mod2 {n : ℕ} (hn : 1 ≤ n) :
    z0 (3 * n + 2) 7 = 16 * ((2 ^ (3 * n) - 1) / 7) + 2 := by
  set u : ℕ := (2 ^ (3 * n) - 1) / 7
  have hu : 7 * u = 2 ^ (3 * n) - 1 := by simp [u, u_seven_three_mul]
  have hu' : 2 ^ (3 * n) = 7 * u + 1 := by
    calc
      2 ^ (3 * n) = (2 ^ (3 * n) - 1) + 1 := by
        have : 1 ≤ 2 ^ (3 * n) := Nat.succ_le_of_lt (by positivity)
        omega
      _ = 7 * u + 1 := by rw [← hu]
  rw [z0_bw3 (k := 3 * n + 2) (y := 7) (by omega) bw_seven]
  rw [show 2 ^ (3 * n + 4) = 7 * (16 * u + 2) + 2 by
    rw [show 3 * n + 4 = 4 + 3 * n by ring, pow_add]
    norm_num
    rw [hu']
    ring]
  -- simplified by the following division identity
  simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using
    (Nat.add_mul_div_right 2 (16 * u + 2) (by decide : 0 < 7))

lemma d_seven_mod2 {n : ℕ} (hn : 1 ≤ n) :
    7 * z1 (3 * n + 2) 7 = 2 ^ (6 * n + 5) + 3 := by
  have hz1 := z1_bw3 (k := 3 * n + 2) (y := 7) (by omega) bw_seven
  rw [hz1, z0_seven_mod2 hn]
  set u : ℕ := (2 ^ (3 * n) - 1) / 7
  have hu : 7 * u = 2 ^ (3 * n) - 1 := by simp [u, u_seven_three_mul]
  have hu' : 2 ^ (3 * n) = 7 * u + 1 := by
    calc
      2 ^ (3 * n) = (2 ^ (3 * n) - 1) + 1 := by
        have : 1 ≤ 2 ^ (3 * n) := Nat.succ_le_of_lt (by positivity)
        omega
      _ = 7 * u + 1 := by rw [← hu]
  have hfloor : (7 * (16 * u + 2) ^ 2) / 8 = 3 + (u * 56 + u ^ 2 * 224) := by
    rw [show 7 * (16 * u + 2) ^ 2 = 4 + 8 * (3 + (u * 56 + u ^ 2 * 224)) by ring]
    -- simplified by the following division identity
    simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc, Nat.mul_comm, Nat.mul_left_comm] using
      (Nat.add_mul_div_right 4 (3 + (u * 56 + u ^ 2 * 224)) (by decide : 0 < 8))
  rw [hfloor]
  rw [show 2 ^ (3 * n + 2) = 4 * (7 * u + 1) by
    rw [show 3 * n + 2 = 2 + 3 * n by ring, pow_add]
    norm_num
    rw [hu', Nat.mul_comm]]
  rw [show 2 ^ (6 * n + 5) = 32 * (7 * u + 1) ^ 2 by
    rw [show 6 * n + 5 = 5 + 2 * (3 * n) by ring, pow_add]
    norm_num
    have hsq : 2 ^ (2 * (3 * n)) = (2 ^ (3 * n)) ^ 2 := by
      rw [← pow_mul]
      congr 1
      ring
    rw [hsq, hu']
  ]
  have hprod : 4 * (7 * u + 1) * (16 * u + 2) = 8 + u * 120 + u ^ 2 * 448 := by
    ring
  have hred :
      4 * (7 * u + 1) * (16 * u + 2) - (3 + (u * 56 + u ^ 2 * 224)) = 5 + u * 64 + u ^ 2 * 224 := by
    rw [hprod]
    omega
  rw [hred]
  ring_nf

lemma pow_two_three_plus_one_mod_seven (n : ℕ) : 2 ^ (3 * n + 1) % 7 = 2 := by
  rw [show 3 * n + 1 = 1 + 3 * n by ring, pow_add]
  norm_num [pow_two_three_mod_seven, Nat.mul_mod]

lemma pow_two_three_plus_two_mod_seven (n : ℕ) : 2 ^ (3 * n + 2) % 7 = 4 := by
  rw [show 3 * n + 2 = 2 + 3 * n by ring, pow_add]
  norm_num [pow_two_three_mod_seven, Nat.mul_mod]

lemma u_seven_three_plus_one_mul (n : ℕ) :
    7 * ((2 ^ (3 * n + 1) - 2) / 7) = 2 ^ (3 * n + 1) - 2 := by
  have hdecomp := Nat.mod_add_div (2 ^ (3 * n + 1)) 7
  rw [pow_two_three_plus_one_mod_seven] at hdecomp
  have hmul1 : 2 ^ (3 * n + 1) / 7 * 7 = 2 ^ (3 * n + 1) - 2 := by omega
  have hdvd : 7 ∣ 2 ^ (3 * n + 1) - 2 := ⟨2 ^ (3 * n + 1) / 7, by
    simpa [Nat.mul_comm] using hmul1.symm⟩
  simpa [Nat.mul_comm] using (Nat.div_mul_cancel hdvd)

lemma u_seven_three_plus_two_mul (n : ℕ) :
    7 * ((2 ^ (3 * n + 2) - 4) / 7) = 2 ^ (3 * n + 2) - 4 := by
  have hdecomp := Nat.mod_add_div (2 ^ (3 * n + 2)) 7
  rw [pow_two_three_plus_two_mod_seven] at hdecomp
  have hmul1 : 2 ^ (3 * n + 2) / 7 * 7 = 2 ^ (3 * n + 2) - 4 := by omega
  have hdvd : 7 ∣ 2 ^ (3 * n + 2) - 4 := ⟨2 ^ (3 * n + 2) / 7, by
    simpa [Nat.mul_comm] using hmul1.symm⟩
  simpa [Nat.mul_comm] using (Nat.div_mul_cancel hdvd)

lemma q0_seven_c2 (t : ℕ) :
    ((2 ^ (3 * t + 3) - 8) / 7) / 8 = (2 ^ (3 * t) - 1) / 7 := by
  set q : ℕ := (2 ^ (3 * t) - 1) / 7
  have hq : 7 * q = 2 ^ (3 * t) - 1 := by
    simp [q, u_seven_three_mul]
  have hA : (2 ^ (3 * t + 3) - 8) / 7 = 8 * q := by
    apply Nat.div_eq_of_eq_mul_right (by decide : 0 < 7)
    rw [show 2 ^ (3 * t + 3) - 8 = 8 * (2 ^ (3 * t) - 1) by
      rw [show 3 * t + 3 = 3 + 3 * t by ring, pow_add]
      norm_num
      omega]
    rw [← hq]
    ring
  rw [hA]
  exact Nat.mul_div_right q (by decide : 0 < 8)

lemma q0_seven_c6 {t : ℕ} (ht : 1 ≤ t) :
    ((2 ^ (3 * t + 4) - 72) / 7) / 8 = (2 ^ (3 * t + 1) - 2) / 7 - 1 := by
  set q : ℕ := (2 ^ (3 * t + 1) - 2) / 7
  have hq : 7 * q = 2 ^ (3 * t + 1) - 2 := by
    simp [q, u_seven_three_plus_one_mul]
  have hq1 : 1 ≤ q := by
    have h8 : 8 ≤ 2 ^ (3 * t + 1) := by
      have : 2 ^ 3 ≤ 2 ^ (3 * t + 1) := by
        apply Nat.pow_le_pow_right
        decide
        omega
      simpa using this
    have h6 : 6 ≤ 7 * q := by
      rw [hq]
      omega
    have h7 : 7 ≤ 7 * q := by omega
    omega
  have hA : (2 ^ (3 * t + 4) - 72) / 7 = 8 * q - 8 := by
    apply Nat.div_eq_of_eq_mul_right (by decide : 0 < 7)
    rw [show 2 ^ (3 * t + 4) - 72 = 7 * (8 * q - 8) by
      rw [show 2 ^ (3 * t + 4) - 72 = 8 * (2 ^ (3 * t + 1) - 2) - 56 by
        rw [show 3 * t + 4 = 3 + (3 * t + 1) by ring, pow_add]
        norm_num
        omega]
      rw [← hq]
      omega]
  rw [hA]
  rw [show 8 * q - 8 = 8 * (q - 1) by omega]
  exact Nat.mul_div_right (q - 1) (by decide : 0 < 8)

lemma q0_seven_c3 (t : ℕ) :
    ((2 ^ (3 * t + 5) - 18) / 7) / 8 = (2 ^ (3 * t + 2) - 4) / 7 := by
  set q : ℕ := (2 ^ (3 * t + 2) - 4) / 7
  have hq : 7 * q = 2 ^ (3 * t + 2) - 4 := by
    simp [q, u_seven_three_plus_two_mul]
  have hA : (2 ^ (3 * t + 5) - 18) / 7 = 2 + 8 * q := by
    apply Nat.div_eq_of_eq_mul_right (by decide : 0 < 7)
    rw [show 2 ^ (3 * t + 5) - 18 = 7 * (2 + 8 * q) by
      rw [show 2 ^ (3 * t + 5) - 18 = 8 * (2 ^ (3 * t + 2)) - 18 by
        rw [show 3 * t + 5 = 3 + (3 * t + 2) by ring, pow_add]
        norm_num]
      have h4 : 4 ≤ 2 ^ (3 * t + 2) := by
        have : 2 ^ 2 ≤ 2 ^ (3 * t + 2) := by
          apply Nat.pow_le_pow_right
          decide
          omega
        simpa using this
      rw [show 8 * 2 ^ (3 * t + 2) - 18 = (8 * 2 ^ (3 * t + 2) - 32) + 14 by omega]
      rw [show 8 * 2 ^ (3 * t + 2) - 32 = 8 * (2 ^ (3 * t + 2) - 4) by omega]
      rw [← hq]
      omega]
  rw [hA]
  -- simplified by the following division identity
  simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc, Nat.mul_comm, Nat.mul_left_comm] using
    (Nat.add_mul_div_right 2 q (by decide : 0 < 8))

lemma q0_seven_mod0 {n : ℕ} (hn : 1 ≤ n) :
    q0 (3 * n) 7 = (2 ^ (W (3 * n)) - 1) / 7 := by
  unfold q0
  rw [z2_bw3_from_sub (k := 3 * n) (y := 7) (c := 2) (by omega) bw_seven (by decide)
    (by simpa [show 2 * (3 * n) + 1 = 6 * n + 1 by ring] using d_seven_mod0 hn), bw_seven]
  rw [show W (3 * n) = 3 * (4 * n) by unfold W; ring]
  exact q0_seven_c2 (4 * n)

lemma q0_seven_mod1 {n : ℕ} (hn : 1 ≤ n) :
    q0 (3 * n + 1) 7 = (2 ^ (W (3 * n + 1)) - 2) / 7 - 1 := by
  unfold q0
  rw [z2_bw3_from_add (k := 3 * n + 1) (y := 7) (c := 6) (by omega) bw_seven (by decide)
    (by simpa [show 2 * (3 * n + 1) + 1 = 6 * n + 3 by ring] using d_seven_mod1 hn), bw_seven]
  rw [show W (3 * n + 1) = 3 * (4 * n + 1) + 1 by unfold W; ring]
  exact q0_seven_c6 (by omega : 1 ≤ 4 * n + 1)

lemma q0_seven_mod2 {n : ℕ} (hn : 1 ≤ n) :
    q0 (3 * n + 2) 7 = (2 ^ (W (3 * n + 2)) - 4) / 7 := by
  unfold q0
  rw [z2_bw3_from_add (k := 3 * n + 2) (y := 7) (c := 3) (by omega) bw_seven (by decide)
    (by simpa [show 2 * (3 * n + 2) + 1 = 6 * n + 5 by ring] using d_seven_mod2 hn), bw_seven]
  rw [show W (3 * n + 2) = 3 * (4 * n + 2) + 2 by unfold W; ring]
  exact q0_seven_c3 (4 * n + 2)

lemma Fk_seven_mod0 {n : ℕ} (hn : 1 ≤ n) :
    Fk (3 * n) 7 = (2 ^ (W (3 * n)) - 1) / 7 := by
  unfold Fk qTilde
  rw [q0_seven_mod0 hn]
  have hclamp : (2 ^ (W (3 * n)) - 1) / 7 ≤ 2 ^ (W (3 * n)) - 1 := by
    exact Nat.div_le_self _ _
  rw [min_eq_left hclamp]
  have hmul : 7 * ((2 ^ (W (3 * n)) - 1) / 7) = 2 ^ (W (3 * n)) - 1 := by
    rw [show W (3 * n) = 3 * (4 * n) by unfold W; ring]
    simpa using u_seven_three_mul (4 * n)
  have hfalse : ¬((2 ^ (W (3 * n)) - 1) / 7) * 7 < 2 ^ (W (3 * n)) - 7 := by
    rw [show ((2 ^ (W (3 * n)) - 1) / 7) * 7 = 2 ^ (W (3 * n)) - 1 by simpa [Nat.mul_comm] using hmul]
    omega
  simp [hfalse]

lemma Fk_seven_mod1 {n : ℕ} (hn : 1 ≤ n) :
    Fk (3 * n + 1) 7 = (2 ^ (W (3 * n + 1)) - 2) / 7 := by
  unfold Fk qTilde
  rw [q0_seven_mod1 hn]
  set q : ℕ := (2 ^ (W (3 * n + 1)) - 2) / 7
  have hclamp : q - 1 ≤ 2 ^ (W (3 * n + 1)) - 1 := by
    have hqle : q ≤ 2 ^ (W (3 * n + 1)) - 2 := by
      exact Nat.div_le_self _ _
    omega
  rw [min_eq_left hclamp]
  have hmul : 7 * q = 2 ^ (W (3 * n + 1)) - 2 := by
    dsimp [q]
    rw [show W (3 * n + 1) = 3 * (4 * n + 1) + 1 by unfold W; ring]
    simpa using u_seven_three_plus_one_mul (4 * n + 1)
  have hq1 : 1 ≤ q := by
    have h8 : 8 ≤ 2 ^ (W (3 * n + 1)) := by
      have hexp : 3 ≤ W (3 * n + 1) := by
        unfold W
        omega
      have : 2 ^ 3 ≤ 2 ^ (W (3 * n + 1)) := by
        exact Nat.pow_le_pow_right (by decide) hexp
      simpa using this
    have h6 : 6 ≤ 7 * q := by
      rw [hmul]
      omega
    have h7 : 7 ≤ 7 * q := by omega
    omega
  have htrue : (q - 1) * 7 < 2 ^ (W (3 * n + 1)) - 7 := by
    rw [show (q - 1) * 7 = 7 * q - 7 by omega]
    rw [hmul]
    omega
  have hqsucc : q - 1 + 1 = q := by omega
  simp [q, htrue, hqsucc]

lemma Fk_seven_mod2 {n : ℕ} (hn : 1 ≤ n) :
    Fk (3 * n + 2) 7 = (2 ^ (W (3 * n + 2)) - 4) / 7 := by
  unfold Fk qTilde
  rw [q0_seven_mod2 hn]
  have hclamp : (2 ^ (W (3 * n + 2)) - 4) / 7 ≤ 2 ^ (W (3 * n + 2)) - 1 := by
    exact le_trans (Nat.div_le_self _ _) (by omega)
  rw [min_eq_left hclamp]
  have hmul : 7 * ((2 ^ (W (3 * n + 2)) - 4) / 7) = 2 ^ (W (3 * n + 2)) - 4 := by
    rw [show W (3 * n + 2) = 3 * (4 * n + 2) + 2 by unfold W; ring]
    simpa using u_seven_three_plus_two_mul (4 * n + 2)
  have hfalse : ¬((2 ^ (W (3 * n + 2)) - 4) / 7) * 7 < 2 ^ (W (3 * n + 2)) - 7 := by
    rw [show ((2 ^ (W (3 * n + 2)) - 4) / 7) * 7 = 2 ^ (W (3 * n + 2)) - 4 by simpa [Nat.mul_comm] using hmul]
    omega
  simp [hfalse]

lemma Fk_seven {k : ℕ} (hk : 3 ≤ k) : Fk k 7 = 2 ^ (W k) / 7 := by
  have hmod : k % 3 < 3 := Nat.mod_lt _ (by decide)
  interval_cases hkmod : k % 3
  · have hk0 : k = 3 * (k / 3) := by
      simpa [hkmod, Nat.mul_comm] using (Nat.div_add_mod k 3).symm
    rw [hk0, Fk_seven_mod0 (by omega)]
    rw [show W (3 * (k / 3)) = 3 * (4 * (k / 3)) by unfold W; ring]
    have hq : 7 * ((2 ^ (3 * (4 * (k / 3))) - 1) / 7) = 2 ^ (3 * (4 * (k / 3))) - 1 := by
      simpa using u_seven_three_mul (4 * (k / 3))
    apply Nat.eq_of_mul_eq_mul_left (by decide : 0 < 7)
    rw [hq]
    have hdecomp := Nat.mod_add_div (2 ^ (3 * (4 * (k / 3)))) 7
    rw [pow_two_three_mod_seven] at hdecomp
    omega
  · have hk1 : k = 3 * (k / 3) + 1 := by
      simpa [hkmod, Nat.mul_comm] using (Nat.div_add_mod k 3).symm
    rw [hk1, Fk_seven_mod1 (by omega)]
    rw [show W (3 * (k / 3) + 1) = 3 * (4 * (k / 3) + 1) + 1 by unfold W; ring]
    apply Nat.eq_of_mul_eq_mul_left (by decide : 0 < 7)
    have hq : 7 * ((2 ^ (3 * (4 * (k / 3) + 1) + 1) - 2) / 7) = 2 ^ (3 * (4 * (k / 3) + 1) + 1) - 2 := by
      simpa using u_seven_three_plus_one_mul (4 * (k / 3) + 1)
    rw [hq]
    have hdecomp := Nat.mod_add_div (2 ^ (3 * (4 * (k / 3) + 1) + 1)) 7
    rw [pow_two_three_plus_one_mod_seven] at hdecomp
    omega
  · have hk2 : k = 3 * (k / 3) + 2 := by
      simpa [hkmod, Nat.mul_comm] using (Nat.div_add_mod k 3).symm
    rw [hk2, Fk_seven_mod2 (by omega)]
    rw [show W (3 * (k / 3) + 2) = 3 * (4 * (k / 3) + 2) + 2 by unfold W; ring]
    apply Nat.eq_of_mul_eq_mul_left (by decide : 0 < 7)
    have hq : 7 * ((2 ^ (3 * (4 * (k / 3) + 2) + 2) - 4) / 7) = 2 ^ (3 * (4 * (k / 3) + 2) + 2) - 4 := by
      simpa using u_seven_three_plus_two_mul (4 * (k / 3) + 2)
    rw [hq]
    have hdecomp := Nat.mod_add_div (2 ^ (3 * (4 * (k / 3) + 2) + 2)) 7
    rw [pow_two_three_plus_two_mod_seven] at hdecomp
    omega

lemma floor_mul_bounds {n y : ℕ} (hy : 0 < y) : n - y ≤ y * (n / y) ∧ y * (n / y) ≤ n := by
  constructor
  · have hdecomp := Nat.mod_add_div n y
    have hmod : n % y < y := Nat.mod_lt _ hy
    omega
  · exact Nat.mul_div_le _ _

theorem Good_seven {k : ℕ} (hk : 3 ≤ k) : Good k 7 := by
  unfold Good
  rw [Fk_seven hk]
  exact floor_mul_bounds (n := 2 ^ (W k)) (y := 7) (by decide)

end Seven



open Reciprocal

namespace Reciprocal

/--
TeX correspondence: `prop:tiny`.

The TeX packages the tiny inputs through the periodicity lemma `lem:tiny-period`.
Here the same result is assembled from explicit symbolic proofs for
`y = 3, 5, 6, 7`.
-/
theorem tiny_proved :
    ∀ {k y : ℕ},
      3 ≤ k →
      1 ≤ y →
      y ≤ 7 →
      y ≠ 3 →
      y ≠ 6 →
      (¬ ∃ m : ℕ, y = 2 ^ m) →
      Good k y := by
  intro k y hk hy1 hy7 hy3 hy6 hpow
  interval_cases y
  · exfalso
    exact hpow ⟨0, by norm_num⟩
  · exfalso
    exact hpow ⟨1, by norm_num⟩
  · exact False.elim (hy3 rfl)
  · exfalso
    exact hpow ⟨2, by norm_num⟩
  · exact Five.Good_five hk
  · exact False.elim (hy6 rfl)
  · exact Seven.Good_seven hk

end Reciprocal



open Reciprocal

namespace ReciprocalLowExactCore

lemma z0_low {k y : ℕ} (hyk : bw y ≤ k) :
    z0 k y = 2 ^ (k + bw y - 1) / y := by
  unfold z0
  rw [g_low hyk]
  rw [Nat.mul_comm y _, ← Nat.div_div_eq_div_mul]
  have hpow : 2 ^ (2 * k - 1) / 2 ^ (k - bw y) = 2 ^ (k + bw y - 1) := by
    apply Nat.div_eq_of_eq_mul_right (by positivity : 0 < 2 ^ (k - bw y))
    rw [show 2 * k - 1 = (k + bw y - 1) + (k - bw y) by omega, pow_add, Nat.mul_comm]
  rw [hpow]

lemma z1_low {k y : ℕ} (hyk : bw y ≤ k) :
    z1 k y = 2 ^ k * z0 k y - (y * (z0 k y) ^ 2) / 2 ^ (bw y) := by
  unfold z1
  rw [h_low hyk]
  simp only
  have hinner : (y * 2 ^ (2 * k - bw y) * z0 k y) / 2 ^ k = y * z0 k y * 2 ^ (k - bw y) := by
    apply Nat.div_eq_of_eq_mul_right (by positivity : 0 < 2 ^ k)
    rw [show 2 * k - bw y = k + (k - bw y) by omega, pow_add]
    ac_rfl
  rw [hinner]
  have hsecond :
      (z0 k y * (y * z0 k y * 2 ^ (k - bw y))) / 2 ^ k =
        (y * (z0 k y) ^ 2) / 2 ^ (bw y) := by
    rw [show z0 k y * (y * z0 k y * 2 ^ (k - bw y)) = (y * (z0 k y) ^ 2) * 2 ^ (k - bw y) by
      rw [pow_two]
      ac_rfl]
    have hpowk : 2 ^ k = 2 ^ (k - bw y) * 2 ^ (bw y) := by
      calc
        2 ^ k = 2 ^ (k - bw y + bw y) := by rw [Nat.sub_add_cancel hyk]
        _ = 2 ^ (k - bw y) * 2 ^ (bw y) := by rw [pow_add]
    rw [hpowk]
    rw [← Nat.div_div_eq_div_mul]
    have htmp :
        (y * (z0 k y) ^ 2 * 2 ^ (k - bw y)) / 2 ^ (k - bw y) = y * (z0 k y) ^ 2 := by
      apply Nat.div_eq_of_eq_mul_right (by positivity : 0 < 2 ^ (k - bw y))
      ac_rfl
    exact congrArg (fun t => t / 2 ^ (bw y)) htmp
  rw [hsecond]

lemma z2_low_core {k y : ℕ} (hyk : bw y ≤ k) (hb4 : 4 ≤ bw y) :
    z2 k y = 2 ^ (2 * k + 3) * z1 k y - (y * (z1 k y) ^ 2) / 2 ^ (bw y - 4) := by
  unfold z2
  rw [x_low hyk]
  simp only
  have hinner :
      (y * 2 ^ (2 * k + (2 * k - bw y)) * z1 k y) / 2 ^ (2 * k - 4) =
        y * z1 k y * 2 ^ (2 * k + 4 - bw y) := by
    apply Nat.div_eq_of_eq_mul_right (by positivity : 0 < 2 ^ (2 * k - 4))
    rw [show 2 * k + (2 * k - bw y) = (2 * k - 4) + (2 * k + 4 - bw y) by omega, pow_add]
    ac_rfl
  rw [hinner]
  have hsecond :
      (z1 k y * (y * z1 k y * 2 ^ (2 * k + 4 - bw y))) / 2 ^ (2 * k) =
        (y * (z1 k y) ^ 2) / 2 ^ (bw y - 4) := by
    rw [show z1 k y * (y * z1 k y * 2 ^ (2 * k + 4 - bw y)) =
      (y * (z1 k y) ^ 2) * 2 ^ (2 * k + 4 - bw y) by
      rw [pow_two]
      ac_rfl]
    have hpowk : 2 ^ (2 * k) = 2 ^ (2 * k + 4 - bw y) * 2 ^ (bw y - 4) := by
      have hdecomp : 2 * k + 4 - bw y + (bw y - 4) = 2 * k := by omega
      calc
        2 ^ (2 * k) = 2 ^ (2 * k + 4 - bw y + (bw y - 4)) := by rw [hdecomp]
        _ = 2 ^ (2 * k + 4 - bw y) * 2 ^ (bw y - 4) := by rw [pow_add]
    rw [hpowk]
    rw [← Nat.div_div_eq_div_mul]
    have htmp :
        (y * (z1 k y) ^ 2 * 2 ^ (2 * k + 4 - bw y)) / 2 ^ (2 * k + 4 - bw y) =
          y * (z1 k y) ^ 2 := by
      apply Nat.div_eq_of_eq_mul_right (by positivity : 0 < 2 ^ (2 * k + 4 - bw y))
      ac_rfl
    exact congrArg (fun t => t / 2 ^ (bw y - 4)) htmp
  rw [hsecond]

lemma eq_two_pow_of_dvd_pow_two {y n : ℕ} (h : y ∣ 2 ^ n) : ∃ m ≤ n, y = 2 ^ m := by
  simpa using (Nat.dvd_prime_pow Nat.prime_two).mp h

lemma pow_mod_pos_of_not_two_pow {y n : ℕ} (hy : 1 < y) (hpow : ¬ ∃ m : ℕ, y = 2 ^ m) :
    0 < 2 ^ n % y := by
  have hlt : 2 ^ n % y < y := Nat.mod_lt _ (by omega)
  by_cases hzero : 2 ^ n % y = 0
  · have hdvd : y ∣ 2 ^ n := by
      refine ⟨2 ^ n / y, ?_⟩
      have hdecomp := Nat.mod_add_div (2 ^ n) y
      rw [hzero] at hdecomp
      simpa [Nat.mul_comm, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using hdecomp.symm
    rcases eq_two_pow_of_dvd_pow_two hdvd with ⟨m, hmle, hyeq⟩
    exact False.elim (hpow ⟨m, hyeq⟩)
  · omega

lemma floor_mul_bounds {n y : ℕ} (hy : 0 < y) :
    n - y ≤ y * (n / y) ∧ y * (n / y) ≤ n := by
  constructor
  · have hdecomp := Nat.mod_add_div n y
    have hmod : n % y < y := Nat.mod_lt _ hy
    omega
  · exact Nat.mul_div_le _ _

lemma q0_eq_floor_implies_Good {k y : ℕ} (hy : 1 < y) (hq0 : q0 k y = 2 ^ (W k) / y) :
    Good k y := by
  have hF : Fk k y = 2 ^ (W k) / y := by
    have hclamp : 2 ^ (W k) / y ≤ 2 ^ (W k) - 1 := by
      have hlt : 2 ^ (W k) / y < 2 ^ (W k) := by
        exact Nat.div_lt_self (by positivity) hy
      omega
    have hfalse : ¬(2 ^ (W k) / y) * y < 2 ^ (W k) - y := by
      have hlower := (floor_mul_bounds (n := 2 ^ (W k)) (y := y) (by omega)).1
      have hlower' : 2 ^ (W k) - y ≤ (2 ^ (W k) / y) * y := by
        simpa [Nat.mul_comm] using hlower
      exact Nat.not_lt_of_ge hlower'
    unfold Fk qTilde
    rw [hq0, min_eq_left hclamp]
    simp [hfalse]
  unfold Good
  rw [hF]
  exact floor_mul_bounds (n := 2 ^ (W k)) (y := y) (by omega)

lemma q0_eq_floor_pred_implies_Good {k y : ℕ}
    (hy : 1 < y)
    (hylt : y < 2 ^ (W k))
    (hpow : ¬ ∃ m : ℕ, y = 2 ^ m)
    (hq0 : q0 k y = 2 ^ (W k) / y - 1) :
    Good k y := by
  have hF : Fk k y = 2 ^ (W k) / y := by
    have hclamp : 2 ^ (W k) / y - 1 ≤ 2 ^ (W k) - 1 := by
      have hlt : 2 ^ (W k) / y < 2 ^ (W k) := by
        exact Nat.div_lt_self (by positivity) hy
      omega
    have hspos : 0 < 2 ^ (W k) % y := pow_mod_pos_of_not_two_pow hy hpow
    have hqpos : 1 ≤ 2 ^ (W k) / y := by
      exact Nat.succ_le_of_lt (Nat.div_pos (Nat.le_of_lt hylt) (by omega))
    have hquot_lt : y * (2 ^ (W k) / y) < 2 ^ (W k) := by
      have hmod := Nat.mod_add_div (2 ^ (W k)) y
      omega
    have hpredmul : (2 ^ (W k) / y - 1) * y = y * (2 ^ (W k) / y) - y := by
      calc
        (2 ^ (W k) / y - 1) * y = (2 ^ (W k) / y).pred * y := by
          rw [Nat.pred_eq_sub_one]
        _ = (2 ^ (W k) / y) * y - y := by
          rw [Nat.pred_mul, Nat.mul_comm]
        _ = y * (2 ^ (W k) / y) - y := by
          rw [Nat.mul_comm]
    have htrue : (2 ^ (W k) / y - 1) * y < 2 ^ (W k) - y := by
      rw [hpredmul]
      omega
    have hqsucc : 2 ^ (W k) / y - 1 + 1 = 2 ^ (W k) / y := Nat.sub_add_cancel hqpos
    unfold Fk qTilde
    rw [hq0, min_eq_left hclamp]
    simp [htrue, hqsucc]
  unfold Good
  rw [hF]
  exact floor_mul_bounds (n := 2 ^ (W k)) (y := y) (by omega)

end ReciprocalLowExactCore



open Reciprocal
open ReciprocalLowExactCore

namespace ReciprocalLowExact

/--
TeX correspondence: `prop:exact-low` and `thm:low-exact`.

This namespace formalizes the exact low-bit arithmetic, the `d`-bounds, and the
resulting `q0 ∈ {Q - 1, Q}` argument.
-/
lemma sub_div_eq_ceilDiv {a m n : ℕ} (hn : 0 < n) (hm : m ≤ a * n) :
    a - m / n = (a * n - m) ⌈/⌉ n := by
  set q : ℕ := m / n
  have hqle : q ≤ a := by
    dsimp [q]
    exact (Nat.div_le_iff_le_mul hn).2 (le_trans hm (by omega))
  have hupper : (a * n - m) ⌈/⌉ n ≤ a - q := by
    refine (ceilDiv_le_iff_le_mul hn).2 ?_
    have hmuldiv : q * n ≤ m := by
      dsimp [q]
      simpa [Nat.mul_comm] using (Nat.mul_div_le m n)
    have hsuble : a * n - m ≤ a * n - q * n := Nat.sub_le_sub_left hmuldiv _
    refine hsuble.trans_eq ?_
    rw [Nat.mul_comm a n, Nat.mul_comm q n, Nat.mul_sub_left_distrib]
  by_cases hx0 : a - q = 0
  · have hnonneg : 0 ≤ (a * n - m) ⌈/⌉ n := Nat.zero_le _
    omega
  · have hm_lt : m < (q + 1) * n := by
      have hqsucc : m / n < q + 1 := by
        dsimp [q]
        exact Nat.lt_succ_self _
      exact (Nat.div_lt_iff_lt_mul hn).1 hqsucc
    have hlower : a - q ≤ (a * n - m) ⌈/⌉ n := by
      by_contra hlt
      have hlepred : (a * n - m) ⌈/⌉ n ≤ a - q - 1 := by omega
      have hbad : a * n - m ≤ (a - q - 1) * n := by
        simpa [Nat.mul_comm] using (ceilDiv_le_iff_le_mul hn).1 hlepred
      have hbad' : a * n ≤ m + (a - q - 1) * n := by
        exact (Nat.sub_le_iff_le_add').1 hbad
      have hsplit : a * n = (a - q - 1) * n + (q + 1) * n := by
        calc
          a * n = ((a - q - 1) + (q + 1)) * n := by congr 1; omega
          _ = (a - q - 1) * n + (q + 1) * n := by rw [Nat.add_mul]
      rw [hsplit] at hbad'
      have hmul : (q + 1) * n ≤ m := by
        omega
      exact (Nat.not_le_of_gt hm_lt) hmul
    exact le_antisymm hlower hupper

lemma mul_ceilDiv_lt_add {a b : ℕ} (ha : 0 < a) : a * (b ⌈/⌉ a) < b + a := by
  by_cases hz : b ⌈/⌉ a = 0
  · rw [hz]
    have hba : 0 < b + a := Nat.lt_of_lt_of_le ha (Nat.le_add_left _ _)
    simpa using hba
  · have hzpos : 0 < b ⌈/⌉ a := Nat.pos_of_ne_zero hz
    have hnot : ¬ b ≤ a * (b ⌈/⌉ a - 1) := by
      intro hb
      have hle : b ⌈/⌉ a ≤ b ⌈/⌉ a - 1 := by
        exact (ceilDiv_le_iff_le_mul ha).2 hb
      omega
    have hlt : a * (b ⌈/⌉ a - 1) < b := lt_of_not_ge hnot
    have hsplit : a * (b ⌈/⌉ a) = a * (b ⌈/⌉ a - 1) + a := by
      calc
        a * (b ⌈/⌉ a) = a * ((b ⌈/⌉ a - 1) + 1) := by congr 1; omega
        _ = a * (b ⌈/⌉ a - 1) + a := by rw [Nat.mul_add, Nat.mul_one]
    rw [hsplit]
    omega

lemma lt_ceilDiv_of_mul_lt {a b c : ℕ} (ha : 0 < a) (h : a * c < b) : c < b ⌈/⌉ a := by
  by_contra hnot
  have hle : b ⌈/⌉ a ≤ c := Nat.not_lt.mp hnot
  have hb : b ≤ a * c := (ceilDiv_le_iff_le_mul ha).1 hle
  exact (Nat.not_le_of_gt h) hb

lemma z1_low_ceilDiv {k y : ℕ} (hy : 0 < y) (hyk : bw y ≤ k) :
    z1 k y = (((2 ^ (k + bw y - 1)) ^ 2 - (2 ^ (k + bw y - 1) % y) ^ 2) / y) ⌈/⌉ 2 ^ (bw y) := by
  set ell : ℕ := bw y
  set A : ℕ := 2 ^ (k + ell - 1)
  set r : ℕ := A % y
  have hellpos : 0 < ell := by
    simpa [ell] using bw_pos y
  have hz0 : z0 k y = A / y := by
    dsimp [A, ell]
    simpa using z0_low (k := k) (y := y) hyk
  have hA : A = (A / y) * y + r := by
    dsimp [A, r]
    exact (Nat.div_add_mod' _ _).symm
  have hA' : A = z0 k y * y + r := by
    simpa [hz0, Nat.mul_comm] using hA
  have hAzy : z0 k y * y = A - r := by
    exact Nat.eq_sub_of_add_eq' (by simpa [add_comm] using hA'.symm)
  have hz0bound : y * z0 k y ≤ 2 ^ (k + ell) := by
    have hdouble' : 2 ^ (k + ell) = 2 ^ (k + ell - 1) + 2 ^ (k + ell - 1) := by
      have hsumpos : 1 ≤ k + ell := by
        omega
      have hkell : k + ell = (k + ell - 1) + 1 := by
        exact (Nat.sub_eq_iff_eq_add hsumpos).1 rfl
      calc
        2 ^ (k + ell) = 2 ^ ((k + ell - 1) + 1) := by
          conv_lhs => rw [hkell]
        _ = 2 ^ (k + ell - 1) * 2 ^ 1 := by rw [pow_add]
        _ = 2 ^ (k + ell - 1) * 2 := by norm_num
        _ = 2 ^ (k + ell - 1) + 2 ^ (k + ell - 1) := by
          rw [Nat.mul_comm, two_mul]
    have hdouble : 2 ^ (k + ell) = A + A := by
      simpa [A] using hdouble'
    have hposA : 0 < A := by
      dsimp [A]
      positivity
    have hltA : A < 2 ^ (k + ell) := by
      calc
        A < A + A := Nat.lt_add_of_pos_right hposA
        _ = 2 ^ (k + ell) := hdouble.symm
    have hEq : y * z0 k y = A - r := by
      simpa [Nat.mul_comm] using hAzy
    rw [hEq]
    exact le_of_lt (lt_of_le_of_lt (Nat.sub_le _ _) hltA)
  have hm : y * (z0 k y) ^ 2 ≤ (2 ^ k * z0 k y) * 2 ^ ell := by
    calc
      y * (z0 k y) ^ 2 = z0 k y * (y * z0 k y) := by
        rw [pow_two]
        ac_rfl
      _ ≤ z0 k y * 2 ^ (k + ell) := Nat.mul_le_mul_left _ hz0bound
      _ = (2 ^ k * z0 k y) * 2 ^ ell := by
        rw [pow_add]
        ac_rfl
  rw [z1_low hyk, sub_div_eq_ceilDiv (hn := by positivity) hm]
  have hnum : ((2 ^ k * z0 k y) * 2 ^ ell - y * (z0 k y) ^ 2) = (A ^ 2 - r ^ 2) / y := by
    symm
    apply Nat.div_eq_of_eq_mul_right hy
    have hyz0 : y * z0 k y = A - r := by
      simpa [Nat.mul_comm] using hAzy
    have hAr : 2 ^ (k + ell) - y * z0 k y = A + r := by
      have hdouble' : 2 ^ (k + ell) = 2 ^ (k + ell - 1) + 2 ^ (k + ell - 1) := by
        have hsumpos : 1 ≤ k + ell := by
          omega
        have hkell : k + ell = (k + ell - 1) + 1 := by
          exact (Nat.sub_eq_iff_eq_add hsumpos).1 rfl
        calc
          2 ^ (k + ell) = 2 ^ ((k + ell - 1) + 1) := by
            conv_lhs => rw [hkell]
          _ = 2 ^ (k + ell - 1) * 2 ^ 1 := by rw [pow_add]
          _ = 2 ^ (k + ell - 1) * 2 := by norm_num
          _ = 2 ^ (k + ell - 1) + 2 ^ (k + ell - 1) := by
            rw [Nat.mul_comm, two_mul]
      have hdouble : 2 ^ (k + ell) = A + A := by
        simpa [A] using hdouble'
      rw [hyz0, hdouble]
      omega
    have hnum_mul : A ^ 2 - r ^ 2 = y * ((2 ^ k * z0 k y) * 2 ^ ell - y * (z0 k y) ^ 2) := by
      calc
        A ^ 2 - r ^ 2 = (A - r) * (A + r) := by
          simp [pow_two, Nat.mul_self_sub_mul_self_eq, Nat.mul_comm]
        _ = (y * z0 k y) * (A + r) := by rw [hyz0]
        _ = (y * z0 k y) * (2 ^ (k + ell) - y * z0 k y) := by
          rw [hAr]
        _ = (y * z0 k y) * 2 ^ (k + ell) - (y * z0 k y) * (y * z0 k y) := by
          rw [Nat.mul_sub_left_distrib]
        _ = y * ((2 ^ k * z0 k y) * 2 ^ ell) - y * (y * (z0 k y) ^ 2) := by
          rw [pow_two, pow_add]
          ac_rfl
        _ = y * ((2 ^ k * z0 k y) * 2 ^ ell - y * (z0 k y) ^ 2) := by
          rw [Nat.mul_sub_left_distrib]
    exact hnum_mul
  rw [hnum]

lemma z1_low_dist_lt {k y : ℕ} (hy : 0 < y) (hyk : bw y ≤ k) :
    Nat.dist (2 ^ (2 * k + bw y - 2)) (y * z1 k y) < y := by
  set ell : ℕ := bw y
  set A : ℕ := 2 ^ (k + ell - 1)
  set r : ℕ := A % y
  set N : ℕ := ((A ^ 2 - r ^ 2) / y)
  have hz1 : z1 k y = N ⌈/⌉ 2 ^ ell := by
    simpa [N, A, r, ell] using (z1_low_ceilDiv (k := k) (y := y) hy hyk)
  have hNle : N ≤ 2 ^ ell * z1 k y := by
    apply (ceilDiv_le_iff_le_mul (by positivity : 0 < 2 ^ ell)).mp
    rw [hz1]
  have hNlt : 2 ^ ell * z1 k y < N + 2 ^ ell := by
    simpa [hz1, Nat.mul_comm, Nat.add_comm] using
      (mul_ceilDiv_lt_add (a := 2 ^ ell) (b := N) (by positivity))
  have hNy : N * y = 2 ^ ell * 2 ^ (2 * k + ell - 2) - r ^ 2 := by
    have hA : A = (A / y) * y + r := by
      dsimp [A, r]
      exact (Nat.div_add_mod' _ _).symm
    have hAzy : (A / y) * y = A - r := by
      exact Nat.eq_sub_of_add_eq' (by simpa [add_comm] using hA.symm)
    have hdiv : y ∣ A ^ 2 - r ^ 2 := by
      refine ⟨(A + r) * (A / y), ?_⟩
      calc
        A ^ 2 - r ^ 2 = (A + r) * (A - r) := by
          simp [pow_two, Nat.mul_self_sub_mul_self_eq]
        _ = (A + r) * ((A / y) * y) := by rw [hAzy]
        _ = y * ((A + r) * (A / y)) := by
          ac_rfl
    have hNy0 : N * y = A ^ 2 - r ^ 2 := by
      dsimp [N]
      exact Nat.div_mul_cancel hdiv
    dsimp [N, A]
    rw [hNy0, pow_two]
    have hexp : (k + ell - 1) + (k + ell - 1) = ell + (2 * k + ell - 2) := by
      omega
    rw [show 2 ^ (k + ell - 1) * 2 ^ (k + ell - 1) = 2 ^ ell * 2 ^ (2 * k + ell - 2) by
      rw [← pow_add, hexp, pow_add]]
  have hrlt_y : r < y := by
    dsimp [r]
    exact Nat.mod_lt _ hy
  have hylt_pow : y < 2 ^ ell := by
    dsimp [ell]
    exact (bw_bounds (Nat.ne_of_gt hy)).2
  have hr2lt : r ^ 2 < y * 2 ^ ell := by
    have hrlt_pow : r < 2 ^ ell := lt_trans hrlt_y hylt_pow
    cases Nat.eq_zero_or_pos r with
    | inl hr0 =>
        rw [hr0, pow_two]
        positivity
    | inr hrpos =>
        have h1 : r * r < y * r := Nat.mul_lt_mul_of_pos_right hrlt_y hrpos
        have h2 : y * r < y * 2 ^ ell := Nat.mul_lt_mul_of_pos_left hrlt_pow hy
        simpa [pow_two] using lt_trans h1 h2
  by_cases hcase : y * z1 k y ≤ 2 ^ (2 * k + ell - 2)
  · have hd : (2 ^ (2 * k + ell - 2) - y * z1 k y) * 2 ^ ell ≤ r ^ 2 := by
      have hmain : 2 ^ ell * (2 ^ (2 * k + ell - 2) - y * z1 k y)
          = 2 ^ ell * 2 ^ (2 * k + ell - 2) - y * (2 ^ ell * z1 k y) := by
            rw [Nat.mul_sub_left_distrib]
            ac_rfl
      have hNle' : 2 ^ ell * 2 ^ (2 * k + ell - 2) - r ^ 2 ≤ y * (2 ^ ell * z1 k y) := by
        rw [← hNy]
        simpa [Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using Nat.mul_le_mul_right y hNle
      rw [Nat.mul_comm, hmain]
      omega
    have hdist : Nat.dist (2 ^ (2 * k + ell - 2)) (y * z1 k y) = 2 ^ (2 * k + ell - 2) - y * z1 k y := by
      rw [Nat.dist_comm, Nat.dist_eq_sub_of_le hcase]
    have hdistmul : Nat.dist (2 ^ (2 * k + ell - 2)) (y * z1 k y) * 2 ^ ell < y * 2 ^ ell := by
      rw [hdist]
      exact lt_of_le_of_lt hd hr2lt
    exact Nat.lt_of_mul_lt_mul_right hdistmul
  · have hcase' : 2 ^ (2 * k + ell - 2) < y * z1 k y := lt_of_not_ge hcase
    have hd : (y * z1 k y - 2 ^ (2 * k + ell - 2)) * 2 ^ ell < y * 2 ^ ell := by
      have hmain : y * (2 ^ ell * z1 k y)
          = 2 ^ ell * 2 ^ (2 * k + ell - 2) + (y * z1 k y - 2 ^ (2 * k + ell - 2)) * 2 ^ ell := by
            have hsplit : y * z1 k y = 2 ^ (2 * k + ell - 2) + (y * z1 k y - 2 ^ (2 * k + ell - 2)) := by
              exact (Nat.add_sub_of_le (le_of_lt hcase')).symm
            calc
              y * (2 ^ ell * z1 k y) = (y * z1 k y) * 2 ^ ell := by ac_rfl
              _ = (2 ^ (2 * k + ell - 2) + (y * z1 k y - 2 ^ (2 * k + ell - 2))) * 2 ^ ell := by
                conv_lhs => rw [hsplit]
              _ = 2 ^ ell * 2 ^ (2 * k + ell - 2) + (y * z1 k y - 2 ^ (2 * k + ell - 2)) * 2 ^ ell := by
                rw [Nat.add_mul]
                ac_rfl
      have hNlt' : y * (2 ^ ell * z1 k y) < (2 ^ ell * 2 ^ (2 * k + ell - 2) - r ^ 2) + y * 2 ^ ell := by
        rw [← hNy]
        simpa [Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc, Nat.mul_add, Nat.add_comm, Nat.add_left_comm] using
          Nat.mul_lt_mul_of_pos_right hNlt hy
      rw [hmain] at hNlt'
      omega
    have hdist : Nat.dist (2 ^ (2 * k + ell - 2)) (y * z1 k y) = y * z1 k y - 2 ^ (2 * k + ell - 2) := by
      rw [Nat.dist_eq_sub_of_le (le_of_lt hcase')]
    have hdistmul : Nat.dist (2 ^ (2 * k + ell - 2)) (y * z1 k y) * 2 ^ ell < y * 2 ^ ell := by
      rw [hdist]
      exact hd
    exact Nat.lt_of_mul_lt_mul_right hdistmul

lemma yz1_low_lt_double {k y : ℕ} (hy : 1 < y) (hyk : bw y ≤ k) :
    y * z1 k y < 2 ^ (2 * k + bw y - 1) := by
  set ell : ℕ := bw y
  set S : ℕ := 2 ^ (2 * k + ell - 2)
  set T : ℕ := y * z1 k y
  have hy0 : 0 < y := by omega
  have hdistlt : Nat.dist S T < y := by
    simpa [S, T, ell] using (z1_low_dist_lt (k := k) (y := y) hy0 hyk)
  have hyltpow : y < 2 ^ ell := by
    simpa [ell] using (bw_bounds (Nat.ne_of_gt hy0)).2
  have hSge : 2 ^ ell ≤ S := by
    dsimp [S]
    rw [show 2 * k + ell - 2 = ell + (2 * k - 2) by omega, pow_add]
    simpa [Nat.mul_comm] using
      (Nat.le_mul_of_pos_right (2 ^ ell) (show 0 < 2 ^ (2 * k - 2) by positivity))
  have hyLeS : y ≤ S := le_trans (le_of_lt hyltpow) hSge
  have hpow : 2 * S = 2 ^ (2 * k + ell - 1) := by
    have hsumpos : 1 ≤ 2 * k + ell - 1 := by
      have hellpos : 0 < ell := by
        simpa [ell] using bw_pos y
      omega
    have hexp : 2 * k + ell - 1 = (2 * k + ell - 2) + 1 := by
      exact (Nat.sub_eq_iff_eq_add hsumpos).1 rfl
    dsimp [S]
    rw [hexp, pow_add]
    norm_num
    rw [Nat.mul_comm]
  by_cases hTS : T ≤ S
  · have hSpos : 0 < S := lt_of_lt_of_le hy0 hyLeS
    have hlt2S : S < 2 * S := by
      rw [two_mul]
      exact Nat.lt_add_of_pos_right hSpos
    calc
      T < 2 * S := lt_of_le_of_lt hTS hlt2S
      _ = 2 ^ (2 * k + ell - 1) := hpow
  · have hST : S < T := lt_of_not_ge hTS
    have hgap : T - S < S := by
      have hdist : Nat.dist S T = T - S := by
        rw [Nat.dist_eq_sub_of_le (le_of_lt hST)]
      rw [← hdist]
      exact lt_of_lt_of_le hdistlt hyLeS
    have hsplit : T = S + (T - S) := (Nat.add_sub_of_le (le_of_lt hST)).symm
    calc
      T = S + (T - S) := hsplit
      _ < S + S := Nat.add_lt_add_left hgap S
      _ = 2 * S := by rw [two_mul]
      _ = 2 ^ (2 * k + ell - 1) := hpow

lemma z2_low_num_eq {k y : ℕ} (hy : 1 < y) (hyk : bw y ≤ k) (hb4 : 4 ≤ bw y) :
    2 ^ (W k + 2 * bw y - 4) - (Nat.dist (2 ^ (2 * k + bw y - 2)) (y * z1 k y)) ^ 2 =
      y * (((2 ^ (2 * k + 3) * z1 k y) * 2 ^ (bw y - 4)) - y * (z1 k y) ^ 2) := by
  set ell : ℕ := bw y
  set S : ℕ := 2 ^ (2 * k + ell - 2)
  set D : ℕ := Nat.dist S (y * z1 k y)
  set T : ℕ := y * z1 k y
  have hTlt : T < 2 ^ (2 * k + ell - 1) := by
    simpa [T, S, ell] using (yz1_low_lt_double (k := k) (y := y) hy hyk)
  have hTlt2S : T < 2 * S := by
    have hpow : 2 * S = 2 ^ (2 * k + ell - 1) := by
      dsimp [S]
      rw [show 2 * k + ell - 1 = (2 * k + ell - 2) + 1 by omega, pow_add]
      norm_num
      rw [Nat.mul_comm]
    simpa [hpow] using hTlt
  have hSsq : S ^ 2 = 2 ^ (W k + 2 * ell - 4) := by
    dsimp [S]
    rw [pow_two, ← pow_add]
    unfold W
    congr 1
    omega
  have hpow2 : 2 ^ (2 * k + ell - 1) = 2 ^ (2 * k + 3) * 2 ^ (ell - 4) := by
    rw [show 2 * k + ell - 1 = (2 * k + 3) + (ell - 4) by omega, pow_add]
  have hprod :
      y * (((2 ^ (2 * k + 3) * z1 k y) * 2 ^ (ell - 4)) - y * (z1 k y) ^ 2) = T * (2 * S - T) := by
    calc
      y * (((2 ^ (2 * k + 3) * z1 k y) * 2 ^ (ell - 4)) - y * (z1 k y) ^ 2) =
          y * (z1 k y * (2 ^ (2 * k + ell - 1) - y * z1 k y)) := by
            rw [show ((2 ^ (2 * k + 3) * z1 k y) * 2 ^ (ell - 4)) = z1 k y * 2 ^ (2 * k + ell - 1) by
              rw [hpow2]
              ac_rfl]
            rw [pow_two]
            rw [show z1 k y * 2 ^ (2 * k + ell - 1) - y * (z1 k y * z1 k y) =
              z1 k y * (2 ^ (2 * k + ell - 1) - y * z1 k y) by
              rw [Nat.mul_sub_left_distrib]
              ac_rfl]
      _ = T * (2 * S - T) := by
            dsimp [T, S]
            rw [show 2 ^ (2 * k + ell - 1) = 2 * 2 ^ (2 * k + ell - 2) by
              rw [show 2 * k + ell - 1 = (2 * k + ell - 2) + 1 by omega, pow_add]
              norm_num
              rw [Nat.mul_comm]]
            ac_rfl
  by_cases hTS : T ≤ S
  · have hD : D = S - T := by
      dsimp [D]
      rw [Nat.dist_comm, Nat.dist_eq_sub_of_le hTS]
    have hsum : S + (S - T) = 2 * S - T := by
      omega
    calc
      2 ^ (W k + 2 * ell - 4) - D ^ 2 = S ^ 2 - (S - T) ^ 2 := by
        rw [hD, hSsq]
      _ = (S - (S - T)) * (S + (S - T)) := by
        simp [pow_two, Nat.mul_self_sub_mul_self_eq, Nat.mul_comm]
      _ = T * (2 * S - T) := by
        have hsub : S - (S - T) = T := by omega
        rw [hsub, hsum]
      _ = y * (((2 ^ (2 * k + 3) * z1 k y) * 2 ^ (ell - 4)) - y * (z1 k y) ^ 2) := hprod.symm
  · have hST : S < T := lt_of_not_ge hTS
    have hD : D = T - S := by
      dsimp [D]
      rw [Nat.dist_eq_sub_of_le (le_of_lt hST)]
    have hsplit : T = S + (T - S) := (Nat.add_sub_of_le (le_of_lt hST)).symm
    have hsum : S + (T - S) = T := Nat.add_sub_of_le (le_of_lt hST)
    have hsub : S - (T - S) = 2 * S - T := by
      rw [hsplit] at hTlt2S
      omega
    calc
      2 ^ (W k + 2 * ell - 4) - D ^ 2 = S ^ 2 - (T - S) ^ 2 := by
        rw [hD, hSsq]
      _ = (S - (T - S)) * (S + (T - S)) := by
        simp [pow_two, Nat.mul_self_sub_mul_self_eq, Nat.mul_comm]
      _ = (2 * S - T) * T := by
        rw [hsub, hsum]
      _ = T * (2 * S - T) := by ac_rfl
      _ = y * (((2 ^ (2 * k + 3) * z1 k y) * 2 ^ (ell - 4)) - y * (z1 k y) ^ 2) := hprod.symm

lemma z2_low_ceilDiv_dist {k y : ℕ} (hy : 1 < y) (hyk : bw y ≤ k) (hb4 : 4 ≤ bw y) :
    z2 k y =
      (((2 ^ (W k + 2 * bw y - 4)) - (Nat.dist (2 ^ (2 * k + bw y - 2)) (y * z1 k y)) ^ 2) / y) ⌈/⌉
        2 ^ (bw y - 4) := by
  set ell : ℕ := bw y
  set T : ℕ := y * z1 k y
  have hy0 : 0 < y := by omega
  have hTle : T ≤ 2 ^ (2 * k + ell - 1) := by
    exact le_of_lt (by simpa [T, ell] using (yz1_low_lt_double (k := k) (y := y) hy hyk))
  have hm : y * (z1 k y) ^ 2 ≤ (2 ^ (2 * k + 3) * z1 k y) * 2 ^ (ell - 4) := by
    calc
      y * (z1 k y) ^ 2 = z1 k y * T := by
        dsimp [T]
        rw [pow_two]
        ac_rfl
      _ ≤ z1 k y * 2 ^ (2 * k + ell - 1) := Nat.mul_le_mul_left _ hTle
      _ = (2 ^ (2 * k + 3) * z1 k y) * 2 ^ (ell - 4) := by
        rw [show 2 ^ (2 * k + ell - 1) = 2 ^ (2 * k + 3) * 2 ^ (ell - 4) by
          rw [show 2 * k + ell - 1 = (2 * k + 3) + (ell - 4) by omega, pow_add]]
        ac_rfl
  rw [z2_low_core hyk hb4, sub_div_eq_ceilDiv (hn := by positivity) hm]
  have hnum :
      ((2 ^ (2 * k + 3) * z1 k y) * 2 ^ (ell - 4) - y * (z1 k y) ^ 2) =
        (2 ^ (W k + 2 * ell - 4) - (Nat.dist (2 ^ (2 * k + ell - 2)) (y * z1 k y)) ^ 2) / y := by
    symm
    exact Nat.div_eq_of_eq_mul_right hy0 (by simpa [ell] using (z2_low_num_eq (k := k) (y := y) hy hyk hb4))
  simpa [ell] using congrArg (fun t => t ⌈/⌉ 2 ^ (ell - 4)) hnum

lemma q0_low_bounds {k y : ℕ} (hy : 1 < y) (hyk : bw y ≤ k) (hb4 : 4 ≤ bw y) :
    2 ^ (W k) / y - 1 ≤ q0 k y ∧ q0 k y ≤ 2 ^ (W k) / y := by
  set ell : ℕ := bw y
  set a : ℕ := 2 ^ (ell - 4)
  set P : ℕ := 2 ^ (2 * ell - 4)
  set D : ℕ := Nat.dist (2 ^ (2 * k + ell - 2)) (y * z1 k y)
  set M : ℕ := (2 ^ (W k + 2 * ell - 4) - D ^ 2) / y
  set Q : ℕ := 2 ^ (W k) / y
  set s : ℕ := 2 ^ (W k) % y
  have hy0 : 0 < y := by
    omega
  have hz2 : z2 k y = M ⌈/⌉ a := by
    simpa [M, D, ell, a] using (z2_low_ceilDiv_dist (k := k) (y := y) hy hyk hb4)
  have hslt : s < y := by
    dsimp [s]
    exact Nat.mod_lt _ hy0
  have hdivmod : 2 ^ (W k) = Q * y + s := by
    dsimp [Q, s]
    simpa [Nat.mul_comm] using (Nat.div_add_mod' (2 ^ (W k)) y).symm
  have hyltpow : y < 2 ^ ell := by
    simpa [ell] using (bw_bounds (Nat.ne_of_gt hy0)).2
  have hPsplit : P = 2 ^ ell * a := by
    dsimp [P, a]
    rw [show 2 * ell - 4 = ell + (ell - 4) by omega, pow_add]
  have hAleP : a ≤ P := by
    rw [hPsplit]
    simpa [Nat.mul_comm] using
      (Nat.le_mul_of_pos_right a (show 0 < 2 ^ ell by positivity))
  have hpowLeP : 2 ^ ell ≤ P := by
    rw [hPsplit]
    simpa [Nat.mul_comm] using
      (Nat.le_mul_of_pos_right (2 ^ ell) (show 0 < a by positivity))
  have hDlt : D < y := by
    simpa [D, ell] using (z1_low_dist_lt (k := k) (y := y) hy0 hyk)
  have hD2ltPow : D ^ 2 < y * 2 ^ ell := by
    have hDltPow : D < 2 ^ ell := lt_trans hDlt hyltpow
    cases Nat.eq_zero_or_pos D with
    | inl hD0 =>
        rw [hD0, pow_two]
        positivity
    | inr hDpos =>
        have h1 : D * D < y * D := Nat.mul_lt_mul_of_pos_right hDlt hDpos
        have h2 : y * D < y * 2 ^ ell := Nat.mul_lt_mul_of_pos_left hDltPow hy0
        simpa [pow_two] using lt_trans h1 h2
  have hsP : s * P ≤ y * (P - a) := by
    have hsle : s ≤ y - 1 := Nat.le_pred_of_lt hslt
    have hpred : (y - 1) * P = y * P - P := by
      simpa [Nat.pred_eq_sub_one] using (Nat.pred_mul y P)
    have hyPa : y * a ≤ P := by
      rw [hPsplit]
      simpa [Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm] using
        (Nat.mul_le_mul_left a (le_of_lt hyltpow))
    calc
      s * P ≤ (y - 1) * P := Nat.mul_le_mul_right _ hsle
      _ = y * P - P := hpred
      _ ≤ y * P - y * a := Nat.sub_le_sub_left hyPa _
      _ = y * (P - a) := by rw [Nat.mul_sub_left_distrib]
  have hpowWP : 2 ^ (W k + 2 * ell - 4) = 2 ^ (W k) * P := by
    dsimp [P]
    rw [show W k + 2 * ell - 4 = W k + (2 * ell - 4) by omega, pow_add]
  have hMupper : M ≤ Q * P + (P - a) := by
    apply Nat.div_le_of_le_mul
    calc
      2 ^ (W k + 2 * ell - 4) - D ^ 2 ≤ 2 ^ (W k + 2 * ell - 4) := Nat.sub_le _ _
      _ = y * (Q * P) + s * P := by
        rw [hpowWP, hdivmod, Nat.add_mul]
        ac_rfl
      _ ≤ y * (Q * P) + y * (P - a) := Nat.add_le_add_left hsP _
      _ = y * (Q * P + (P - a)) := by rw [Nat.mul_add]
  have hyltW : y < 2 ^ (W k) := by
    have hbwW : bw y ≤ W k := by
      unfold W
      omega
    exact lt_of_lt_of_le (by simpa [ell] using hyltpow)
      (Nat.pow_le_pow_right (by decide : 0 < 2) hbwW)
  have hQpos : 0 < Q := by
    dsimp [Q]
    exact Nat.div_pos (Nat.le_of_lt hyltW) hy0
  have hPq : P ≤ Q * P := by
    simpa [Nat.mul_comm] using (Nat.le_mul_of_pos_right P hQpos)
  have hQbound : y * P ≤ y * (Q * P) := Nat.mul_le_mul_left _ hPq
  have hD2ltP : D ^ 2 < y * P := lt_of_lt_of_le hD2ltPow (Nat.mul_le_mul_left y hpowLeP)
  have hMdiv : y ∣ 2 ^ (W k + 2 * ell - 4) - D ^ 2 := by
    refine ⟨((2 ^ (2 * k + 3) * z1 k y) * 2 ^ (ell - 4)) - y * (z1 k y) ^ 2, ?_⟩
    simpa [D, ell] using (z2_low_num_eq (k := k) (y := y) hy hyk hb4)
  have hMlower : Q * P - P < M := by
    apply (Nat.lt_div_iff_mul_lt' hMdiv _).2
    have hraw : y * (Q * P - P) < y * (Q * P) + s * P - D ^ 2 := by
      rw [Nat.mul_sub_left_distrib]
      omega
    calc
      y * (Q * P - P) < y * (Q * P) + s * P - D ^ 2 := hraw
      _ = 2 ^ (W k + 2 * ell - 4) - D ^ 2 := by
        rw [hpowWP, hdivmod, Nat.add_mul]
        ac_rfl
  have hz2lower : (Q - 1) * 2 ^ ell < z2 k y := by
    rw [hz2]
    apply lt_ceilDiv_of_mul_lt (a := a) (by positivity)
    calc
      a * ((Q - 1) * 2 ^ ell) = (Q - 1) * P := by
        rw [hPsplit]
        ac_rfl
      _ = Q * P - P := by
        simpa [Nat.pred_eq_sub_one] using (Nat.pred_mul Q P)
      _ < M := hMlower
  have hz2upper : z2 k y < (Q + 1) * 2 ^ ell := by
    have hmul : a * z2 k y < M + a := by
      simpa [hz2, Nat.mul_comm] using (mul_ceilDiv_lt_add (a := a) (b := M) (by positivity))
    have hbound : M + a ≤ (Q + 1) * P := by
      have hsumPa : (Q * P + (P - a)) + a = Q * P + P := by
        calc
          (Q * P + (P - a)) + a = Q * P + ((P - a) + a) := by ac_rfl
          _ = Q * P + P := by rw [Nat.sub_add_cancel hAleP]
      calc
        M + a ≤ (Q * P + (P - a)) + a := Nat.add_le_add_right hMupper a
        _ = Q * P + P := hsumPa
        _ = (Q + 1) * P := by rw [Nat.add_mul, one_mul]
    have hmul' : a * z2 k y < a * ((Q + 1) * 2 ^ ell) := by
      calc
        a * z2 k y < M + a := hmul
        _ ≤ (Q + 1) * P := hbound
        _ = a * ((Q + 1) * 2 ^ ell) := by
          rw [hPsplit]
          ac_rfl
    exact (Nat.mul_lt_mul_left (by positivity : 0 < a)).1 hmul'
  have hq0lower : Q - 1 ≤ q0 k y := by
    unfold q0
    refine (Nat.le_div_iff_mul_le (by positivity : 0 < 2 ^ ell)).2 ?_
    exact le_of_lt (by simpa [ell, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using hz2lower)
  have hq0upper : q0 k y ≤ Q := by
    unfold q0
    exact Nat.le_of_lt_succ ((Nat.div_lt_iff_lt_mul (by positivity : 0 < 2 ^ ell)).2
      (by simpa [ell, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using hz2upper))
  simpa [Q] using And.intro hq0lower hq0upper

theorem low_exact_proved {k y : ℕ}
    (hk : 3 ≤ k)
    (hy : 1 < y)
    (hyk : bw y ≤ k)
    (hb4 : 4 ≤ bw y)
    (hpow : ¬ ∃ m : ℕ, y = 2 ^ m) :
    Good k y := by
  have hbounds := q0_low_bounds (k := k) (y := y) hy hyk hb4
  set Q : ℕ := 2 ^ (W k) / y
  have hy0 : 0 < y := by
    omega
  have hyltW : y < 2 ^ (W k) := by
    have hyltpow : y < 2 ^ (bw y) := (bw_bounds (Nat.ne_of_gt hy0)).2
    have hbwW : bw y ≤ W k := by
      unfold W
      omega
    exact lt_of_lt_of_le hyltpow (Nat.pow_le_pow_right (by decide : 0 < 2) hbwW)
  have hlow : Q - 1 ≤ q0 k y := by
    simpa [Q] using hbounds.1
  have hupp : q0 k y ≤ Q := by
    simpa [Q] using hbounds.2
  have hq0 : q0 k y = Q ∨ q0 k y = Q - 1 := by
    omega
  rcases hq0 with hq0 | hq0
  · exact q0_eq_floor_implies_Good hy (by simpa [Q] using hq0)
  · exact q0_eq_floor_pred_implies_Good hy hyltW hpow (by simpa [Q] using hq0)

end ReciprocalLowExact



open Reciprocal

namespace Reciprocal

end Reciprocal



open Reciprocal
open ReciprocalLowExactCore
open ReciprocalLowExact

namespace ReciprocalMiddle

/--
TeX correspondence: the middle/high symbolic core, especially
`lem:delta-bounds` and `lem:second-refinement`.

Lean expands the same argument into smaller Nat- and floor-arithmetic lemmas.
-/
lemma x_bounds {k y : ℕ} (hy : 0 < y) (hyW : y < 2 ^ (W k)) :
    2 ^ (W k - 1) ≤ x k y ∧ x k y < 2 ^ (W k) := by
  have hy0 : y ≠ 0 := Nat.ne_of_gt hy
  have hbw := bw_bounds (y := y) hy0
  have hbwW : bw y ≤ W k := bw_le_of_lt_pow hy0 hyW
  constructor
  · have hexp : 2 ^ (W k - 1) = 2 ^ (bw y - 1) * 2 ^ (W k - bw y) := by
      calc
        2 ^ (W k - 1) = 2 ^ ((bw y - 1) + (W k - bw y)) := by
          congr 1
          have : 1 ≤ bw y := Nat.succ_le_of_lt (bw_pos y)
          omega
        _ = 2 ^ (bw y - 1) * 2 ^ (W k - bw y) := by rw [pow_add]
    calc
      2 ^ (W k - 1) = 2 ^ (bw y - 1) * 2 ^ (W k - bw y) := hexp
      _ ≤ y * 2 ^ (W k - bw y) := Nat.mul_le_mul_right _ hbw.1
      _ = x k y := by
        show y * 2 ^ (W k - bw y) = x k y
        rfl
  · have hexp : 2 ^ (W k) = 2 ^ (bw y) * 2 ^ (W k - bw y) := by
      calc
        2 ^ (W k) = 2 ^ (bw y + (W k - bw y)) := by
          congr 1
          omega
        _ = 2 ^ (bw y) * 2 ^ (W k - bw y) := by rw [pow_add]
    calc
      x k y = y * 2 ^ (W k - bw y) := by
        show x k y = y * 2 ^ (W k - bw y)
        rfl
      _ < 2 ^ (bw y) * 2 ^ (W k - bw y) := Nat.mul_lt_mul_of_pos_right hbw.2 (by positivity)
      _ = 2 ^ (W k) := hexp.symm

lemma h_bounds {k y : ℕ} (hy : 0 < y) (hyW : y < 2 ^ (W k)) :
    2 ^ (2 * k - 1) ≤ h k y ∧ h k y < 2 ^ (2 * k) := by
  have hx := x_bounds (k := k) (y := y) hy hyW
  constructor
  · unfold h
    refine (Nat.le_div_iff_mul_le (by positivity : 0 < 2 ^ (2 * k))).2 ?_
    calc
      2 ^ (2 * k - 1) * 2 ^ (2 * k) = 2 ^ ((2 * k - 1) + 2 * k) := by rw [← pow_add]
      _ = 2 ^ (W k - 1) := by
        unfold W
        congr 1
        omega
      _ ≤ x k y := hx.1
  · unfold h
    refine (Nat.div_lt_iff_lt_mul (by positivity : 0 < 2 ^ (2 * k))).2 ?_
    calc
      x k y < 2 ^ (W k) := hx.2
      _ = 2 ^ ((2 * k) + 2 * k) := by
        unfold W
        congr 1
        omega
      _ = 2 ^ (2 * k) * 2 ^ (2 * k) := by rw [pow_add]

lemma g_bounds {k y : ℕ} (hy : 0 < y) (hyW : y < 2 ^ (W k)) :
    2 ^ (k - 1) ≤ g k y ∧ g k y < 2 ^ k := by
  have hh := h_bounds (k := k) (y := y) hy hyW
  constructor
  · unfold g
    refine (Nat.le_div_iff_mul_le (by positivity : 0 < 2 ^ k)).2 ?_
    calc
      2 ^ (k - 1) * 2 ^ k = 2 ^ ((k - 1) + k) := by rw [← pow_add]
      _ = 2 ^ (2 * k - 1) := by
        congr 1
        omega
      _ ≤ h k y := hh.1
  · unfold g
    refine (Nat.div_lt_iff_lt_mul (by positivity : 0 < 2 ^ k)).2 ?_
    calc
      h k y < 2 ^ (2 * k) := hh.2
      _ = 2 ^ (k + k) := by
        congr 1
        omega
      _ = 2 ^ k * 2 ^ k := by rw [pow_add]

lemma z0_le_pow {k y : ℕ} (hy : 0 < y) (hyW : y < 2 ^ (W k)) :
    z0 k y ≤ 2 ^ k := by
  have hg := g_bounds (k := k) (y := y) hy hyW
  have hgpos : 0 < g k y := lt_of_lt_of_le (by positivity : 0 < 2 ^ (k - 1)) hg.1
  unfold z0
  apply Nat.div_le_of_le_mul
  calc
    2 ^ (2 * k - 1) = 2 ^ k * 2 ^ (k - 1) := by
      rw [← pow_add]
      congr 1
      omega
    _ ≤ g k y * 2 ^ k := by
      simpa [Nat.mul_comm] using Nat.mul_le_mul_left (2 ^ k) hg.1

lemma h_middle_exact {k y : ℕ} (hyk : bw y ≤ 2 * k) :
    h k y = y * 2 ^ (2 * k - bw y) := by
  unfold h x W
  rw [show 4 * k - bw y = 2 * k + (2 * k - bw y) by omega, pow_add]
  rw [show y * (2 ^ (2 * k) * 2 ^ (2 * k - bw y)) = 2 ^ (2 * k) * (y * 2 ^ (2 * k - bw y)) by ac_rfl]
  exact Nat.mul_div_right (y * 2 ^ (2 * k - bw y)) (by positivity : 0 < 2 ^ (2 * k))

lemma x_middle_from_h {k y : ℕ} (hyk : bw y ≤ 2 * k) :
    x k y = 2 ^ (2 * k) * h k y := by
  unfold x W
  rw [h_middle_exact hyk]
  rw [show 4 * k - bw y = 2 * k + (2 * k - bw y) by omega, pow_add]
  ac_rfl

lemma h_middle_bounds {k y : ℕ} (hy : 0 < y) (_hyW : y < 2 ^ (W k)) (hklt : k < bw y) (hyk : bw y ≤ 2 * k) :
    2 ^ (2 * k - 1) ≤ h k y ∧ h k y < 2 ^ (2 * k) := by
  have hy0 : y ≠ 0 := Nat.ne_of_gt hy
  have hbw := bw_bounds (y := y) hy0
  constructor
  · have hexp : 2 ^ (2 * k - 1) = 2 ^ (bw y - 1) * 2 ^ (2 * k - bw y) := by
      calc
        2 ^ (2 * k - 1) = 2 ^ ((bw y - 1) + (2 * k - bw y)) := by
          congr 1
          omega
        _ = 2 ^ (bw y - 1) * 2 ^ (2 * k - bw y) := by rw [pow_add]
    calc
      2 ^ (2 * k - 1) = 2 ^ (bw y - 1) * 2 ^ (2 * k - bw y) := hexp
      _ ≤ y * 2 ^ (2 * k - bw y) := Nat.mul_le_mul_right _ hbw.1
      _ = h k y := by rw [h_middle_exact hyk]
  · have hexp : 2 ^ (2 * k) = 2 ^ (bw y) * 2 ^ (2 * k - bw y) := by
      calc
        2 ^ (2 * k) = 2 ^ (bw y + (2 * k - bw y)) := by
          congr 1
          omega
        _ = 2 ^ (bw y) * 2 ^ (2 * k - bw y) := by rw [pow_add]
    calc
      h k y = y * 2 ^ (2 * k - bw y) := h_middle_exact hyk
      _ < 2 ^ (bw y) * 2 ^ (2 * k - bw y) := Nat.mul_lt_mul_of_pos_right hbw.2 (by positivity)
      _ = 2 ^ (2 * k) := hexp.symm

lemma div_eq_of_add_mul_lt {d q r : ℕ} (hd : 0 < d) (hr : r < d) :
    (r + q * d) / d = q := by
  rw [Nat.add_mul_div_right _ _ hd, Nat.div_eq_of_lt hr, zero_add]

lemma m_dist_lt_pow {k y : ℕ} (hk : 3 ≤ k) (hy : 0 < y) (hyW : y < 2 ^ (W k)) :
    Nat.dist ((h k y * z0 k y) / 2 ^ k) (2 ^ (2 * k - 1)) < 2 ^ k := by
  have hh := h_bounds (k := k) (y := y) hy hyW
  have hg := g_bounds (k := k) (y := y) hy hyW
  have hz0le := z0_le_pow (k := k) (y := y) hy hyW
  have hk1 : 1 ≤ k := by omega
  have hgpos : 0 < g k y := lt_of_lt_of_le (by positivity : 0 < 2 ^ (k - 1)) hg.1
  set a : ℕ := h k y % 2 ^ k
  set e : ℕ := 2 ^ (2 * k - 1) % g k y
  set t : ℕ := (a * z0 k y) / 2 ^ k
  set u : ℕ := (a * z0 k y) % 2 ^ k
  set m : ℕ := (h k y * z0 k y) / 2 ^ k
  set C : ℕ := 2 ^ (2 * k - 1)
  have ha_lt : a < 2 ^ k := by
    dsimp [a]
    exact Nat.mod_lt _ (by positivity)
  have hhdecomp : h k y = 2 ^ k * g k y + a := by
    dsimp [a]
    simpa [g, Nat.mul_comm, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using
      (Nat.div_add_mod' (h k y) (2 ^ k)).symm
  have hCdecomp : C = g k y * z0 k y + e := by
    dsimp [C, e]
    simpa [z0, Nat.mul_comm, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using
      (Nat.div_add_mod' (2 ^ (2 * k - 1)) (g k y)).symm
  have he_lt : e < g k y := by
    dsimp [e]
    exact Nat.mod_lt _ hgpos
  have hgz : g k y * z0 k y = C - e := by
    exact Nat.eq_sub_of_add_eq' (by simpa [Nat.add_comm] using hCdecomp.symm)
  have hu_lt : u < 2 ^ k := by
    dsimp [u]
    exact Nat.mod_lt _ (by positivity)
  have haz0_lt : a * z0 k y < 2 ^ (2 * k) := by
    calc
      a * z0 k y ≤ a * 2 ^ k := Nat.mul_le_mul_left _ hz0le
      _ < 2 ^ k * 2 ^ k := Nat.mul_lt_mul_of_pos_right ha_lt (by positivity)
      _ = 2 ^ (2 * k) := by
        rw [← pow_add]
        congr 1
        omega
  have ht_lt : t < 2 ^ k := by
    dsimp [t]
    exact (Nat.div_lt_iff_lt_mul (by positivity : 0 < 2 ^ k)).2 <| by
      simpa [show 2 ^ k * 2 ^ k = 2 ^ (2 * k) by
        rw [← pow_add]
        congr 1
        omega] using haz0_lt
  have haz0decomp : a * z0 k y = 2 ^ k * t + u := by
    dsimp [t, u]
    simpa [Nat.mul_comm, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using
      (Nat.div_add_mod' (a * z0 k y) (2 ^ k)).symm
  have hhz0eq : h k y * z0 k y = 2 ^ k * (C + t - e) + u := by
    calc
      h k y * z0 k y = (2 ^ k * g k y + a) * z0 k y := by rw [hhdecomp]
      _ = 2 ^ k * (g k y * z0 k y) + a * z0 k y := by
        rw [Nat.add_mul, Nat.mul_assoc]
      _ = 2 ^ k * (C - e) + (2 ^ k * t + u) := by rw [hgz, haz0decomp]
      _ = 2 ^ k * ((C - e) + t) + u := by
        rw [Nat.mul_add]
        ac_rfl
      _ = 2 ^ k * (C + t - e) + u := by
        have hpowle : 2 ^ k ≤ C := by
          dsimp [C]
          exact Nat.pow_le_pow_right (by decide : 0 < 2) (by omega)
        have he_le : e ≤ C := le_trans (Nat.le_of_lt he_lt) (le_trans (le_of_lt hg.2) hpowle)
        have hCe' : t + C - e = t + (C - e) := Nat.add_sub_assoc he_le t
        have hCe : C - e + t = C + t - e := by
          simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using hCe'.symm
        rw [hCe]
  have hm : m = C + t - e := by
    dsimp [m]
    rw [hhz0eq, Nat.add_comm, Nat.mul_comm (2 ^ k) (C + t - e)]
    exact div_eq_of_add_mul_lt (by positivity) hu_lt
  have hm_upper : m ≤ C + (2 ^ k - 1) := by
    have htt : t ≤ 2 ^ k - 1 := Nat.le_pred_of_lt ht_lt
    rw [hm]
    exact le_trans (Nat.sub_le _ _) (Nat.add_le_add_left htt _)
  have hm_lower : C ≤ m + (2 ^ k - 1) := by
    have he_le : e ≤ 2 ^ k - 1 := by
      have : e < 2 ^ k := lt_trans he_lt hg.2
      exact Nat.le_pred_of_lt this
    have hme : m + e = C + t := by
      rw [hm]
      have hpowle : 2 ^ k ≤ C := by
        dsimp [C]
        exact Nat.pow_le_pow_right (by decide : 0 < 2) (by omega)
      have hgleC : g k y ≤ C := le_trans (Nat.le_of_lt hg.2) hpowle
      have he_le_Ct : e ≤ C + t := by
        exact le_trans (Nat.le_of_lt he_lt) (le_trans hgleC (Nat.le_add_right _ _))
      exact Nat.sub_add_cancel he_le_Ct
    have hCe : C ≤ m + e := by
      rw [hme]
      exact Nat.le_add_right _ _
    exact le_trans hCe (Nat.add_le_add_left he_le _)
  have hdist : Nat.dist m C < 2 ^ k := by
    by_cases hmC : m ≤ C
    · rw [Nat.dist_eq_sub_of_le hmC]
      have : C - m ≤ 2 ^ k - 1 := by omega
      omega
    · have hCm : C < m := lt_of_not_ge hmC
      rw [Nat.dist_eq_sub_of_le_right (le_of_lt hCm)]
      have : m - C ≤ 2 ^ k - 1 := by omega
      omega
  simpa [m, C] using hdist

lemma quad_eq_sub_dist_sq {k m : ℕ} (hk : 1 ≤ k) (hm : m ≤ 2 ^ (2 * k)) :
    m * (2 ^ (2 * k) - m) = 2 ^ (W k - 2) - (Nat.dist m (2 ^ (2 * k - 1))) ^ 2 := by
  set C : ℕ := 2 ^ (2 * k - 1)
  have hCC : 2 ^ (2 * k) = 2 * C := by
    dsimp [C]
    rw [show 2 * k = (2 * k - 1) + 1 by omega, pow_add]
    norm_num
    rw [Nat.mul_comm]
  have hCsq : C ^ 2 = 2 ^ (W k - 2) := by
    dsimp [C]
    rw [pow_two, ← pow_add]
    unfold W
    congr 1
    omega
  by_cases hmC : m ≤ C
  · have hD : Nat.dist m C = C - m := by
      rw [Nat.dist_eq_sub_of_le hmC]
    have h1 : C - (C - m) = m := by
      exact Nat.sub_sub_self hmC
    have h2 : C + (C - m) = 2 * C - m := by
      omega
    calc
      m * (2 ^ (2 * k) - m) = m * (2 * C - m) := by rw [hCC]
      _ = (C - (C - m)) * (C + (C - m)) := by rw [h1, h2]
      _ = C ^ 2 - (C - m) ^ 2 := by
        simp [pow_two, Nat.mul_self_sub_mul_self_eq, Nat.mul_comm]
      _ = 2 ^ (W k - 2) - (Nat.dist m C) ^ 2 := by rw [hCsq, hD]
  · have hCm : C < m := lt_of_not_ge hmC
    have hm2C : m ≤ 2 * C := by
      rw [← hCC]
      exact hm
    have hD : Nat.dist m C = m - C := by
      rw [Nat.dist_eq_sub_of_le_right (le_of_lt hCm)]
    have h1 : C + (m - C) = m := by
      exact Nat.add_sub_of_le (le_of_lt hCm)
    have h2 : C - (m - C) = 2 * C - m := by
      omega
    calc
      m * (2 ^ (2 * k) - m) = m * (2 * C - m) := by rw [hCC]
      _ = (C + (m - C)) * (C - (m - C)) := by rw [h1, h2]
      _ = C ^ 2 - (m - C) ^ 2 := by
        simp [pow_two, Nat.mul_self_sub_mul_self_eq]
      _ = 2 ^ (W k - 2) - (Nat.dist m C) ^ 2 := by rw [hCsq, hD]

lemma quad_term_dist_lt_two_h {k y : ℕ} (hk : 3 ≤ k) (hy : 0 < y) (hyW : y < 2 ^ (W k)) :
    let m := (h k y * z0 k y) / 2 ^ k
    Nat.dist (m * (2 ^ (2 * k) - m)) (2 ^ (W k - 2)) < 2 * h k y := by
  have hh := h_bounds (k := k) (y := y) hy hyW
  have hz0le := z0_le_pow (k := k) (y := y) hy hyW
  have hk1 : 1 ≤ k := by omega
  let m : ℕ := (h k y * z0 k y) / 2 ^ k
  have hm_lt : m < 2 ^ (2 * k) := by
    dsimp [m]
    refine (Nat.div_lt_iff_lt_mul (by positivity : 0 < 2 ^ k)).2 ?_
    calc
      h k y * z0 k y ≤ h k y * 2 ^ k := Nat.mul_le_mul_left _ hz0le
      _ < 2 ^ (2 * k) * 2 ^ k := Nat.mul_lt_mul_of_pos_right hh.2 (by positivity)
  have hquad : m * (2 ^ (2 * k) - m) = 2 ^ (W k - 2) - (Nat.dist m (2 ^ (2 * k - 1))) ^ 2 := by
    simpa [m] using quad_eq_sub_dist_sq hk1 (le_of_lt hm_lt)
  have hDlt : Nat.dist m (2 ^ (2 * k - 1)) < 2 ^ k := by
    simpa [m] using (m_dist_lt_pow (k := k) (y := y) hk hy hyW)
  have hDsq : (Nat.dist m (2 ^ (2 * k - 1))) ^ 2 < 2 * h k y := by
    have hsmall : (Nat.dist m (2 ^ (2 * k - 1))) ^ 2 < 2 ^ (2 * k) := by
      cases Nat.eq_zero_or_pos (Nat.dist m (2 ^ (2 * k - 1))) with
      | inl hzero =>
          rw [hzero, pow_two]
          positivity
      | inr hpos =>
          calc
            (Nat.dist m (2 ^ (2 * k - 1))) ^ 2
                = Nat.dist m (2 ^ (2 * k - 1)) * Nat.dist m (2 ^ (2 * k - 1)) := by rw [pow_two]
            _ < Nat.dist m (2 ^ (2 * k - 1)) * 2 ^ k := Nat.mul_lt_mul_of_pos_left hDlt hpos
            _ < 2 ^ k * 2 ^ k := Nat.mul_lt_mul_of_pos_right hDlt (by positivity)
            _ = 2 ^ (2 * k) := by
                  rw [← pow_add]
                  congr 1
                  omega
    calc
      (Nat.dist m (2 ^ (2 * k - 1))) ^ 2 < 2 ^ (2 * k) := hsmall
      _ ≤ 2 * h k y := by
        have hpow : 2 ^ (2 * k) = 2 * 2 ^ (2 * k - 1) := by
          rw [show 2 * k = 1 + (2 * k - 1) by omega, pow_add]
          norm_num
        rw [hpow]
        exact Nat.mul_le_mul_left _ hh.1
  have hDle : (Nat.dist m (2 ^ (2 * k - 1))) ^ 2 ≤ 2 ^ (W k - 2) := by
    refine le_trans (le_of_lt hDsq) ?_
    calc
      2 * h k y ≤ 2 * 2 ^ (2 * k) := Nat.mul_le_mul_left _ (le_of_lt hh.2)
      _ = 2 ^ (2 * k + 1) := by
        rw [show 2 * k + 1 = 1 + 2 * k by omega, pow_add]
        norm_num
      _ ≤ 2 ^ (W k - 2) := by
        exact Nat.pow_le_pow_right (by decide : 0 < 2) (by
          unfold W
          omega)
  calc
    Nat.dist (m * (2 ^ (2 * k) - m)) (2 ^ (W k - 2))
        = 2 ^ (W k - 2) - (m * (2 ^ (2 * k) - m)) := by
            rw [Nat.dist_eq_sub_of_le]
            rw [hquad]
            exact Nat.sub_le _ _
    _ = (Nat.dist m (2 ^ (2 * k - 1))) ^ 2 := by rw [hquad, Nat.sub_sub_self hDle]
    _ < 2 * h k y := hDsq

lemma m_lt_pow {k y : ℕ} (hy : 0 < y) (hyW : y < 2 ^ (W k)) :
    let m := (h k y * z0 k y) / 2 ^ k
    m < 2 ^ (2 * k) := by
  have hh := h_bounds (k := k) (y := y) hy hyW
  have hz0le := z0_le_pow (k := k) (y := y) hy hyW
  dsimp
  refine (Nat.div_lt_iff_lt_mul (by positivity : 0 < 2 ^ k)).2 ?_
  calc
    h k y * z0 k y ≤ h k y * 2 ^ k := Nat.mul_le_mul_left _ hz0le
    _ < 2 ^ (2 * k) * 2 ^ k := Nat.mul_lt_mul_of_pos_right hh.2 (by positivity)

lemma hp_add_R_eq_aux {d H Z m p rho u : ℕ} (hd : 0 < d)
    (hz : H * Z = d * m + u) (hp : Z * m = d * p + rho) :
    H * p + (H * rho) / d = m ^ 2 + (u * m) / d := by
  have hleft : (H * (Z * m)) / d = H * p + (H * rho) / d := by
    rw [hp]
    calc
      (H * (d * p + rho)) / d = (H * (d * p) + H * rho) / d := by rw [Nat.mul_add]
      _ = (d * (H * p) + H * rho) / d := by
        congr 1
        ac_rfl
      _ = H * p + (H * rho) / d := by rw [Nat.mul_add_div hd]
  have hright : (H * (Z * m)) / d = m ^ 2 + (u * m) / d := by
    rw [show H * (Z * m) = (H * Z) * m by ac_rfl, hz]
    calc
      ((d * m + u) * m) / d = (d * m * m + u * m) / d := by rw [Nat.add_mul]
      _ = (d * (m ^ 2) + u * m) / d := by
        congr 1
        rw [pow_two, Nat.mul_assoc]
      _ = m ^ 2 + (u * m) / d := by rw [Nat.mul_add_div hd]
  
  exact hleft.symm.trans hright

lemma hz1_quad_correction {k y : ℕ} (hy : 0 < y) (hyW : y < 2 ^ (W k)) :
    let m : ℕ := (h k y * z0 k y) / 2 ^ k
    -- `p` is introduced in the proof body below.
    let rho : ℕ := (z0 k y * m) % 2 ^ k
    let u : ℕ := (h k y * z0 k y) % 2 ^ k
    let V : ℕ := (u * m) / 2 ^ k
    let R : ℕ := (h k y * rho) / 2 ^ k
    h k y * z1 k y = m * (2 ^ (2 * k) - m) + (2 ^ k * u - V) + R := by
  set m : ℕ := (h k y * z0 k y) / 2 ^ k
  set p : ℕ := (z0 k y * m) / 2 ^ k
  set rho : ℕ := (z0 k y * m) % 2 ^ k
  set u : ℕ := (h k y * z0 k y) % 2 ^ k
  set V : ℕ := (u * m) / 2 ^ k
  set R : ℕ := (h k y * rho) / 2 ^ k
  change h k y * z1 k y = m * (2 ^ (2 * k) - m) + (2 ^ k * u - V) + R
  have hu_decomp : h k y * z0 k y = 2 ^ k * m + u := by
    dsimp [m, u]
    simpa [Nat.mul_comm, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using
      (Nat.div_add_mod' (h k y * z0 k y) (2 ^ k)).symm
  have hp_decomp : z0 k y * m = 2 ^ k * p + rho := by
    dsimp [p, rho]
    simpa [Nat.mul_comm, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using
      (Nat.div_add_mod' (z0 k y * m) (2 ^ k)).symm
  have hpR_eq : h k y * p + R = m ^ 2 + V := by
    dsimp [R, V]
    exact hp_add_R_eq_aux (d := 2 ^ k) (H := h k y) (Z := z0 k y) (by positivity) hu_decomp hp_decomp
  have hm_lt : m < 2 ^ (2 * k) := by
    simpa [m] using (m_lt_pow (k := k) (y := y) hy hyW)
  have hm_le : m ≤ 2 ^ (2 * k) := le_of_lt hm_lt
  have hz1_eq : z1 k y = 2 ^ k * z0 k y - p := by
    dsimp [p, m]
    rfl
  have hmain : h k y * (2 ^ k * z0 k y) = 2 ^ (2 * k) * m + 2 ^ k * u := by
    have hpow : 2 ^ k * 2 ^ k = 2 ^ (2 * k) := by
      rw [← pow_add]
      congr 1
      omega
    have hpow2 : 2 ^ k * (2 ^ k * m) = 2 ^ (2 * k) * m := by
      calc
        2 ^ k * (2 ^ k * m) = (2 ^ k * 2 ^ k) * m := by ac_rfl
        _ = 2 ^ (2 * k) * m := by rw [hpow]
    calc
      h k y * (2 ^ k * z0 k y) = 2 ^ k * (h k y * z0 k y) := by ac_rfl
      _ = 2 ^ k * (2 ^ k * m + u) := by rw [hu_decomp]
      _ = 2 ^ k * (2 ^ k * m) + 2 ^ k * u := by rw [Nat.mul_add]
      _ = 2 ^ (2 * k) * m + 2 ^ k * u := by rw [hpow2]
  have hRle : R ≤ m ^ 2 + V := by
    rw [← hpR_eq]
    exact Nat.le_add_left _ _
  have hp_eq : h k y * p = m ^ 2 + V - R := by
    exact Nat.eq_sub_of_add_eq hpR_eq
  have hsub : m * (2 ^ (2 * k) - m) = 2 ^ (2 * k) * m - m ^ 2 := by
    rw [Nat.mul_sub_left_distrib, pow_two]
    ac_rfl
  have hVle : V ≤ 2 ^ k * u := by
    dsimp [V]
    have hpow : 2 ^ k * 2 ^ k = 2 ^ (2 * k) := by
      rw [← pow_add]
      congr 1
      omega
    have hpow2 : u * 2 ^ (2 * k) = (2 ^ k * u) * 2 ^ k := by
      calc
        u * 2 ^ (2 * k) = u * (2 ^ k * 2 ^ k) := by rw [← hpow]
        _ = (2 ^ k * u) * 2 ^ k := by ac_rfl
    apply Nat.div_le_of_le_mul
    calc
      u * m ≤ u * 2 ^ (2 * k) := Nat.mul_le_mul_left _ hm_le
      _ = (2 ^ k * u) * 2 ^ k := hpow2
      _ = 2 ^ k * (2 ^ k * u) := by ac_rfl
  have hm_sq_le : m ^ 2 ≤ 2 ^ (2 * k) * m := by
    rw [pow_two]
    exact Nat.mul_le_mul_right _ hm_le
  have hsum_le : m ^ 2 + V ≤ 2 ^ (2 * k) * m + 2 ^ k * u := Nat.add_le_add hm_sq_le hVle
  have htmp1 :
      (2 ^ (2 * k) * m + 2 ^ k * u) - (m ^ 2 + V - R)
        = ((2 ^ (2 * k) * m + 2 ^ k * u) - (m ^ 2 + V)) + R := by
    apply
      (Nat.sub_eq_iff_eq_add
        (a := 2 ^ (2 * k) * m + 2 ^ k * u)
        (b := m ^ 2 + V - R)
        (c := ((2 ^ (2 * k) * m + 2 ^ k * u) - (m ^ 2 + V)) + R)
        (le_trans (Nat.sub_le _ _) hsum_le)).2
    calc
      (2 ^ (2 * k) * m + 2 ^ k * u)
          = ((2 ^ (2 * k) * m + 2 ^ k * u) - (m ^ 2 + V)) + (m ^ 2 + V) := by
              exact (Nat.sub_add_cancel hsum_le).symm
      _ = ((2 ^ (2 * k) * m + 2 ^ k * u) - (m ^ 2 + V)) + ((m ^ 2 + V - R) + R) := by
            rw [Nat.sub_add_cancel hRle]
      _ = (((2 ^ (2 * k) * m + 2 ^ k * u) - (m ^ 2 + V)) + R) + (m ^ 2 + V - R) := by
            ac_rfl
  have htmp2 :
      ((2 ^ (2 * k) * m + 2 ^ k * u) - (m ^ 2 + V))
        = (2 ^ (2 * k) * m - m ^ 2) + (2 ^ k * u - V) := by
    apply (Nat.sub_eq_iff_eq_add hsum_le).2
    calc
      (2 ^ (2 * k) * m + 2 ^ k * u)
          = ((2 ^ (2 * k) * m - m ^ 2) + m ^ 2) + ((2 ^ k * u - V) + V) := by
              rw [Nat.sub_add_cancel hm_sq_le, Nat.sub_add_cancel hVle]
      _ = ((2 ^ (2 * k) * m - m ^ 2) + (2 ^ k * u - V)) + (m ^ 2 + V) := by
            ac_rfl
  calc
    h k y * z1 k y = h k y * (2 ^ k * z0 k y - p) := by rw [hz1_eq]
    _ = h k y * (2 ^ k * z0 k y) - h k y * p := by rw [Nat.mul_sub_left_distrib]
    _ = (2 ^ (2 * k) * m + 2 ^ k * u) - (m ^ 2 + V - R) := by rw [hmain, hp_eq]
    _ = ((2 ^ (2 * k) * m + 2 ^ k * u) - (m ^ 2 + V)) + R := htmp1
    _ = (2 ^ (2 * k) * m - m ^ 2) + (2 ^ k * u - V) + R := by rw [htmp2]
    _ = m * (2 ^ (2 * k) - m) + (2 ^ k * u - V) + R := by rw [← hsub]

lemma correction_lt_h_add_sq {k y : ℕ} (hk : 3 ≤ k) (hy : 0 < y) (hyW : y < 2 ^ (W k)) :
    let m : ℕ := (h k y * z0 k y) / 2 ^ k
    let u : ℕ := (h k y * z0 k y) % 2 ^ k
    let V : ℕ := (u * m) / 2 ^ k
    let D : ℕ := Nat.dist m (2 ^ (2 * k - 1))
    2 ^ k * u - V < h k y + D ^ 2 := by
  set m : ℕ := (h k y * z0 k y) / 2 ^ k
  set u : ℕ := (h k y * z0 k y) % 2 ^ k
  set V : ℕ := (u * m) / 2 ^ k
  set C : ℕ := 2 ^ (2 * k - 1)
  set D : ℕ := Nat.dist m C
  change 2 ^ k * u - V < h k y + D ^ 2
  have hh := h_bounds (k := k) (y := y) hy hyW
  have hu_lt : u < 2 ^ k := by
    dsimp [u]
    exact Nat.mod_lt _ (by positivity)
  have huhalf_lt_h : u * 2 ^ (k - 1) < h k y := by
    have hlt : u * 2 ^ (k - 1) < 2 ^ (2 * k - 1) := by
      calc
        u * 2 ^ (k - 1) < 2 ^ k * 2 ^ (k - 1) := Nat.mul_lt_mul_of_pos_right hu_lt (by positivity)
        _ = 2 ^ (2 * k - 1) := by
          rw [← pow_add]
          congr 1
          omega
    exact lt_of_lt_of_le hlt hh.1
  by_cases hCm : C ≤ m
  · have hV_lower : u * 2 ^ (k - 1) ≤ V := by
      dsimp [V]
      refine (Nat.le_div_iff_mul_le (by positivity : 0 < 2 ^ k)).2 ?_
      calc
        (u * 2 ^ (k - 1)) * 2 ^ k = u * C := by
          dsimp [C]
          rw [show 2 * k - 1 = (k - 1) + k by omega, pow_add]
          ac_rfl
        _ ≤ u * m := Nat.mul_le_mul_left _ hCm
    have hBle : 2 ^ k * u - V ≤ u * 2 ^ (k - 1) := by
      have hpow_base : 2 ^ k = 2 ^ (k - 1) + 2 ^ (k - 1) := by
        calc
          2 ^ k = 2 ^ (k - 1) * 2 := by
            rw [show k = (k - 1) + 1 by omega, pow_add]
            norm_num
          _ = 2 ^ (k - 1) + 2 ^ (k - 1) := by
            rw [Nat.mul_comm, two_mul]
      have hpow : 2 ^ k * u = u * 2 ^ (k - 1) + u * 2 ^ (k - 1) := by
        rw [hpow_base, Nat.add_mul]
        ac_rfl
      rw [hpow]
      omega
    exact lt_of_le_of_lt hBle (lt_of_lt_of_le huhalf_lt_h (Nat.le_add_right _ _))
  · have hCm' : m < C := lt_of_not_ge hCm
    have hD_eq : D = C - m := by
      dsimp [D]
      rw [Nat.dist_eq_sub_of_le (le_of_lt hCm')]
    have hD_pos : 0 < D := by
      rw [hD_eq]
      exact Nat.sub_pos_of_lt hCm'
    have hV_lower : u * 2 ^ (k - 1) - D ≤ V := by
      dsimp [V]
      refine (Nat.le_div_iff_mul_le (by positivity : 0 < 2 ^ k)).2 ?_
      have hud_le : u * D ≤ D * 2 ^ k := by
        have hu_le : u ≤ 2 ^ k := Nat.le_of_lt hu_lt
        simpa [Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using (Nat.mul_le_mul_right D hu_le)
      calc
        (u * 2 ^ (k - 1) - D) * 2 ^ k = u * C - D * 2 ^ k := by
          rw [Nat.sub_mul]
          dsimp [C]
          rw [show 2 * k - 1 = (k - 1) + k by omega, pow_add]
          ac_rfl
        _ ≤ u * C - u * D := by exact Nat.sub_le_sub_left hud_le _
        _ = u * (C - D) := by rw [Nat.mul_sub_left_distrib]
        _ = u * m := by
          rw [hD_eq, Nat.sub_sub_self (le_of_lt hCm')]
    have hBle : 2 ^ k * u - V ≤ u * 2 ^ (k - 1) + D := by
      have hpow_base : 2 ^ k = 2 ^ (k - 1) + 2 ^ (k - 1) := by
        calc
          2 ^ k = 2 ^ (k - 1) * 2 := by
            rw [show k = (k - 1) + 1 by omega, pow_add]
            norm_num
          _ = 2 ^ (k - 1) + 2 ^ (k - 1) := by
            rw [Nat.mul_comm, two_mul]
      have hpow : 2 ^ k * u = u * 2 ^ (k - 1) + u * 2 ^ (k - 1) := by
        rw [hpow_base, Nat.add_mul]
        ac_rfl
      rw [hpow]
      omega
    have hDleSq : D ≤ D ^ 2 := by
      have : D ≤ D * D := Nat.le_mul_of_pos_right D hD_pos
      simpa [pow_two, Nat.mul_comm] using this
    have hbound : u * 2 ^ (k - 1) + D < h k y + D ^ 2 := by
      exact Nat.add_lt_add_of_lt_of_le huhalf_lt_h hDleSq
    exact lt_of_le_of_lt hBle hbound

lemma V_le_pow_mul_u {k y : ℕ} (hy : 0 < y) (hyW : y < 2 ^ (W k)) :
    let m : ℕ := (h k y * z0 k y) / 2 ^ k
    let u : ℕ := (h k y * z0 k y) % 2 ^ k
    let V : ℕ := (u * m) / 2 ^ k
    V ≤ 2 ^ k * u := by
  set m : ℕ := (h k y * z0 k y) / 2 ^ k
  set u : ℕ := (h k y * z0 k y) % 2 ^ k
  set V : ℕ := (u * m) / 2 ^ k
  change V ≤ 2 ^ k * u
  have hm_lt : m < 2 ^ (2 * k) := by
    simpa [m] using (m_lt_pow (k := k) (y := y) hy hyW)
  have hm_le : m ≤ 2 ^ (2 * k) := le_of_lt hm_lt
  dsimp [V]
  have hpow : 2 ^ k * 2 ^ k = 2 ^ (2 * k) := by
    rw [← pow_add]
    congr 1
    omega
  have hpow2 : u * 2 ^ (2 * k) = (2 ^ k * u) * 2 ^ k := by
    calc
      u * 2 ^ (2 * k) = u * (2 ^ k * 2 ^ k) := by rw [← hpow]
      _ = (2 ^ k * u) * 2 ^ k := by ac_rfl
  apply Nat.div_le_of_le_mul
  calc
    u * m ≤ u * 2 ^ (2 * k) := Nat.mul_le_mul_left _ hm_le
    _ = (2 ^ k * u) * 2 ^ k := hpow2
    _ = 2 ^ k * (2 ^ k * u) := by ac_rfl

lemma R_lt_h {k y : ℕ} (hy : 0 < y) (hyW : y < 2 ^ (W k)) :
    let m : ℕ := (h k y * z0 k y) / 2 ^ k
    let rho : ℕ := (z0 k y * m) % 2 ^ k
    let R : ℕ := (h k y * rho) / 2 ^ k
    R < h k y := by
  set m : ℕ := (h k y * z0 k y) / 2 ^ k
  set rho : ℕ := (z0 k y * m) % 2 ^ k
  set R : ℕ := (h k y * rho) / 2 ^ k
  change R < h k y
  have hh := h_bounds (k := k) (y := y) hy hyW
  have hhpos : 0 < h k y := lt_of_lt_of_le (by positivity : 0 < 2 ^ (2 * k - 1)) hh.1
  have hrho_lt : rho < 2 ^ k := by
    dsimp [rho]
    exact Nat.mod_lt _ (by positivity)
  dsimp [R]
  exact (Nat.div_lt_iff_lt_mul (by positivity : 0 < 2 ^ k)).2 <| by
    calc
      h k y * rho < h k y * 2 ^ k := Nat.mul_lt_mul_of_pos_left hrho_lt hhpos
      _ = 2 ^ k * h k y := by ac_rfl
      _ = h k y * 2 ^ k := by ac_rfl

lemma hz1_lower_bound {k y : ℕ} (hk : 3 ≤ k) (hy : 0 < y) (hyW : y < 2 ^ (W k)) :
    let m : ℕ := (h k y * z0 k y) / 2 ^ k
    let D : ℕ := Nat.dist m (2 ^ (2 * k - 1))
    2 ^ (W k - 2) - D ^ 2 ≤ h k y * z1 k y := by
  set m : ℕ := (h k y * z0 k y) / 2 ^ k
  set u : ℕ := (h k y * z0 k y) % 2 ^ k
  set V : ℕ := (u * m) / 2 ^ k
  set rho : ℕ := (z0 k y * m) % 2 ^ k
  set R : ℕ := (h k y * rho) / 2 ^ k
  set D : ℕ := Nat.dist m (2 ^ (2 * k - 1))
  change 2 ^ (W k - 2) - D ^ 2 ≤ h k y * z1 k y
  have hdecomp : h k y * z1 k y = m * (2 ^ (2 * k) - m) + (2 ^ k * u - V) + R := by
    simpa [m, u, V, R] using (hz1_quad_correction (k := k) (y := y) hy hyW)
  have hm_lt : m < 2 ^ (2 * k) := by
    simpa [m] using (m_lt_pow (k := k) (y := y) hy hyW)
  have hquad : m * (2 ^ (2 * k) - m) = 2 ^ (W k - 2) - D ^ 2 := by
    have hk1 : 1 ≤ k := by omega
    simpa [m, D] using (quad_eq_sub_dist_sq (k := k) (m := m) hk1 (le_of_lt hm_lt))
  have hVle : V ≤ 2 ^ k * u := by
    simpa [m, u, V] using (V_le_pow_mul_u (k := k) (y := y) hy hyW)
  calc
    2 ^ (W k - 2) - D ^ 2 = m * (2 ^ (2 * k) - m) := by rw [hquad]
    _ ≤ m * (2 ^ (2 * k) - m) + ((2 ^ k * u - V) + R) := Nat.le_add_right _ _
    _ = h k y * z1 k y := by simpa [Nat.add_assoc] using hdecomp.symm

lemma hz1_upper_bound {k y : ℕ} (hk : 3 ≤ k) (hy : 0 < y) (hyW : y < 2 ^ (W k)) :
    h k y * z1 k y < 2 ^ (W k - 2) + 2 * h k y := by
  set m : ℕ := (h k y * z0 k y) / 2 ^ k
  set u : ℕ := (h k y * z0 k y) % 2 ^ k
  set V : ℕ := (u * m) / 2 ^ k
  set rho : ℕ := (z0 k y * m) % 2 ^ k
  set R : ℕ := (h k y * rho) / 2 ^ k
  set D : ℕ := Nat.dist m (2 ^ (2 * k - 1))
  have hdecomp : h k y * z1 k y = m * (2 ^ (2 * k) - m) + (2 ^ k * u - V) + R := by
    simpa [m, u, V, R] using (hz1_quad_correction (k := k) (y := y) hy hyW)
  have hm_lt : m < 2 ^ (2 * k) := by
    simpa [m] using (m_lt_pow (k := k) (y := y) hy hyW)
  have hquad : m * (2 ^ (2 * k) - m) = 2 ^ (W k - 2) - D ^ 2 := by
    have hk1 : 1 ≤ k := by omega
    simpa [m, D] using (quad_eq_sub_dist_sq (k := k) (m := m) hk1 (le_of_lt hm_lt))
  have hcorr : 2 ^ k * u - V < h k y + D ^ 2 := by
    simpa [m, u, V, D] using (correction_lt_h_add_sq (k := k) (y := y) hk hy hyW)
  have hRlt : R < h k y := by
    simpa [m, rho, R] using (R_lt_h (k := k) (y := y) hy hyW)
  have hDlt : D < 2 ^ k := by
    simpa [m, D] using (m_dist_lt_pow (k := k) (y := y) hk hy hyW)
  have hDsq_le : D ^ 2 ≤ 2 ^ (W k - 2) := by
    have hsmall : D ^ 2 < 2 ^ (2 * k) := by
      cases Nat.eq_zero_or_pos D with
      | inl hzero =>
          rw [hzero, pow_two]
          positivity
      | inr hpos =>
          calc
            D ^ 2 = D * D := by rw [pow_two]
            _ < D * 2 ^ k := Nat.mul_lt_mul_of_pos_left hDlt hpos
            _ < 2 ^ k * 2 ^ k := Nat.mul_lt_mul_of_pos_right hDlt (by positivity)
            _ = 2 ^ (2 * k) := by
                  rw [← pow_add]
                  congr 1
                  omega
    exact le_trans (le_of_lt hsmall) <| by
      exact Nat.pow_le_pow_right (by decide : 0 < 2) (by
        unfold W
        omega)
  calc
    h k y * z1 k y = (2 ^ (W k - 2) - D ^ 2) + (2 ^ k * u - V) + R := by
      rw [hdecomp, hquad]
    _ < (2 ^ (W k - 2) - D ^ 2) + (h k y + D ^ 2) + h k y := by
      omega
    _ = 2 ^ (W k - 2) + 2 * h k y := by
      calc
        (2 ^ (W k - 2) - D ^ 2) + (h k y + D ^ 2) + h k y
            = ((2 ^ (W k - 2) - D ^ 2) + D ^ 2) + (h k y + h k y) := by
                ac_rfl
        _ = 2 ^ (W k - 2) + (h k y + h k y) := by rw [Nat.sub_add_cancel hDsq_le]
        _ = 2 ^ (W k - 2) + 2 * h k y := by rw [two_mul]

lemma hz1_dist_lt_two_h {k y : ℕ} (hk : 3 ≤ k) (hy : 0 < y) (hyW : y < 2 ^ (W k)) :
    Nat.dist (h k y * z1 k y) (2 ^ (W k - 2)) < 2 * h k y := by
  set m : ℕ := (h k y * z0 k y) / 2 ^ k
  set D : ℕ := Nat.dist m (2 ^ (2 * k - 1))
  have hlow : 2 ^ (W k - 2) - D ^ 2 ≤ h k y * z1 k y := by
    simpa [m, D] using (hz1_lower_bound (k := k) (y := y) hk hy hyW)
  have hupp : h k y * z1 k y < 2 ^ (W k - 2) + 2 * h k y := by
    exact hz1_upper_bound (k := k) (y := y) hk hy hyW
  have hDlt : D < 2 ^ k := by
    simpa [m, D] using (m_dist_lt_pow (k := k) (y := y) hk hy hyW)
  have hDsq_lt : D ^ 2 < 2 * h k y := by
    have hsmall : D ^ 2 < 2 ^ (2 * k) := by
      cases Nat.eq_zero_or_pos D with
      | inl hzero =>
          rw [hzero, pow_two]
          positivity
      | inr hpos =>
          calc
            D ^ 2 = D * D := by rw [pow_two]
            _ < D * 2 ^ k := Nat.mul_lt_mul_of_pos_left hDlt hpos
            _ < 2 ^ k * 2 ^ k := Nat.mul_lt_mul_of_pos_right hDlt (by positivity)
            _ = 2 ^ (2 * k) := by
                  rw [← pow_add]
                  congr 1
                  omega
    have hh := h_bounds (k := k) (y := y) hy hyW
    calc
      D ^ 2 < 2 ^ (2 * k) := hsmall
      _ ≤ 2 * h k y := by
        have hpow : 2 ^ (2 * k) = 2 * 2 ^ (2 * k - 1) := by
          rw [show 2 * k = 1 + (2 * k - 1) by omega, pow_add]
          norm_num
        rw [hpow]
        exact Nat.mul_le_mul_left _ hh.1
  have hleft : 2 ^ (W k - 2) < h k y * z1 k y + 2 * h k y := by
    omega
  have hright : h k y * z1 k y < 2 ^ (W k - 2) + 2 * h k y := hupp
  by_cases hhz1 : h k y * z1 k y ≤ 2 ^ (W k - 2)
  · rw [Nat.dist_eq_sub_of_le hhz1]
    omega
  · have hTle : 2 ^ (W k - 2) ≤ h k y * z1 k y := le_of_lt (lt_of_not_ge hhz1)
    rw [Nat.dist_eq_sub_of_le_right hTle]
    omega

lemma z1_lt_pow_half_add_two {k y : ℕ} (hk : 3 ≤ k) (hy : 0 < y) (hyW : y < 2 ^ (W k)) :
    z1 k y < 2 ^ (2 * k - 1) + 2 := by
  have hh := h_bounds (k := k) (y := y) hy hyW
  have hhpos : 0 < h k y := lt_of_lt_of_le (by positivity : 0 < 2 ^ (2 * k - 1)) hh.1
  have hupp : h k y * z1 k y < 2 ^ (W k - 2) + 2 * h k y := by
    exact hz1_upper_bound (k := k) (y := y) hk hy hyW
  have hbound : 2 ^ (W k - 2) + 2 * h k y ≤ h k y * (2 ^ (2 * k - 1) + 2) := by
    calc
      2 ^ (W k - 2) + 2 * h k y ≤ h k y * 2 ^ (2 * k - 1) + 2 * h k y := by
        have hpow : 2 ^ (W k - 2) = 2 ^ (2 * k - 1) * 2 ^ (2 * k - 1) := by
          rw [show W k - 2 = (2 * k - 1) + (2 * k - 1) by
            unfold W
            omega, pow_add]
        rw [hpow]
        exact Nat.add_le_add_right (Nat.mul_le_mul_right _ hh.1) _
      _ = h k y * 2 ^ (2 * k - 1) + h k y * 2 := by ac_rfl
      _ = h k y * (2 ^ (2 * k - 1) + 2) := by rw [Nat.mul_add]
  have hupp' : z1 k y * h k y < (2 ^ (2 * k - 1) + 2) * h k y := by
    simpa [Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using lt_of_lt_of_le hupp hbound
  exact (Nat.mul_lt_mul_right hhpos).1 hupp'

lemma z1_lt_tPrime_add_two {k y : ℕ} (hk : 3 ≤ k) (hy : 0 < y) (hyW : y < 2 ^ (W k)) :
    (z1 k y : ℚ) < (2 ^ (W k - 2) : ℚ) / h k y + 2 := by
  have hh := h_bounds (k := k) (y := y) hy hyW
  have hhpos_nat : 0 < h k y := lt_of_lt_of_le (by positivity : 0 < 2 ^ (2 * k - 1)) hh.1
  have hupp_nat : h k y * z1 k y < 2 ^ (W k - 2) + 2 * h k y := by
    exact hz1_upper_bound (k := k) (y := y) hk hy hyW
  have hhpos : (0 : ℚ) < h k y := by
    exact_mod_cast hhpos_nat
  have hupp : (z1 k y : ℚ) * h k y < (2 ^ (W k - 2) : ℚ) + 2 * h k y := by
    have : (((h k y * z1 k y : ℕ) : ℚ)) < ((2 ^ (W k - 2) + 2 * h k y : ℕ) : ℚ) := by
      exact_mod_cast hupp_nat
    simpa [Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat, mul_comm, mul_left_comm, mul_assoc] using this
  have hupp' : (z1 k y : ℚ) * h k y < ((2 ^ (W k - 2) : ℚ) / h k y + 2) * h k y := by
    have hdiv : ((2 ^ (W k - 2) : ℚ) / h k y + 2) * h k y =
        (2 ^ (W k - 2) : ℚ) + 2 * h k y := by
      field_simp [hhpos.ne']
    rw [hdiv]
    exact hupp
  exact lt_of_mul_lt_mul_right hupp' hhpos.le

lemma tPrime_lt_z1_add_two {k y : ℕ} (hk : 3 ≤ k) (hy : 0 < y) (hyW : y < 2 ^ (W k)) :
    (2 ^ (W k - 2) : ℚ) / h k y < (z1 k y : ℚ) + 2 := by
  have hh := h_bounds (k := k) (y := y) hy hyW
  have hhpos_nat : 0 < h k y := lt_of_lt_of_le (by positivity : 0 < 2 ^ (2 * k - 1)) hh.1
  have hdist : Nat.dist (h k y * z1 k y) (2 ^ (W k - 2)) < 2 * h k y := by
    exact hz1_dist_lt_two_h (k := k) (y := y) hk hy hyW
  have hlow_nat : 2 ^ (W k - 2) < h k y * z1 k y + 2 * h k y := by
    exact lt_of_le_of_lt (Nat.dist_tri_left _ _) (by simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using Nat.add_lt_add_right hdist (h k y * z1 k y))
  have hhpos : (0 : ℚ) < h k y := by
    exact_mod_cast hhpos_nat
  have hlow : (2 ^ (W k - 2) : ℚ) < ((z1 k y : ℚ) + 2) * h k y := by
    have : (2 ^ (W k - 2) : ℚ) < ((h k y * z1 k y + 2 * h k y : ℕ) : ℚ) := by
      exact_mod_cast hlow_nat
    have : (2 ^ (W k - 2) : ℚ) < (h k y : ℚ) * z1 k y + ((h k y : ℚ) + h k y) := by
      simpa [Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat, two_mul, mul_comm, mul_left_comm, mul_assoc] using this
    have hrew : (h k y : ℚ) * z1 k y + ((h k y : ℚ) + h k y) = ((z1 k y : ℚ) + 2) * h k y := by
      ring
    exact hrew ▸ this
  have hlow' : ((2 ^ (W k - 2) : ℚ) / h k y) * h k y < ((z1 k y : ℚ) + 2) * h k y := by
    simpa [hhpos.ne'] using hlow
  exact lt_of_mul_lt_mul_right hlow' hhpos.le

lemma tPrime_sub_tRat_nonneg {k y : ℕ} (hy : 0 < y) (hyW : y < 2 ^ (W k)) :
    0 ≤ (2 ^ (W k - 2) : ℚ) / h k y - (2 ^ (W k + 2 * k - 2) : ℚ) / x k y := by
  set s : ℕ := x k y % 2 ^ (2 * k)
  have hh := h_bounds (k := k) (y := y) hy hyW
  have hx := x_bounds (k := k) (y := y) hy hyW
  have hhpos_nat : 0 < h k y := lt_of_lt_of_le (by positivity : 0 < 2 ^ (2 * k - 1)) hh.1
  have hxpos_nat : 0 < x k y := lt_of_lt_of_le (by positivity : 0 < 2 ^ (W k - 1)) hx.1
  have hhpos : (0 : ℚ) < h k y := by
    exact_mod_cast hhpos_nat
  have hxpos : (0 : ℚ) < x k y := by
    exact_mod_cast hxpos_nat
  have hxdecomp : x k y = 2 ^ (2 * k) * h k y + s := by
    dsimp [s]
    simpa [h, Nat.mul_comm, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using
      (Nat.div_add_mod' (x k y) (2 ^ (2 * k))).symm
  have hnum_nat :
      2 ^ (W k - 2) * x k y =
        2 ^ (W k + 2 * k - 2) * h k y + 2 ^ (W k - 2) * s := by
    calc
      2 ^ (W k - 2) * x k y = 2 ^ (W k - 2) * (2 ^ (2 * k) * h k y + s) := by rw [hxdecomp]
      _ = 2 ^ (W k - 2) * (2 ^ (2 * k) * h k y) + 2 ^ (W k - 2) * s := by rw [Nat.mul_add]
      _ = (2 ^ (W k - 2) * 2 ^ (2 * k)) * h k y + 2 ^ (W k - 2) * s := by ac_rfl
      _ = 2 ^ (W k + 2 * k - 2) * h k y + 2 ^ (W k - 2) * s := by
            rw [← pow_add]
            have hExp : (W k - 2) + 2 * k = W k + 2 * k - 2 := by
              unfold W
              omega
            simp [hExp]
  have hnum : (2 ^ (W k - 2) : ℚ) * x k y =
      (2 ^ (W k + 2 * k - 2) : ℚ) * h k y + (2 ^ (W k - 2) : ℚ) * s := by
    exact_mod_cast hnum_nat
  have hmul_eq :
      ((2 ^ (W k - 2) : ℚ) / h k y - (2 ^ (W k + 2 * k - 2) : ℚ) / x k y) *
          ((h k y : ℚ) * x k y) =
        (2 ^ (W k - 2) : ℚ) * s := by
    field_simp [hhpos.ne', hxpos.ne']
    nlinarith [hnum]
  have hmul : 0 ≤
      ((2 ^ (W k - 2) : ℚ) / h k y - (2 ^ (W k + 2 * k - 2) : ℚ) / x k y) *
        ((h k y : ℚ) * x k y) := by
    rw [hmul_eq]
    positivity
  exact nonneg_of_mul_nonneg_left hmul (mul_pos hhpos hxpos)

lemma tPrime_sub_tRat_lt_one {k y : ℕ} (hy : 0 < y) (hyW : y < 2 ^ (W k)) :
    (2 ^ (W k - 2) : ℚ) / h k y - (2 ^ (W k + 2 * k - 2) : ℚ) / x k y < 1 := by
  set s : ℕ := x k y % 2 ^ (2 * k)
  have hh := h_bounds (k := k) (y := y) hy hyW
  have hx := x_bounds (k := k) (y := y) hy hyW
  have hhpos_nat : 0 < h k y := lt_of_lt_of_le (by positivity : 0 < 2 ^ (2 * k - 1)) hh.1
  have hxpos_nat : 0 < x k y := lt_of_lt_of_le (by positivity : 0 < 2 ^ (W k - 1)) hx.1
  have hhpos : (0 : ℚ) < h k y := by
    exact_mod_cast hhpos_nat
  have hxpos : (0 : ℚ) < x k y := by
    exact_mod_cast hxpos_nat
  have hxdecomp : x k y = 2 ^ (2 * k) * h k y + s := by
    dsimp [s]
    simpa [h, Nat.mul_comm, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using
      (Nat.div_add_mod' (x k y) (2 ^ (2 * k))).symm
  have hnum_nat :
      2 ^ (W k - 2) * x k y =
        2 ^ (W k + 2 * k - 2) * h k y + 2 ^ (W k - 2) * s := by
    calc
      2 ^ (W k - 2) * x k y = 2 ^ (W k - 2) * (2 ^ (2 * k) * h k y + s) := by rw [hxdecomp]
      _ = 2 ^ (W k - 2) * (2 ^ (2 * k) * h k y) + 2 ^ (W k - 2) * s := by rw [Nat.mul_add]
      _ = (2 ^ (W k - 2) * 2 ^ (2 * k)) * h k y + 2 ^ (W k - 2) * s := by ac_rfl
      _ = 2 ^ (W k + 2 * k - 2) * h k y + 2 ^ (W k - 2) * s := by
            rw [← pow_add]
            have hExp : (W k - 2) + 2 * k = W k + 2 * k - 2 := by
              unfold W
              omega
            simp [hExp]
  have hnum : (2 ^ (W k - 2) : ℚ) * x k y =
      (2 ^ (W k + 2 * k - 2) : ℚ) * h k y + (2 ^ (W k - 2) : ℚ) * s := by
    exact_mod_cast hnum_nat
  have hslt : s < 2 ^ (2 * k) := by
    dsimp [s]
    exact Nat.mod_lt _ (by positivity)
  have htop_nat : 2 ^ (W k - 2) * s < h k y * x k y := by
    calc
      2 ^ (W k - 2) * s < 2 ^ (W k - 2) * 2 ^ (2 * k) := Nat.mul_lt_mul_of_pos_left hslt (by positivity)
      _ = 2 ^ (W k + 2 * k - 2) := by
            rw [← pow_add]
            have hExp : (W k - 2) + 2 * k = W k + 2 * k - 2 := by
              unfold W
              omega
            simp [hExp]
      _ = 2 ^ (2 * k - 1) * 2 ^ (W k - 1) := by
            rw [← pow_add]
            congr 1
            unfold W
            omega
      _ ≤ h k y * 2 ^ (W k - 1) := Nat.mul_le_mul_right _ hh.1
      _ ≤ h k y * x k y := Nat.mul_le_mul_left _ hx.1
  have htop : ((2 ^ (W k - 2) * s : ℕ) : ℚ) < (h k y : ℚ) * x k y := by
    exact_mod_cast htop_nat
  have hmul_eq :
      (((2 ^ (W k - 2) : ℚ) / h k y) - ((2 ^ (W k + 2 * k - 2) : ℚ) / x k y)) *
          ((h k y : ℚ) * x k y) =
        (2 ^ (W k - 2) : ℚ) * s := by
    field_simp [hhpos.ne', hxpos.ne']
    nlinarith [hnum]
  have hmul :
      (((2 ^ (W k - 2) : ℚ) / h k y) - ((2 ^ (W k + 2 * k - 2) : ℚ) / x k y)) *
          ((h k y : ℚ) * x k y) <
        1 * ((h k y : ℚ) * x k y) := by
    rw [hmul_eq]
    simpa using htop
  nlinarith [hmul, hhpos, hxpos]

lemma xz1_middle_dist_lt_two_x {k y : ℕ} (hk : 3 ≤ k) (hy : 0 < y) (hyW : y < 2 ^ (W k))
    (hyk : bw y ≤ 2 * k) :
    Nat.dist (x k y * z1 k y) (2 ^ (W k + 2 * k - 2)) < 2 * x k y := by
  have hxeq : x k y = 2 ^ (2 * k) * h k y := x_middle_from_h hyk
  have hdist : Nat.dist (h k y * z1 k y) (2 ^ (W k - 2)) < 2 * h k y := by
    exact hz1_dist_lt_two_h (k := k) (y := y) hk hy hyW
  rw [hxeq, show 2 ^ (W k + 2 * k - 2) = 2 ^ (2 * k) * 2 ^ (W k - 2) by
    rw [← pow_add]
    congr 1
    unfold W
    omega]
  rw [show 2 ^ (2 * k) * h k y * z1 k y = 2 ^ (2 * k) * (h k y * z1 k y) by ac_rfl]
  rw [Nat.dist_mul_left]
  calc
    2 ^ (2 * k) * Nat.dist (h k y * z1 k y) (2 ^ (W k - 2))
        < 2 ^ (2 * k) * (2 * h k y) := Nat.mul_lt_mul_of_pos_left hdist (by positivity)
    _ = 2 * (2 ^ (2 * k) * h k y) := by ac_rfl

lemma xz1_high_dist_lt_three_x {k y : ℕ} (hk : 3 ≤ k) (hy : 0 < y) (hyW : y < 2 ^ (W k)) :
    Nat.dist (x k y * z1 k y) (2 ^ (W k + 2 * k - 2)) < 3 * x k y := by
  have hx := x_bounds (k := k) (y := y) hy hyW
  have hxpos_nat : 0 < x k y := lt_of_lt_of_le (by positivity : 0 < 2 ^ (W k - 1)) hx.1
  have hxpos : (0 : ℚ) < x k y := by
    exact_mod_cast hxpos_nat
  have hz1_up : (z1 k y : ℚ) < (2 ^ (W k - 2) : ℚ) / h k y + 2 := by
    exact z1_lt_tPrime_add_two (k := k) (y := y) hk hy hyW
  have ht_up : (2 ^ (W k - 2) : ℚ) / h k y < (2 ^ (W k + 2 * k - 2) : ℚ) / x k y + 1 := by
    have hnonneg := tPrime_sub_tRat_nonneg (k := k) (y := y) hy hyW
    have hltone := tPrime_sub_tRat_lt_one (k := k) (y := y) hy hyW
    linarith
  have hz1_up' : (z1 k y : ℚ) < (2 ^ (W k + 2 * k - 2) : ℚ) / x k y + 3 := by
    linarith
  have ht_low : (2 ^ (W k + 2 * k - 2) : ℚ) / x k y < (z1 k y : ℚ) + 3 := by
    have hnonneg := tPrime_sub_tRat_nonneg (k := k) (y := y) hy hyW
    have htprime := tPrime_lt_z1_add_two (k := k) (y := y) hk hy hyW
    linarith
  have hup : x k y * z1 k y < 2 ^ (W k + 2 * k - 2) + 3 * x k y := by
    have hupQ :
        (x k y : ℚ) * z1 k y < (x k y : ℚ) * ((2 ^ (W k + 2 * k - 2) : ℚ) / x k y + 3) := by
      exact mul_lt_mul_of_pos_left hz1_up' hxpos
    have hupQ' :
        ((x k y * z1 k y : ℕ) : ℚ) < ((2 ^ (W k + 2 * k - 2) + 3 * x k y : ℕ) : ℚ) := by
      have hrew : (x k y : ℚ) * ((2 ^ (W k + 2 * k - 2) : ℚ) / x k y + 3) =
          (2 ^ (W k + 2 * k - 2) : ℚ) + 3 * x k y := by
        field_simp [hxpos.ne']
      calc
        ((x k y * z1 k y : ℕ) : ℚ) < (x k y : ℚ) * ((2 ^ (W k + 2 * k - 2) : ℚ) / x k y + 3) := by
          simpa [Nat.cast_mul, Nat.cast_ofNat, mul_comm, mul_left_comm, mul_assoc] using hupQ
        _ = (2 ^ (W k + 2 * k - 2) : ℚ) + 3 * x k y := hrew
        _ = ((2 ^ (W k + 2 * k - 2) + 3 * x k y : ℕ) : ℚ) := by
          norm_num [Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat]
    exact_mod_cast hupQ'
  have hlow : 2 ^ (W k + 2 * k - 2) < x k y * z1 k y + 3 * x k y := by
    have hlowQ :
        (((2 ^ (W k + 2 * k - 2)) : ℕ) : ℚ) <
          (((x k y * z1 k y + 3 * x k y) : ℕ) : ℚ) := by
      have hlowQ0 :
          ((2 ^ (W k + 2 * k - 2) : ℚ) / x k y) * x k y < ((z1 k y : ℚ) + 3) * x k y := by
        have := mul_lt_mul_of_pos_right ht_low hxpos
        simpa [mul_comm, mul_left_comm, mul_assoc] using this
      have hrew : ((2 ^ (W k + 2 * k - 2) : ℚ) / x k y) * x k y = (2 ^ (W k + 2 * k - 2) : ℚ) := by
        field_simp [hxpos.ne']
      calc
        ((2 ^ (W k + 2 * k - 2) : ℕ) : ℚ) = ((2 ^ (W k + 2 * k - 2) : ℚ) / x k y) * x k y := by
          simpa using hrew.symm
        _ < ((z1 k y : ℚ) + 3) * x k y := hlowQ0
        _ = ((x k y * z1 k y + 3 * x k y : ℕ) : ℚ) := by
          norm_num [Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat, add_mul, mul_comm, mul_left_comm, mul_assoc]
          ring
    exact_mod_cast hlowQ
  by_cases hle : x k y * z1 k y ≤ 2 ^ (W k + 2 * k - 2)
  · rw [Nat.dist_eq_sub_of_le hle]
    omega
  · have hge : 2 ^ (W k + 2 * k - 2) ≤ x k y * z1 k y := le_of_lt (lt_of_not_ge hle)
    rw [Nat.dist_eq_sub_of_le_right hge]
    omega

lemma xz1_middle_lt_double_target {k y : ℕ} (hk : 3 ≤ k) (hy : 0 < y) (hyW : y < 2 ^ (W k))
    (hyk : bw y ≤ 2 * k) :
    x k y * z1 k y < 2 ^ (W k + 2 * k - 1) := by
  let S : ℕ := 2 ^ (W k + 2 * k - 2)
  have hdist : Nat.dist (x k y * z1 k y) (2 ^ (W k + 2 * k - 2)) < 2 * x k y := by
    exact xz1_middle_dist_lt_two_x (k := k) (y := y) hk hy hyW hyk
  have hx := x_bounds (k := k) (y := y) hy hyW
  have hsmall : 2 * x k y ≤ 2 ^ (W k + 2 * k - 2) := by
    exact le_of_lt <| calc
      2 * x k y < 2 * 2 ^ (W k) := Nat.mul_lt_mul_of_pos_left hx.2 (by positivity)
      _ = 2 ^ (W k + 1) := by
            rw [show W k + 1 = 1 + W k by omega, pow_add]
            norm_num
      _ ≤ 2 ^ (W k + 2 * k - 2) := Nat.pow_le_pow_right (by decide : 0 < 2) (by
            unfold W
            omega)
  have hup0 : x k y * z1 k y ≤ S + Nat.dist (x k y * z1 k y) S := by
    simpa [S] using (Nat.dist_tri_right' (x k y * z1 k y) (2 ^ (W k + 2 * k - 2)))
  have hup1 : S + Nat.dist (x k y * z1 k y) S < S + 2 * x k y := by
    simpa [S] using Nat.add_lt_add_left hdist S
  have hup : x k y * z1 k y < S + 2 * x k y := lt_of_le_of_lt hup0 hup1
  calc
    x k y * z1 k y < S + 2 * x k y := hup
    _ ≤ S + S := Nat.add_le_add_left hsmall _
    _ = 2 ^ (W k + 2 * k - 1) := by
          dsimp [S]
          have hpow : 2 ^ (W k + 2 * k - 1) = 2 ^ (W k + 2 * k - 2) + 2 ^ (W k + 2 * k - 2) := by
            rw [show W k + 2 * k - 1 = (W k + 2 * k - 2) + 1 by omega, pow_add]
            norm_num
            rw [Nat.mul_comm, two_mul]
          exact hpow.symm

lemma xz1_high_lt_double_target {k y : ℕ} (hk : 3 ≤ k) (hy : 0 < y) (hyW : y < 2 ^ (W k)) :
    x k y * z1 k y < 2 ^ (W k + 2 * k - 1) := by
  let S : ℕ := 2 ^ (W k + 2 * k - 2)
  have hdist : Nat.dist (x k y * z1 k y) (2 ^ (W k + 2 * k - 2)) < 3 * x k y := by
    exact xz1_high_dist_lt_three_x (k := k) (y := y) hk hy hyW
  have hx := x_bounds (k := k) (y := y) hy hyW
  have hsmall : 3 * x k y ≤ 2 ^ (W k + 2 * k - 2) := by
    exact le_of_lt <| calc
      3 * x k y < 3 * 2 ^ (W k) := Nat.mul_lt_mul_of_pos_left hx.2 (by positivity)
      _ ≤ 2 ^ (2 * k - 2) * 2 ^ (W k) := by
            have hkpow : 3 ≤ 2 ^ (2 * k - 2) := by
              have hpow : 2 ^ 2 ≤ 2 ^ (2 * k - 2) := by
                exact Nat.pow_le_pow_right (by decide : 0 < 2) (by omega)
              have : 4 ≤ 2 ^ (2 * k - 2) := by simpa using hpow
              omega
            exact Nat.mul_le_mul_right _ hkpow
      _ = 2 ^ (W k + 2 * k - 2) := by
            rw [← pow_add]
            congr 1
            omega
  have hup0 : x k y * z1 k y ≤ S + Nat.dist (x k y * z1 k y) S := by
    simpa [S] using (Nat.dist_tri_right' (x k y * z1 k y) (2 ^ (W k + 2 * k - 2)))
  have hup1 : S + Nat.dist (x k y * z1 k y) S < S + 3 * x k y := by
    simpa [S] using Nat.add_lt_add_left hdist S
  have hup : x k y * z1 k y < S + 3 * x k y := lt_of_le_of_lt hup0 hup1
  calc
    x k y * z1 k y < S + 3 * x k y := hup
    _ ≤ S + S := Nat.add_le_add_left hsmall _
    _ = 2 ^ (W k + 2 * k - 1) := by
          dsimp [S]
          have hpow : 2 ^ (W k + 2 * k - 1) = 2 ^ (W k + 2 * k - 2) + 2 ^ (W k + 2 * k - 2) := by
            rw [show W k + 2 * k - 1 = (W k + 2 * k - 2) + 1 by omega, pow_add]
            norm_num
            rw [Nat.mul_comm, two_mul]
          exact hpow.symm

end ReciprocalMiddle



open Reciprocal
open ReciprocalMiddle
open ReciprocalLowExactCore
open ReciprocalLowExact

lemma quad_center_dist_sq {S T : ℕ} (hT : T < 2 * S) :
    T * (2 * S - T) = S ^ 2 - (Nat.dist T S) ^ 2 := by
  by_cases hTS : T ≤ S
  · have hD : Nat.dist T S = S - T := by
      rw [Nat.dist_eq_sub_of_le hTS]
    have h1 : S - (S - T) = T := by
      exact Nat.sub_sub_self hTS
    have h2 : S + (S - T) = 2 * S - T := by
      omega
    calc
      T * (2 * S - T) = (S - (S - T)) * (S + (S - T)) := by rw [h1, h2]
      _ = S ^ 2 - (S - T) ^ 2 := by
        simp [pow_two, Nat.mul_self_sub_mul_self_eq, Nat.mul_comm]
      _ = S ^ 2 - (Nat.dist T S) ^ 2 := by rw [hD]
  · have hST : S < T := lt_of_not_ge hTS
    have hD : Nat.dist T S = T - S := by
      rw [Nat.dist_eq_sub_of_le_right (le_of_lt hST)]
    have h1 : S + (T - S) = T := by
      exact Nat.add_sub_of_le (le_of_lt hST)
    have h2 : S - (T - S) = 2 * S - T := by
      omega
    calc
      T * (2 * S - T) = (S + (T - S)) * (S - (T - S)) := by rw [h1, h2]
      _ = S ^ 2 - (T - S) ^ 2 := by
        simp [pow_two, Nat.mul_self_sub_mul_self_eq]
      _ = S ^ 2 - (Nat.dist T S) ^ 2 := by rw [hD]

lemma yz1_middle_dist_lt_two_y {k y : ℕ} (hk : 3 ≤ k) (hy : 0 < y) (hyW : y < 2 ^ (W k))
    (hyk : bw y ≤ 2 * k) :
    Nat.dist (y * z1 k y) (2 ^ (2 * k + bw y - 2)) < 2 * y := by
  have hbwW : bw y ≤ W k := bw_le_of_lt_pow (Nat.ne_of_gt hy) hyW
  set c : ℕ := 2 ^ (W k - bw y)
  have hcpos : 0 < c := by
    dsimp [c]
    positivity
  have hdist :
      Nat.dist (x k y * z1 k y) (2 ^ (W k + 2 * k - 2)) < 2 * x k y := by
    exact xz1_middle_dist_lt_two_x (k := k) (y := y) hk hy hyW hyk
  have hdist' :
      c * Nat.dist (y * z1 k y) (2 ^ (2 * k + bw y - 2)) < c * (2 * y) := by
    have hpow :
        2 ^ (W k + 2 * k - 2) = c * 2 ^ (2 * k + bw y - 2) := by
      dsimp [c]
      rw [show W k + 2 * k - 2 = (W k - bw y) + (2 * k + bw y - 2) by
        omega, pow_add]
    have hx :
        x k y = c * y := by
      dsimp [c]
      unfold x
      ac_rfl
    rw [hx, hpow, show c * y * z1 k y = c * (y * z1 k y) by ac_rfl, Nat.dist_mul_left] at hdist
    simpa [Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm] using hdist
  exact (Nat.mul_lt_mul_left hcpos).1 hdist'

lemma yz1_high_dist_lt_three_y {k y : ℕ} (hk : 3 ≤ k) (hy : 0 < y) (hyW : y < 2 ^ (W k)) :
    Nat.dist (y * z1 k y) (2 ^ (2 * k + bw y - 2)) < 3 * y := by
  have hbwW : bw y ≤ W k := bw_le_of_lt_pow (Nat.ne_of_gt hy) hyW
  set c : ℕ := 2 ^ (W k - bw y)
  have hcpos : 0 < c := by
    dsimp [c]
    positivity
  have hdist :
      Nat.dist (x k y * z1 k y) (2 ^ (W k + 2 * k - 2)) < 3 * x k y := by
    exact xz1_high_dist_lt_three_x (k := k) (y := y) hk hy hyW
  have hdist' :
      c * Nat.dist (y * z1 k y) (2 ^ (2 * k + bw y - 2)) < c * (3 * y) := by
    have hpow :
        2 ^ (W k + 2 * k - 2) = c * 2 ^ (2 * k + bw y - 2) := by
      dsimp [c]
      rw [show W k + 2 * k - 2 = (W k - bw y) + (2 * k + bw y - 2) by
        omega, pow_add]
    have hx :
        x k y = c * y := by
      dsimp [c]
      unfold x
      ac_rfl
    rw [hx, hpow, show c * y * z1 k y = c * (y * z1 k y) by ac_rfl, Nat.dist_mul_left] at hdist
    simpa [Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm] using hdist
  exact (Nat.mul_lt_mul_left hcpos).1 hdist'

lemma yz1_middle_lt_double_target {k y : ℕ} (hk : 3 ≤ k) (hy : 0 < y) (hyW : y < 2 ^ (W k))
    (hyk : bw y ≤ 2 * k) :
    y * z1 k y < 2 ^ (2 * k + bw y - 1) := by
  have hbwW : bw y ≤ W k := bw_le_of_lt_pow (Nat.ne_of_gt hy) hyW
  set c : ℕ := 2 ^ (W k - bw y)
  have hcpos : 0 < c := by
    dsimp [c]
    positivity
  have hlt : x k y * z1 k y < 2 ^ (W k + 2 * k - 1) := by
    exact xz1_middle_lt_double_target (k := k) (y := y) hk hy hyW hyk
  have hlt' : c * (y * z1 k y) < c * 2 ^ (2 * k + bw y - 1) := by
    have hpow :
        2 ^ (W k + 2 * k - 1) = c * 2 ^ (2 * k + bw y - 1) := by
      dsimp [c]
      rw [show W k + 2 * k - 1 = (W k - bw y) + (2 * k + bw y - 1) by
        omega, pow_add]
    have hx :
        x k y = c * y := by
      dsimp [c]
      unfold x
      ac_rfl
    rw [hx, hpow, show c * y * z1 k y = c * (y * z1 k y) by ac_rfl] at hlt
    simpa [Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm] using hlt
  exact (Nat.mul_lt_mul_left hcpos).1 hlt'

lemma yz1_high_lt_double_target {k y : ℕ} (hk : 3 ≤ k) (hy : 0 < y) (hyW : y < 2 ^ (W k)) :
    y * z1 k y < 2 ^ (2 * k + bw y - 1) := by
  have hbwW : bw y ≤ W k := bw_le_of_lt_pow (Nat.ne_of_gt hy) hyW
  set c : ℕ := 2 ^ (W k - bw y)
  have hcpos : 0 < c := by
    dsimp [c]
    positivity
  have hlt : x k y * z1 k y < 2 ^ (W k + 2 * k - 1) := by
    exact xz1_high_lt_double_target (k := k) (y := y) hk hy hyW
  have hlt' : c * (y * z1 k y) < c * 2 ^ (2 * k + bw y - 1) := by
    have hpow :
        2 ^ (W k + 2 * k - 1) = c * 2 ^ (2 * k + bw y - 1) := by
      dsimp [c]
      rw [show W k + 2 * k - 1 = (W k - bw y) + (2 * k + bw y - 1) by
        omega, pow_add]
    have hx :
        x k y = c * y := by
      dsimp [c]
      unfold x
      ac_rfl
    rw [hx, hpow, show c * y * z1 k y = c * (y * z1 k y) by ac_rfl] at hlt
    simpa [Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm] using hlt
  exact (Nat.mul_lt_mul_left hcpos).1 hlt'

lemma z2_exact_of_bw_le_two_k_add_four {k y : ℕ} (hk : 3 ≤ k) (hb4 : 4 ≤ bw y)
    (hℓ : bw y ≤ 2 * k + 4) :
    z2 k y = 2 ^ (2 * k + 3) * z1 k y - (y * (z1 k y) ^ 2) / 2 ^ (bw y - 4) := by
  unfold z2
  unfold x
  simp only
  have hinner :
      (y * 2 ^ (W k - bw y) * z1 k y) / 2 ^ (2 * k - 4) =
        y * z1 k y * 2 ^ (2 * k + 4 - bw y) := by
    apply Nat.div_eq_of_eq_mul_right (by positivity : 0 < 2 ^ (2 * k - 4))
    rw [show W k - bw y = (2 * k - 4) + (2 * k + 4 - bw y) by
      unfold W
      omega, pow_add]
    ac_rfl
  rw [hinner]
  have hsecond :
      (z1 k y * (y * z1 k y * 2 ^ (2 * k + 4 - bw y))) / 2 ^ (2 * k) =
        (y * (z1 k y) ^ 2) / 2 ^ (bw y - 4) := by
    rw [show z1 k y * (y * z1 k y * 2 ^ (2 * k + 4 - bw y)) =
      (y * (z1 k y) ^ 2) * 2 ^ (2 * k + 4 - bw y) by
      rw [pow_two]
      ac_rfl]
    have hpowk : 2 ^ (2 * k) = 2 ^ (2 * k + 4 - bw y) * 2 ^ (bw y - 4) := by
      have hdecomp : 2 * k + 4 - bw y + (bw y - 4) = 2 * k := by omega
      calc
        2 ^ (2 * k) = 2 ^ (2 * k + 4 - bw y + (bw y - 4)) := by rw [hdecomp]
        _ = 2 ^ (2 * k + 4 - bw y) * 2 ^ (bw y - 4) := by rw [pow_add]
    rw [hpowk]
    rw [← Nat.div_div_eq_div_mul]
    have htmp :
        (y * (z1 k y) ^ 2 * 2 ^ (2 * k + 4 - bw y)) / 2 ^ (2 * k + 4 - bw y) =
          y * (z1 k y) ^ 2 := by
      apply Nat.div_eq_of_eq_mul_right (by positivity : 0 < 2 ^ (2 * k + 4 - bw y))
      ac_rfl
    exact congrArg (fun t => t / 2 ^ (bw y - 4)) htmp
  rw [hsecond]

lemma exact_regime_num_eq {k y : ℕ} (_hk : 3 ≤ k) (hb4 : 4 ≤ bw y) (hℓ : bw y ≤ 2 * k + 4)
    (hTlt : y * z1 k y < 2 ^ (2 * k + bw y - 1)) :
    let S : ℕ := 2 ^ (2 * k + bw y - 2)
    let D : ℕ := Nat.dist (y * z1 k y) S
    2 ^ (W k + 2 * bw y - 4) - D ^ 2 =
      y * (((2 ^ (2 * k + 3) * z1 k y) * 2 ^ (bw y - 4)) - y * (z1 k y) ^ 2) := by
  set S : ℕ := 2 ^ (2 * k + bw y - 2)
  set T : ℕ := y * z1 k y
  set D : ℕ := Nat.dist T S
  change 2 ^ (W k + 2 * bw y - 4) - D ^ 2 =
    y * (((2 ^ (2 * k + 3) * z1 k y) * 2 ^ (bw y - 4)) - y * (z1 k y) ^ 2)
  have hprod :
      y * (((2 ^ (2 * k + 3) * z1 k y) * 2 ^ (bw y - 4)) - y * (z1 k y) ^ 2) =
        T * (2 * S - T) := by
    dsimp [T, S]
    have hyzle : y * z1 k y ≤ 2 ^ (2 * k + bw y - 1) := le_of_lt hTlt
    have hpowterm :
        (2 ^ (2 * k + 3) * z1 k y) * 2 ^ (bw y - 4) =
          z1 k y * 2 ^ (2 * k + bw y - 1) := by
      calc
        (2 ^ (2 * k + 3) * z1 k y) * 2 ^ (bw y - 4)
            = z1 k y * (2 ^ (2 * k + 3) * 2 ^ (bw y - 4)) := by ac_rfl
        _ = z1 k y * 2 ^ ((2 * k + 3) + (bw y - 4)) := by rw [← pow_add]
        _ = z1 k y * 2 ^ (2 * k + bw y - 1) := by
              congr 2
              omega
    calc
      y * (((2 ^ (2 * k + 3) * z1 k y) * 2 ^ (bw y - 4)) - y * (z1 k y) ^ 2)
          = y * (z1 k y * 2 ^ (2 * k + bw y - 1) - y * (z1 k y) ^ 2) := by
              rw [hpowterm]
      _ = y * (z1 k y * (2 ^ (2 * k + bw y - 1) - y * z1 k y)) := by
            rw [show y * (z1 k y) ^ 2 = z1 k y * (y * z1 k y) by
              rw [pow_two]
              ac_rfl]
            rw [← Nat.mul_sub_left_distrib]
      _ = (y * z1 k y) * (2 ^ (2 * k + bw y - 1) - y * z1 k y) := by
            ac_rfl
      _ = T * (2 * S - T) := by
            rw [show 2 ^ (2 * k + bw y - 1) = 2 * 2 ^ (2 * k + bw y - 2) by
              rw [show 2 * k + bw y - 1 = (2 * k + bw y - 2) + 1 by omega, pow_add]
              norm_num
              rw [Nat.mul_comm]]
  have hquad : T * (2 * S - T) = S ^ 2 - D ^ 2 := by
    have : T < 2 * S := by
      dsimp [S, T]
      rw [show 2 * 2 ^ (2 * k + bw y - 2) = 2 ^ (2 * k + bw y - 1) by
        rw [show 2 * k + bw y - 1 = (2 * k + bw y - 2) + 1 by omega, pow_add]
        norm_num
        rw [Nat.mul_comm]]
      exact hTlt
    rw [quad_center_dist_sq this]
  have hSsq : S ^ 2 = 2 ^ (W k + 2 * bw y - 4) := by
    dsimp [S]
    rw [pow_two, ← pow_add]
    unfold W
    congr 1
    omega
  calc
    2 ^ (W k + 2 * bw y - 4) - D ^ 2 = S ^ 2 - D ^ 2 := by rw [hSsq]
    _ = T * (2 * S - T) := hquad.symm
    _ = y * (((2 ^ (2 * k + 3) * z1 k y) * 2 ^ (bw y - 4)) - y * (z1 k y) ^ 2) := hprod.symm

lemma z2_exact_ceilDiv_dist {k y : ℕ} (hk : 3 ≤ k) (hb4 : 4 ≤ bw y) (hℓ : bw y ≤ 2 * k + 4)
    (hTlt : y * z1 k y < 2 ^ (2 * k + bw y - 1)) :
    z2 k y =
      (((2 ^ (W k + 2 * bw y - 4)) - (Nat.dist (y * z1 k y) (2 ^ (2 * k + bw y - 2))) ^ 2) / y) ⌈/⌉
        2 ^ (bw y - 4) := by
  set ell : ℕ := bw y
  set D : ℕ := Nat.dist (y * z1 k y) (2 ^ (2 * k + ell - 2))
  have hm : y * (z1 k y) ^ 2 ≤ (2 ^ (2 * k + 3) * z1 k y) * 2 ^ (ell - 4) := by
    calc
      y * (z1 k y) ^ 2 = z1 k y * (y * z1 k y) := by
        rw [pow_two]
        ac_rfl
      _ ≤ z1 k y * 2 ^ (2 * k + ell - 1) := Nat.mul_le_mul_left _ (le_of_lt hTlt)
      _ = (2 ^ (2 * k + 3) * z1 k y) * 2 ^ (ell - 4) := by
            calc
              z1 k y * 2 ^ (2 * k + ell - 1)
                  = z1 k y * (2 ^ (2 * k + 3) * 2 ^ (ell - 4)) := by
                      rw [show 2 * k + ell - 1 = (2 * k + 3) + (ell - 4) by omega, pow_add]
              _ = (2 ^ (2 * k + 3) * z1 k y) * 2 ^ (ell - 4) := by ac_rfl
  rw [z2_exact_of_bw_le_two_k_add_four hk hb4 hℓ, sub_div_eq_ceilDiv (hn := by positivity) hm]
  have hy : 0 < y := by
    cases y with
    | zero =>
        exfalso
        simp [ell, bw] at hb4
    | succ y => positivity
  have hnum :
      ((2 ^ (2 * k + 3) * z1 k y) * 2 ^ (ell - 4) - y * (z1 k y) ^ 2) =
        (2 ^ (W k + 2 * ell - 4) - D ^ 2) / y := by
    have hnum' :
        2 ^ (W k + 2 * ell - 4) - D ^ 2 =
          y * (((2 ^ (2 * k + 3) * z1 k y) * 2 ^ (ell - 4)) - y * (z1 k y) ^ 2) := by
      have htmp := exact_regime_num_eq (k := k) (y := y) hk hb4 hℓ hTlt
      dsimp [D, ell] at htmp
      simpa using htmp
    symm
    exact Nat.div_eq_of_eq_mul_right hy (by
      exact hnum')
  simpa [ell, D] using congrArg (fun t => t ⌈/⌉ 2 ^ (ell - 4)) hnum

lemma q0_exact_bounds {k y : ℕ} (hk : 3 ≤ k) (hyW : y < 2 ^ (W k))
    (hreg :
      (((k < bw y) ∧ (bw y ≤ 2 * k) ∧ 6 ≤ bw y) ∨
        ((2 * k < bw y) ∧ (bw y ≤ 2 * k + 4) ∧ 8 ≤ bw y))) :
    2 ^ (W k) / y - 1 ≤ q0 k y ∧ q0 k y ≤ 2 ^ (W k) / y := by
  set ell : ℕ := bw y
  set a : ℕ := 2 ^ (ell - 4)
  set P : ℕ := 2 ^ (2 * ell - 4)
  set D : ℕ := Nat.dist (y * z1 k y) (2 ^ (2 * k + ell - 2))
  set M : ℕ := (2 ^ (W k + 2 * ell - 4) - D ^ 2) / y
  set Q : ℕ := 2 ^ (W k) / y
  set s : ℕ := 2 ^ (W k) % y
  have hb4 : 4 ≤ ell := by
    rcases hreg with hmid | hhigh
    · omega
    · omega
  have hℓ : ell ≤ 2 * k + 4 := by
    rcases hreg with hmid | hhigh
    · omega
    · omega
  have hy0 : 0 < y := by
    cases y with
    | zero =>
        have : ¬(((k < bw 0) ∧ (bw 0 ≤ 2 * k) ∧ 6 ≤ bw 0) ∨
            ((2 * k < bw 0) ∧ (bw 0 ≤ 2 * k + 4) ∧ 8 ≤ bw 0)) := by
          simp [bw]
        exact False.elim (this hreg)
    | succ y => positivity
  have hTlt : y * z1 k y < 2 ^ (2 * k + ell - 1) := by
    rcases hreg with hmid | hhigh
    · simpa [ell] using yz1_middle_lt_double_target (k := k) (y := y) hk hy0 hyW hmid.2.1
    · simpa [ell] using yz1_high_lt_double_target (k := k) (y := y) hk hy0 hyW
  have hz2 : z2 k y = M ⌈/⌉ a := by
    simpa [M, D, ell, a] using (z2_exact_ceilDiv_dist (k := k) (y := y) hk hb4 hℓ hTlt)
  have hslt : s < y := by
    dsimp [s]
    exact Nat.mod_lt _ hy0
  have hdivmod : 2 ^ (W k) = Q * y + s := by
    dsimp [Q, s]
    simpa [Nat.mul_comm] using (Nat.div_add_mod' (2 ^ (W k)) y).symm
  have hyltpow : y < 2 ^ ell := by
    simpa [ell] using (bw_bounds (Nat.ne_of_gt hy0)).2
  have hPsplit : P = 2 ^ ell * a := by
    dsimp [P, a]
    rw [show 2 * ell - 4 = ell + (ell - 4) by omega, pow_add]
  have hAleP : a ≤ P := by
    rw [hPsplit]
    simpa [Nat.mul_comm] using
      (Nat.le_mul_of_pos_right a (show 0 < 2 ^ ell by positivity))
  have hpowLeP : 2 ^ ell ≤ P := by
    rw [hPsplit]
    simpa [Nat.mul_comm] using
      (Nat.le_mul_of_pos_right (2 ^ ell) (show 0 < a by positivity))
  have hD2ltP : D ^ 2 < y * P := by
    rcases hreg with hmid | hhigh
    · have hDlt : D < 2 * y := by
        simpa [D, ell] using yz1_middle_dist_lt_two_y (k := k) (y := y) hk hy0 hyW hmid.2.1
      cases Nat.eq_zero_or_pos D with
      | inl hD0 =>
          rw [hD0, pow_two]
          positivity
      | inr hDpos =>
          calc
            D ^ 2 = D * D := by rw [pow_two]
            _ < D * (2 * y) := Nat.mul_lt_mul_of_pos_left hDlt hDpos
            _ < (2 * y) * (2 * y) := Nat.mul_lt_mul_of_pos_right hDlt (by positivity)
            _ = y * (4 * y) := by ring
            _ < y * P := by
                  apply Nat.mul_lt_mul_of_pos_left ?_ hy0
                  calc
                    4 * y < 4 * 2 ^ ell := Nat.mul_lt_mul_of_pos_left hyltpow (by positivity)
                    _ = 2 ^ (ell + 2) := by
                          rw [show ell + 2 = 2 + ell by omega, pow_add]
                          norm_num
                    _ ≤ P := by
                          dsimp [P]
                          exact Nat.pow_le_pow_right (by decide : 0 < 2) (by omega)
    · have hDlt : D < 3 * y := by
        simpa [D, ell] using yz1_high_dist_lt_three_y (k := k) (y := y) hk hy0 hyW
      cases Nat.eq_zero_or_pos D with
      | inl hD0 =>
          rw [hD0, pow_two]
          positivity
      | inr hDpos =>
          calc
            D ^ 2 = D * D := by rw [pow_two]
            _ < D * (3 * y) := Nat.mul_lt_mul_of_pos_left hDlt hDpos
            _ < (3 * y) * (3 * y) := Nat.mul_lt_mul_of_pos_right hDlt (by positivity)
            _ = y * (9 * y) := by ring
            _ < y * P := by
                  apply Nat.mul_lt_mul_of_pos_left ?_ hy0
                  calc
                    9 * y < 9 * 2 ^ ell := Nat.mul_lt_mul_of_pos_left hyltpow (by positivity)
                    _ ≤ 2 ^ (ell - 4) * 2 ^ ell := by
                          have hpow9 : 9 ≤ 2 ^ (ell - 4) := by
                            have : 2 ^ 4 ≤ 2 ^ (ell - 4) := by
                              exact Nat.pow_le_pow_right (by decide : 0 < 2) (by omega)
                            have h16 : 16 ≤ 2 ^ (ell - 4) := by simpa using this
                            omega
                          exact Nat.mul_le_mul_right _ hpow9
                    _ = P := by
                          dsimp [P]
                          rw [← pow_add]
                          congr 1
                          omega
  have hsP : s * P ≤ y * (P - a) := by
    have hsle : s ≤ y - 1 := Nat.le_pred_of_lt hslt
    have hpred : (y - 1) * P = y * P - P := by
      simpa [Nat.pred_eq_sub_one] using (Nat.pred_mul y P)
    have hyPa : y * a ≤ P := by
      rw [hPsplit]
      simpa [Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm] using
        (Nat.mul_le_mul_left a (le_of_lt hyltpow))
    calc
      s * P ≤ (y - 1) * P := Nat.mul_le_mul_right _ hsle
      _ = y * P - P := hpred
      _ ≤ y * P - y * a := Nat.sub_le_sub_left hyPa _
      _ = y * (P - a) := by rw [Nat.mul_sub_left_distrib]
  have hpowWP : 2 ^ (W k + 2 * ell - 4) = 2 ^ (W k) * P := by
    dsimp [P]
    rw [show W k + 2 * ell - 4 = W k + (2 * ell - 4) by omega, pow_add]
  have hMupper : M ≤ Q * P + (P - a) := by
    apply Nat.div_le_of_le_mul
    calc
      2 ^ (W k + 2 * ell - 4) - D ^ 2 ≤ 2 ^ (W k + 2 * ell - 4) := Nat.sub_le _ _
      _ = y * (Q * P) + s * P := by
        rw [hpowWP, hdivmod, Nat.add_mul]
        ac_rfl
      _ ≤ y * (Q * P) + y * (P - a) := Nat.add_le_add_left hsP _
      _ = y * (Q * P + (P - a)) := by rw [Nat.mul_add]
  have hMdiv : y ∣ 2 ^ (W k + 2 * ell - 4) - D ^ 2 := by
    refine ⟨((2 ^ (2 * k + 3) * z1 k y) * 2 ^ (ell - 4)) - y * (z1 k y) ^ 2, ?_⟩
    simpa [D, ell] using (exact_regime_num_eq (k := k) (y := y) hk hb4 hℓ hTlt)
  have hMlower : Q * P - P < M := by
    apply (Nat.lt_div_iff_mul_lt' hMdiv _).2
    have hQpos : 0 < Q := by
      dsimp [Q]
      exact Nat.div_pos (Nat.le_of_lt hyW) hy0
    have hQP : P ≤ Q * P := by
      simpa [Nat.mul_comm] using (Nat.le_mul_of_pos_right P hQpos)
    have hleD : D ^ 2 ≤ y * (Q * P) + s * P := by
      exact le_trans (le_of_lt hD2ltP) (by
        have : y * P ≤ y * (Q * P) + s * P := by
          exact le_trans (Nat.mul_le_mul_left _ hQP) (Nat.le_add_right _ _)
        exact this)
    have hraw_add : y * (Q * P - P) + D ^ 2 < y * (Q * P) + s * P := by
      have hstep1 : y * (Q * P - P) + D ^ 2 < y * (Q * P - P) + y * P := by
        exact Nat.add_lt_add_left hD2ltP _
      have hstep2 : y * (Q * P - P) + y * P = y * (Q * P) := by
        rw [Nat.mul_sub_left_distrib]
        exact Nat.sub_add_cancel (Nat.mul_le_mul_left _ hQP)
      have hstep3 : y * (Q * P) ≤ y * (Q * P) + s * P := Nat.le_add_right _ _
      have hstep1' : y * (Q * P - P) + D ^ 2 < y * (Q * P) := by
        simpa [hstep2] using hstep1
      exact lt_of_lt_of_le hstep1' hstep3
    have hraw : y * (Q * P - P) < y * (Q * P) + s * P - D ^ 2 := by
      have hsubadd : (y * (Q * P) + s * P - D ^ 2) + D ^ 2 = y * (Q * P) + s * P := by
        exact Nat.sub_add_cancel hleD
      have htmp : y * (Q * P - P) + D ^ 2 < (y * (Q * P) + s * P - D ^ 2) + D ^ 2 := by
        simpa [hsubadd] using hraw_add
      exact Nat.lt_of_add_lt_add_right htmp
    calc
      y * (Q * P - P) < y * (Q * P) + s * P - D ^ 2 := hraw
      _ = 2 ^ (W k + 2 * ell - 4) - D ^ 2 := by
        rw [hpowWP, hdivmod, Nat.add_mul]
        ac_rfl
  have hz2lower : (Q - 1) * 2 ^ ell < z2 k y := by
    rw [hz2]
    apply lt_ceilDiv_of_mul_lt (a := a) (by positivity)
    calc
      a * ((Q - 1) * 2 ^ ell) = (Q - 1) * P := by
        rw [hPsplit]
        ac_rfl
      _ = Q * P - P := by
        simpa [Nat.pred_eq_sub_one] using (Nat.pred_mul Q P)
      _ < M := hMlower
  have hz2upper : z2 k y < (Q + 1) * 2 ^ ell := by
    have hmul : a * z2 k y < M + a := by
      simpa [hz2, Nat.mul_comm] using (mul_ceilDiv_lt_add (a := a) (b := M) (by positivity))
    have hbound : M + a ≤ (Q + 1) * P := by
      have hsumPa : (Q * P + (P - a)) + a = Q * P + P := by
        calc
          (Q * P + (P - a)) + a = Q * P + ((P - a) + a) := by ac_rfl
          _ = Q * P + P := by rw [Nat.sub_add_cancel hAleP]
      calc
        M + a ≤ (Q * P + (P - a)) + a := Nat.add_le_add_right hMupper a
        _ = Q * P + P := hsumPa
        _ = (Q + 1) * P := by rw [Nat.add_mul, one_mul]
    have hmul' : a * z2 k y < a * ((Q + 1) * 2 ^ ell) := by
      calc
        a * z2 k y < M + a := hmul
        _ ≤ (Q + 1) * P := hbound
        _ = a * ((Q + 1) * 2 ^ ell) := by
          rw [hPsplit]
          ac_rfl
    exact (Nat.mul_lt_mul_left (by positivity : 0 < a)).1 hmul'
  have hq0lower : Q - 1 ≤ q0 k y := by
    unfold q0
    refine (Nat.le_div_iff_mul_le (by positivity : 0 < 2 ^ ell)).2 ?_
    exact le_of_lt (by simpa [ell, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using hz2lower)
  have hq0upper : q0 k y ≤ Q := by
    unfold q0
    exact Nat.le_of_lt_succ ((Nat.div_lt_iff_lt_mul (by positivity : 0 < 2 ^ ell)).2
      (by simpa [ell, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using hz2upper))
  simpa [Q] using And.intro hq0lower hq0upper

theorem exact_regime_proved {k y : ℕ} (hk : 3 ≤ k) (hyW : y < 2 ^ (W k))
    (hreg :
      (((k < bw y) ∧ (bw y ≤ 2 * k) ∧ 6 ≤ bw y) ∨
        ((2 * k < bw y) ∧ (bw y ≤ 2 * k + 4) ∧ 8 ≤ bw y)))
    (hpow : ¬ ∃ m : ℕ, y = 2 ^ m) :
    Good k y := by
  have hbounds := q0_exact_bounds (k := k) (y := y) hk hyW hreg
  set Q : ℕ := 2 ^ (W k) / y
  have hy0 : 0 < y := by
    cases y with
    | zero =>
        have : ¬(((k < bw 0) ∧ (bw 0 ≤ 2 * k) ∧ 6 ≤ bw 0) ∨
            ((2 * k < bw 0) ∧ (bw 0 ≤ 2 * k + 4) ∧ 8 ≤ bw 0)) := by
          simp [bw]
        exact False.elim (this hreg)
    | succ y => positivity
  have hy : 1 < y := by
    have hbw2 : 2 ≤ bw y := by
      rcases hreg with hmid | hhigh <;> omega
    have hbw := bw_bounds (Nat.ne_of_gt hy0)
    have htwo : 2 ≤ 2 ^ (bw y - 1) := by
      have : 2 ^ 1 ≤ 2 ^ (bw y - 1) := by
        exact Nat.pow_le_pow_right (by decide : 0 < 2) (by omega)
      simpa using this
    exact lt_of_lt_of_le (by decide : 1 < 2) (le_trans htwo hbw.1)
  have hlow : Q - 1 ≤ q0 k y := by
    simpa [Q] using hbounds.1
  have hupp : q0 k y ≤ Q := by
    simpa [Q] using hbounds.2
  have hq0 : q0 k y = Q ∨ q0 k y = Q - 1 := by
    omega
  rcases hq0 with hq0 | hq0
  · exact q0_eq_floor_implies_Good hy (by simpa [Q] using hq0)
  · exact q0_eq_floor_pred_implies_Good hy hyW hpow (by simpa [Q] using hq0)

open Reciprocal

namespace Reciprocal

/--
TeX correspondence: `thm:mid-high`.

Lean proves the same regime theorem, but isolates an exact subregime
(`bw y ≤ 2*k + 4`) before handing the remaining late-high branch to a separate
tail proof.
-/
theorem mid_high_reduced_to_late_high
    (late_high :
      ∀ {k y : ℕ},
        3 ≤ k →
        y < 2 ^ (W k) →
        (2 * k + 4 < bw y) →
        8 ≤ bw y →
        (¬ ∃ m : ℕ, y = 2 ^ m) →
        Good k y)
    {k y : ℕ} :
    3 ≤ k →
    y < 2 ^ (W k) →
    (((k < bw y) ∧ (bw y ≤ 2 * k) ∧ 6 ≤ bw y) ∨ ((2 * k < bw y) ∧ 8 ≤ bw y)) →
    Good k y := by
  intro hk hyW hreg
  by_cases hpow : ∃ m : ℕ, y = 2 ^ m
  · rcases hpow with ⟨m, rfl⟩
    have hmW : m + 1 ≤ W k := by
      simpa [bw_two_pow] using
        (bw_le_of_lt_pow (y := 2 ^ m) (a := W k) (by positivity) hyW)
    exact Good_power_of_two hk hmW
  · rcases hreg with hmid | hhigh
    · exact exact_regime_proved hk hyW (Or.inl hmid) hpow
    · by_cases hℓ : bw y ≤ 2 * k + 4
      · exact exact_regime_proved hk hyW (Or.inr ⟨hhigh.1, hℓ, hhigh.2⟩) hpow
      · exact late_high hk hyW (lt_of_not_ge hℓ) hhigh.2 hpow

end Reciprocal



open Reciprocal
open ReciprocalMiddle
open ReciprocalLowExactCore
open ReciprocalLowExact

namespace ReciprocalLateHighProof

/--
TeX correspondence: the late-high tail of `thm:mid-high`.

This is more explicit than the TeX presentation: Lean separates the nonresonant,
resonant-large-distance, and resonant-small branches.
-/
lemma num_eq_general {k y : ℕ}
    (hb4 : 4 ≤ bw y)
    (hTlt : y * z1 k y < 2 ^ (2 * k + bw y - 1)) :
    let S : ℕ := 2 ^ (2 * k + bw y - 2)
    let D : ℕ := Nat.dist (y * z1 k y) S
    2 ^ (W k + 2 * bw y - 4) - D ^ 2 =
      y * (((2 ^ (2 * k + 3) * z1 k y) * 2 ^ (bw y - 4)) - y * (z1 k y) ^ 2) := by
  set S : ℕ := 2 ^ (2 * k + bw y - 2)
  set T : ℕ := y * z1 k y
  set D : ℕ := Nat.dist T S
  change 2 ^ (W k + 2 * bw y - 4) - D ^ 2 =
    y * (((2 ^ (2 * k + 3) * z1 k y) * 2 ^ (bw y - 4)) - y * (z1 k y) ^ 2)
  have hprod :
      y * (((2 ^ (2 * k + 3) * z1 k y) * 2 ^ (bw y - 4)) - y * (z1 k y) ^ 2) =
        T * (2 * S - T) := by
    dsimp [T, S]
    have hyzle : y * z1 k y ≤ 2 ^ (2 * k + bw y - 1) := le_of_lt hTlt
    have hpowterm :
        (2 ^ (2 * k + 3) * z1 k y) * 2 ^ (bw y - 4) =
          z1 k y * 2 ^ (2 * k + bw y - 1) := by
      calc
        (2 ^ (2 * k + 3) * z1 k y) * 2 ^ (bw y - 4)
            = z1 k y * (2 ^ (2 * k + 3) * 2 ^ (bw y - 4)) := by ac_rfl
        _ = z1 k y * 2 ^ ((2 * k + 3) + (bw y - 4)) := by rw [← pow_add]
        _ = z1 k y * 2 ^ (2 * k + bw y - 1) := by
              congr 2
              omega
    calc
      y * (((2 ^ (2 * k + 3) * z1 k y) * 2 ^ (bw y - 4)) - y * (z1 k y) ^ 2)
          = y * (z1 k y * 2 ^ (2 * k + bw y - 1) - y * (z1 k y) ^ 2) := by
              rw [hpowterm]
      _ = y * (z1 k y * (2 ^ (2 * k + bw y - 1) - y * z1 k y)) := by
            rw [show y * (z1 k y) ^ 2 = z1 k y * (y * z1 k y) by
              rw [pow_two]
              ac_rfl]
            rw [← Nat.mul_sub_left_distrib]
      _ = (y * z1 k y) * (2 ^ (2 * k + bw y - 1) - y * z1 k y) := by
            ac_rfl
      _ = T * (2 * S - T) := by
            rw [show 2 ^ (2 * k + bw y - 1) = 2 * 2 ^ (2 * k + bw y - 2) by
              rw [show 2 * k + bw y - 1 = (2 * k + bw y - 2) + 1 by omega, pow_add]
              norm_num
              rw [Nat.mul_comm]]
  have hquad : T * (2 * S - T) = S ^ 2 - D ^ 2 := by
    have : T < 2 * S := by
      dsimp [S, T]
      rw [show 2 * 2 ^ (2 * k + bw y - 2) = 2 ^ (2 * k + bw y - 1) by
        rw [show 2 * k + bw y - 1 = (2 * k + bw y - 2) + 1 by omega, pow_add]
        norm_num
        rw [Nat.mul_comm]]
      exact hTlt
    rw [quad_center_dist_sq this]
  have hSsq : S ^ 2 = 2 ^ (W k + 2 * bw y - 4) := by
    dsimp [S]
    rw [pow_two, ← pow_add]
    unfold W
    congr 1
    omega
  calc
    2 ^ (W k + 2 * bw y - 4) - D ^ 2 = S ^ 2 - D ^ 2 := by rw [hSsq]
    _ = T * (2 * S - T) := hquad.symm
    _ = y * (((2 ^ (2 * k + 3) * z1 k y) * 2 ^ (bw y - 4)) - y * (z1 k y) ^ 2) := hprod.symm

lemma exact_floor_eq_ceilDiv {k y : ℕ} (hy : 0 < y) (hb4 : 4 ≤ bw y)
    (hTlt : y * z1 k y < 2 ^ (2 * k + bw y - 1)) :
    let ell : ℕ := bw y
    let D : ℕ := Nat.dist (y * z1 k y) (2 ^ (2 * k + ell - 2))
    let M : ℕ := (2 ^ (W k + 2 * ell - 4) - D ^ 2) / y
    let a : ℕ := 2 ^ (ell - 4)
    2 ^ (2 * k + 3) * z1 k y - (y * (z1 k y) ^ 2) / a = M ⌈/⌉ a := by
  set ell : ℕ := bw y
  set D : ℕ := Nat.dist (y * z1 k y) (2 ^ (2 * k + ell - 2))
  set M : ℕ := (2 ^ (W k + 2 * ell - 4) - D ^ 2) / y
  set a : ℕ := 2 ^ (ell - 4)
  change 2 ^ (2 * k + 3) * z1 k y - (y * (z1 k y) ^ 2) / a = M ⌈/⌉ a
  have hm : y * (z1 k y) ^ 2 ≤ (2 ^ (2 * k + 3) * z1 k y) * a := by
    calc
      y * (z1 k y) ^ 2 = z1 k y * (y * z1 k y) := by
        rw [pow_two]
        ac_rfl
      _ ≤ z1 k y * 2 ^ (2 * k + ell - 1) := Nat.mul_le_mul_left _ (le_of_lt hTlt)
      _ = (2 ^ (2 * k + 3) * z1 k y) * a := by
            dsimp [a]
            calc
              z1 k y * 2 ^ (2 * k + ell - 1)
                  = z1 k y * (2 ^ (2 * k + 3) * 2 ^ (ell - 4)) := by
                      rw [show 2 * k + ell - 1 = (2 * k + 3) + (ell - 4) by omega, pow_add]
              _ = (2 ^ (2 * k + 3) * z1 k y) * 2 ^ (ell - 4) := by
                    ac_rfl
  rw [sub_div_eq_ceilDiv (hn := by positivity) hm]
  have hnum :
      2 ^ (W k + 2 * ell - 4) - D ^ 2 =
        y * (((2 ^ (2 * k + 3) * z1 k y) * a) - y * (z1 k y) ^ 2) := by
    have htmp := num_eq_general (k := k) (y := y) hb4 hTlt
    dsimp [D, ell, a] at htmp
    simpa using htmp
  have hquot :
      ((2 ^ (2 * k + 3) * z1 k y) * a - y * (z1 k y) ^ 2) = M := by
    symm
    dsimp [M]
    exact Nat.div_eq_of_eq_mul_right hy (by simpa [Nat.mul_comm] using hnum)
  simpa using congrArg (fun t => t ⌈/⌉ a) hquot

lemma inner_late_high {k y : ℕ} (hyW : y < 2 ^ (W k)) (hℓ : 2 * k + 4 < bw y) :
    (x k y * z1 k y) / 2 ^ (2 * k - 4) = (y * z1 k y) / 2 ^ (bw y - 2 * k - 4) := by
  unfold x
  set u : ℕ := bw y - 2 * k - 4
  have hy0 : y ≠ 0 := by
    cases y with
    | zero => simp [bw] at hℓ
    | succ _ => simp
  have hbwW : bw y ≤ W k := bw_le_of_lt_pow hy0 hyW
  have hu : 2 * k + 4 ≤ bw y := by omega
  have hk : 3 ≤ k := by
    unfold W at hbwW
    omega
  have hexp : (W k - bw y) + u = 2 * k - 4 := by
    have hsum : ((W k - bw y) + u) + (2 * k + 4) = (2 * k - 4) + (2 * k + 4) := by
      calc
        ((W k - bw y) + u) + (2 * k + 4)
            = (W k - bw y) + (u + (2 * k + 4)) := by ac_rfl
        _ = (W k - bw y) + bw y := by
              dsimp [u]
              rw [show bw y - 2 * k - 4 = bw y - (2 * k + 4) by omega]
              rw [Nat.sub_add_cancel hu]
        _ = W k := Nat.sub_add_cancel hbwW
        _ = (2 * k - 4) + (2 * k + 4) := by
              unfold W
              omega
    exact Nat.add_right_cancel hsum
  rw [show y * 2 ^ (W k - bw y) * z1 k y = (y * z1 k y) * 2 ^ (W k - bw y) by ac_rfl]
  rw [show 2 ^ (2 * k - 4) = 2 ^ (W k - bw y) * 2 ^ u by
    rw [← hexp, pow_add]]
  rw [← Nat.div_div_eq_div_mul]
  rw [Nat.mul_div_left _ (by positivity : 0 < 2 ^ (W k - bw y))]

lemma div_sub_small_bounds {n v a : ℕ} (ha : 0 < a) (hv : v < a) :
    n / a - 1 ≤ (n - v) / a ∧ (n - v) / a ≤ n / a := by
  constructor
  · refine (Nat.le_div_iff_mul_le ha).2 ?_
    have hpred : (n / a - 1) * a = (n / a) * a - a := by
      simpa [Nat.pred_eq_sub_one] using (Nat.pred_mul (n / a) a)
    have hmul : (n / a) * a ≤ n := by
      simpa [Nat.mul_comm] using (Nat.mul_div_le n a)
    rw [hpred]
    calc
      n / a * a - a ≤ n - a := Nat.sub_le_sub_right hmul _
      _ ≤ n - v := Nat.sub_le_sub_left (Nat.le_of_lt hv) _
  · exact Nat.div_le_div_right (Nat.sub_le _ _)

lemma z1_lt_two_pow {k y : ℕ} (hk : 3 ≤ k) (hy : 0 < y) (hyW : y < 2 ^ (W k)) :
    z1 k y < 2 ^ (2 * k) := by
  have hz1 : z1 k y < 2 ^ (2 * k - 1) + 2 := z1_lt_pow_half_add_two hk hy hyW
  have htwo : 2 ≤ 2 ^ (2 * k - 1) := by
    have hpow : 2 ^ 1 ≤ 2 ^ (2 * k - 1) := by
      exact Nat.pow_le_pow_right (by decide : 0 < 2) (by omega)
    simpa using hpow
  calc
    z1 k y < 2 ^ (2 * k - 1) + 2 := hz1
    _ ≤ 2 ^ (2 * k - 1) + 2 ^ (2 * k - 1) := Nat.add_le_add_left htwo _
    _ = 2 ^ (2 * k) := by
          have hdouble : 2 ^ (2 * k - 1) + 2 ^ (2 * k - 1) = 2 * 2 ^ (2 * k - 1) := by
            omega
          calc
            2 ^ (2 * k - 1) + 2 ^ (2 * k - 1) = 2 * 2 ^ (2 * k - 1) := hdouble
            _ = 2 ^ (2 * k) := by
                  rw [show 2 * k = 1 + (2 * k - 1) by omega, pow_add]
                  norm_num

lemma z2_late_high_floor_succ {k y : ℕ} (hk : 3 ≤ k) (hyW : y < 2 ^ (W k))
    (hℓ : 2 * k + 4 < bw y) :
    let ell : ℕ := bw y
    -- `d` is introduced in the proof body below.
    let a : ℕ := 2 ^ (ell - 4)
    let E : ℕ := 2 ^ (2 * k + 3) * z1 k y - (y * (z1 k y) ^ 2) / a
    E ≤ z2 k y ∧ z2 k y ≤ E + 1 := by
  set ell : ℕ := bw y
  set d : ℕ := 2 ^ (ell - 2 * k - 4)
  set a : ℕ := 2 ^ (ell - 4)
  set A : ℕ := (y * z1 k y) / d
  set R : ℕ := (y * z1 k y) % d
  set C : ℕ := 2 ^ (2 * k + 3) * z1 k y
  set E : ℕ := C - (y * (z1 k y) ^ 2) / a
  change E ≤ z2 k y ∧ z2 k y ≤ E + 1
  have hy0 : 0 < y := by
    cases y with
    | zero =>
        exfalso
        have hell : ell = 1 := by
          decide
        omega
    | succ _ => positivity
  have hdpos : 0 < d := by
    dsimp [d]
    positivity
  have hapos : 0 < a := by
    dsimp [a]
    positivity
  have hu : 2 * k + 4 ≤ ell := by
    dsimp [ell]
    omega
  have hpowa : d * 2 ^ (2 * k) = a := by
    dsimp [d, a]
    rw [show ell - 4 = (ell - 2 * k - 4) + 2 * k by omega, pow_add, Nat.mul_comm]
  have hinner : (x k y * z1 k y) / 2 ^ (2 * k - 4) = A := by
    simpa [A, d, ell] using (inner_late_high (k := k) (y := y) hyW hℓ)
  have hz2eq : z2 k y = C - (z1 k y * A) / 2 ^ (2 * k) := by
    unfold z2
    simp [C, hinner, A]
  have hdivmod : y * z1 k y = A * d + R := by
    dsimp [A, R]
    simpa [Nat.mul_comm, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using
      (Nat.div_add_mod' (y * z1 k y) d).symm
  have hAd : A * d = y * z1 k y - R := by
    omega
  have hmul :
      y * (z1 k y) ^ 2 - z1 k y * R = z1 k y * A * d := by
    calc
      y * (z1 k y) ^ 2 - z1 k y * R = z1 k y * (y * z1 k y - R) := by
        rw [pow_two, Nat.mul_sub_left_distrib]
        ac_rfl
      _ = z1 k y * (A * d) := by rw [hAd]
      _ = z1 k y * A * d := by ac_rfl
  have hAdiv :
      z1 k y * A = (y * (z1 k y) ^ 2 - z1 k y * R) / d := by
    symm
    apply Nat.div_eq_of_eq_mul_right hdpos
    simpa [Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm] using hmul
  have hB :
      (z1 k y * A) / 2 ^ (2 * k) = (y * (z1 k y) ^ 2 - z1 k y * R) / a := by
    rw [hAdiv]
    simpa [hpowa] using
      (Nat.div_div_eq_div_mul (y * (z1 k y) ^ 2 - z1 k y * R) d (2 ^ (2 * k)))
  have hz1lt : z1 k y < 2 ^ (2 * k) := z1_lt_two_pow hk hy0 hyW
  have hRlt : R < d := by
    dsimp [R]
    exact Nat.mod_lt _ hdpos
  have hvsmall : z1 k y * R < a := by
    have hle : z1 k y * R ≤ z1 k y * d := Nat.mul_le_mul_left _ (le_of_lt hRlt)
    have hlt : z1 k y * d < 2 ^ (2 * k) * d := Nat.mul_lt_mul_of_pos_right hz1lt hdpos
    calc
      z1 k y * R ≤ z1 k y * d := hle
      _ < 2 ^ (2 * k) * d := hlt
      _ = a := by rw [Nat.mul_comm, hpowa]
  have hdivbounds :=
    div_sub_small_bounds (n := y * (z1 k y) ^ 2) (v := z1 k y * R) (a := a) hapos hvsmall
  have hqleC : (y * (z1 k y) ^ 2) / a ≤ C := by
    have hm : y * (z1 k y) ^ 2 ≤ C * a := by
      calc
        y * (z1 k y) ^ 2 = z1 k y * (y * z1 k y) := by
          rw [pow_two]
          ac_rfl
        _ ≤ z1 k y * 2 ^ (2 * k + ell - 1) := Nat.mul_le_mul_left _ (by
              have hTlt : y * z1 k y < 2 ^ (2 * k + ell - 1) := by
                simpa [ell] using yz1_high_lt_double_target (k := k) (y := y) hk hy0 hyW
              exact le_of_lt hTlt)
        _ = C * a := by
              dsimp [C, a]
              calc
                z1 k y * 2 ^ (2 * k + ell - 1)
                    = z1 k y * (2 ^ (2 * k + 3) * 2 ^ (ell - 4)) := by
                        rw [show 2 * k + ell - 1 = (2 * k + 3) + (ell - 4) by omega, pow_add]
                _ = (2 ^ (2 * k + 3) * z1 k y) * 2 ^ (ell - 4) := by
                      ac_rfl
    have hltq : (y * (z1 k y) ^ 2) / a < C + 1 := by
      refine (Nat.div_lt_iff_lt_mul hapos).2 ?_
      calc
        y * (z1 k y) ^ 2 ≤ C * a := hm
        _ < (C + 1) * a := by
              simpa [Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using
                (Nat.mul_lt_mul_of_pos_right (Nat.lt_succ_self C) hapos)
    exact Nat.lt_succ_iff.mp hltq
  have hz2eq' :
      z2 k y = C - (y * (z1 k y) ^ 2 - z1 k y * R) / a := by
    rw [hz2eq, hB]
  constructor
  · rw [hz2eq']
    dsimp [E]
    omega
  · rw [hz2eq']
    dsimp [E]
    omega

lemma q0_late_high_bounds_of_Mupper2 {k y : ℕ} (hk : 3 ≤ k) (hyW : y < 2 ^ (W k))
    (hℓ : 2 * k + 4 < bw y)
    (hMupper2 :
      let ell : ℕ := bw y
      let a : ℕ := 2 ^ (ell - 4)
      let P : ℕ := 2 ^ (2 * ell - 4)
      let D : ℕ := Nat.dist (y * z1 k y) (2 ^ (2 * k + ell - 2))
      let M : ℕ := (2 ^ (W k + 2 * ell - 4) - D ^ 2) / y
      let Q : ℕ := 2 ^ (W k) / y
      M + 2 * a ≤ (Q + 1) * P) :
    2 ^ (W k) / y - 1 ≤ q0 k y ∧ q0 k y ≤ 2 ^ (W k) / y := by
  set ell : ℕ := bw y
  set a : ℕ := 2 ^ (ell - 4)
  set P : ℕ := 2 ^ (2 * ell - 4)
  set D : ℕ := Nat.dist (y * z1 k y) (2 ^ (2 * k + ell - 2))
  set M : ℕ := (2 ^ (W k + 2 * ell - 4) - D ^ 2) / y
  set Q : ℕ := 2 ^ (W k) / y
  set s : ℕ := 2 ^ (W k) % y
  have hMupper2' : M + 2 * a ≤ (Q + 1) * P := by
    simpa [ell, a, P, D, M, Q] using hMupper2
  have hb4 : 4 ≤ ell := by
    dsimp [ell]
    omega
  have hy0 : 0 < y := by
    cases y with
    | zero =>
        have hell : ell = 1 := by
          decide
        omega
    | succ _ => positivity
  have hTlt : y * z1 k y < 2 ^ (2 * k + ell - 1) := by
    simpa [ell] using yz1_high_lt_double_target (k := k) (y := y) hk hy0 hyW
  have hslt : s < y := by
    dsimp [s]
    exact Nat.mod_lt _ hy0
  have hdivmod : 2 ^ (W k) = Q * y + s := by
    dsimp [Q, s]
    simpa [Nat.mul_comm] using (Nat.div_add_mod' (2 ^ (W k)) y).symm
  have hyltpow : y < 2 ^ ell := by
    simpa [ell] using (bw_bounds (Nat.ne_of_gt hy0)).2
  have hPsplit : P = 2 ^ ell * a := by
    dsimp [P, a]
    rw [show 2 * ell - 4 = ell + (ell - 4) by omega, pow_add]
  have hAleP : a ≤ P := by
    rw [hPsplit]
    simpa [Nat.mul_comm] using
      (Nat.le_mul_of_pos_right a (show 0 < 2 ^ ell by positivity))
  have hpowWP : 2 ^ (W k + 2 * ell - 4) = 2 ^ (W k) * P := by
    dsimp [P]
    rw [show W k + 2 * ell - 4 = W k + (2 * ell - 4) by omega, pow_add]
  have hD2ltP : D ^ 2 < y * P := by
    have hDlt : D < 3 * y := by
      simpa [D, ell] using yz1_high_dist_lt_three_y (k := k) (y := y) hk hy0 hyW
    cases Nat.eq_zero_or_pos D with
    | inl hD0 =>
        rw [hD0, pow_two]
        positivity
    | inr hDpos =>
        calc
          D ^ 2 = D * D := by rw [pow_two]
          _ < D * (3 * y) := Nat.mul_lt_mul_of_pos_left hDlt hDpos
          _ < (3 * y) * (3 * y) := Nat.mul_lt_mul_of_pos_right hDlt (by positivity)
          _ = y * (9 * y) := by ring
          _ < y * P := by
                apply Nat.mul_lt_mul_of_pos_left ?_ hy0
                calc
                  9 * y < 9 * 2 ^ ell := Nat.mul_lt_mul_of_pos_left hyltpow (by positivity)
                  _ ≤ 2 ^ (ell - 4) * 2 ^ ell := by
                        have hpow9 : 9 ≤ 2 ^ (ell - 4) := by
                          have : 2 ^ 4 ≤ 2 ^ (ell - 4) := by
                            exact Nat.pow_le_pow_right (by decide : 0 < 2) (by omega)
                          have h16 : 16 ≤ 2 ^ (ell - 4) := by simpa using this
                          omega
                        exact Nat.mul_le_mul_right _ hpow9
                  _ = P := by
                        dsimp [P]
                        rw [← pow_add]
                        congr 1
                        omega
  have hMdiv : y ∣ 2 ^ (W k + 2 * ell - 4) - D ^ 2 := by
    refine ⟨((2 ^ (2 * k + 3) * z1 k y) * 2 ^ (ell - 4)) - y * (z1 k y) ^ 2, ?_⟩
    simpa [D, ell] using (num_eq_general (k := k) (y := y) hb4 hTlt)
  have hMlower : Q * P - P < M := by
    apply (Nat.lt_div_iff_mul_lt' hMdiv _).2
    have hQpos : 0 < Q := by
      dsimp [Q]
      exact Nat.div_pos (Nat.le_of_lt hyW) hy0
    have hQP : P ≤ Q * P := by
      simpa [Nat.mul_comm] using (Nat.le_mul_of_pos_right P hQpos)
    have hleD : D ^ 2 ≤ y * (Q * P) + s * P := by
      exact le_trans (le_of_lt hD2ltP) (by
        have : y * P ≤ y * (Q * P) + s * P := by
          exact le_trans (Nat.mul_le_mul_left _ hQP) (Nat.le_add_right _ _)
        exact this)
    have hraw_add : y * (Q * P - P) + D ^ 2 < y * (Q * P) + s * P := by
      have hstep1 : y * (Q * P - P) + D ^ 2 < y * (Q * P - P) + y * P := by
        exact Nat.add_lt_add_left hD2ltP _
      have hstep2 : y * (Q * P - P) + y * P = y * (Q * P) := by
        rw [Nat.mul_sub_left_distrib]
        exact Nat.sub_add_cancel (Nat.mul_le_mul_left _ hQP)
      have hstep3 : y * (Q * P) ≤ y * (Q * P) + s * P := Nat.le_add_right _ _
      have hstep1' : y * (Q * P - P) + D ^ 2 < y * (Q * P) := by
        simpa [hstep2] using hstep1
      exact lt_of_lt_of_le hstep1' hstep3
    have hraw : y * (Q * P - P) < y * (Q * P) + s * P - D ^ 2 := by
      have hsubadd : (y * (Q * P) + s * P - D ^ 2) + D ^ 2 = y * (Q * P) + s * P := by
        exact Nat.sub_add_cancel hleD
      have htmp : y * (Q * P - P) + D ^ 2 < (y * (Q * P) + s * P - D ^ 2) + D ^ 2 := by
        simpa [hsubadd] using hraw_add
      exact Nat.lt_of_add_lt_add_right htmp
    calc
      y * (Q * P - P) < y * (Q * P) + s * P - D ^ 2 := hraw
      _ = 2 ^ (W k + 2 * ell - 4) - D ^ 2 := by
        rw [hpowWP, hdivmod, Nat.add_mul]
        ac_rfl
  have hEeq :
      2 ^ (2 * k + 3) * z1 k y - (y * (z1 k y) ^ 2) / a = M ⌈/⌉ a := by
    simpa [D, M, ell, a] using (exact_floor_eq_ceilDiv (k := k) (y := y) hy0 hb4 hTlt)
  have hz2band :
      M ⌈/⌉ a ≤ z2 k y ∧ z2 k y ≤ M ⌈/⌉ a + 1 := by
    have hz2lower0 :
        2 ^ (2 * k + 3) * z1 k y - (y * (z1 k y) ^ 2) / a ≤ z2 k y := by
      simpa [ell, a] using (z2_late_high_floor_succ (k := k) (y := y) hk hyW hℓ).1
    have hz2upper0 :
        z2 k y ≤ (2 ^ (2 * k + 3) * z1 k y - (y * (z1 k y) ^ 2) / a) + 1 := by
      simpa [ell, a] using (z2_late_high_floor_succ (k := k) (y := y) hk hyW hℓ).2
    constructor
    · rw [← hEeq]
      exact hz2lower0
    · rw [← hEeq]
      exact hz2upper0
  have hz2lower : (Q - 1) * 2 ^ ell < z2 k y := by
    have hEgt : (Q - 1) * 2 ^ ell < M ⌈/⌉ a := by
      apply lt_ceilDiv_of_mul_lt (a := a) (by positivity)
      calc
        a * ((Q - 1) * 2 ^ ell) = (Q - 1) * P := by
          rw [hPsplit]
          ac_rfl
        _ = Q * P - P := by
          simpa [Nat.pred_eq_sub_one] using (Nat.pred_mul Q P)
        _ < M := hMlower
    exact lt_of_lt_of_le hEgt hz2band.1
  have hz2upper : z2 k y < (Q + 1) * 2 ^ ell := by
    have hmul : a * (M ⌈/⌉ a) < M + a := by
      simpa [Nat.mul_comm] using (mul_ceilDiv_lt_add (a := a) (b := M) (by positivity))
    have hmul' : a * ((M ⌈/⌉ a) + 1) < a * ((Q + 1) * 2 ^ ell) := by
      calc
        a * ((M ⌈/⌉ a) + 1) = a * (M ⌈/⌉ a) + a := by rw [Nat.mul_add, Nat.mul_one]
        _ < (M + a) + a := Nat.add_lt_add_right hmul _
        _ = M + 2 * a := by omega
        _ ≤ (Q + 1) * P := hMupper2'
        _ = a * ((Q + 1) * 2 ^ ell) := by
          rw [hPsplit]
          ac_rfl
    have hEupper : (M ⌈/⌉ a) + 1 < (Q + 1) * 2 ^ ell := by
      exact (Nat.mul_lt_mul_left (by positivity : 0 < a)).1 hmul'
    exact lt_of_le_of_lt hz2band.2 hEupper
  have hq0lower : Q - 1 ≤ q0 k y := by
    unfold q0
    refine (Nat.le_div_iff_mul_le (by positivity : 0 < 2 ^ ell)).2 ?_
    exact le_of_lt (by simpa [ell, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using hz2lower)
  have hq0upper : q0 k y ≤ Q := by
    unfold q0
    exact Nat.le_of_lt_succ ((Nat.div_lt_iff_lt_mul (by positivity : 0 < 2 ^ ell)).2
      (by simpa [ell, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using hz2upper))
  simpa [Q] using And.intro hq0lower hq0upper

lemma late_high_nonresonant_q0_bounds {k y : ℕ} (hk : 3 ≤ k) (hyW : y < 2 ^ (W k))
    (hℓ : 2 * k + 4 < bw y) (hsne : 2 ^ (W k) % y ≠ y - 1) :
    2 ^ (W k) / y - 1 ≤ q0 k y ∧ q0 k y ≤ 2 ^ (W k) / y := by
  set ell : ℕ := bw y
  set a : ℕ := 2 ^ (ell - 4)
  set P : ℕ := 2 ^ (2 * ell - 4)
  set D : ℕ := Nat.dist (y * z1 k y) (2 ^ (2 * k + ell - 2))
  set M : ℕ := (2 ^ (W k + 2 * ell - 4) - D ^ 2) / y
  set Q : ℕ := 2 ^ (W k) / y
  set s : ℕ := 2 ^ (W k) % y
  have hy0 : 0 < y := by
    cases y with
    | zero =>
        have hell : ell = 1 := by
          decide
        omega
    | succ _ => positivity
  have hslt : s < y := by
    dsimp [s]
    exact Nat.mod_lt _ hy0
  have hsle : s ≤ y - 2 := by
    omega
  have hyltpow : y < 2 ^ ell := by
    simpa [ell] using (bw_bounds (Nat.ne_of_gt hy0)).2
  have hPsplit : P = 2 ^ ell * a := by
    dsimp [P, a]
    rw [show 2 * ell - 4 = ell + (ell - 4) by omega, pow_add]
  have hyPa : y * a ≤ P := by
    rw [hPsplit]
    simpa [Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm] using
      (Nat.mul_le_mul_left a (le_of_lt hyltpow))
  have hsP : s * P ≤ y * (P - 2 * a) := by
    calc
      s * P ≤ (y - 2) * P := Nat.mul_le_mul_right _ hsle
      _ = y * P - 2 * P := by
            rw [Nat.sub_mul, two_mul]
      _ ≤ y * P - y * (2 * a) := by
            have h2ya : y * (2 * a) ≤ 2 * P := by
              simpa [two_mul, Nat.mul_add, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using
                Nat.add_le_add hyPa hyPa
            exact Nat.sub_le_sub_left h2ya _
      _ = y * (P - 2 * a) := by rw [Nat.mul_sub_left_distrib]
  have hpowWP : 2 ^ (W k + 2 * ell - 4) = 2 ^ (W k) * P := by
    dsimp [P]
    rw [show W k + 2 * ell - 4 = W k + (2 * ell - 4) by omega, pow_add]
  have hdivmod : 2 ^ (W k) = Q * y + s := by
    dsimp [Q, s]
    simpa [Nat.mul_comm] using (Nat.div_add_mod' (2 ^ (W k)) y).symm
  have hMupper2 :
      M + 2 * a ≤ (Q + 1) * P := by
    have hMupper : M ≤ Q * P + (P - 2 * a) := by
      apply Nat.div_le_of_le_mul
      calc
        2 ^ (W k + 2 * ell - 4) - D ^ 2 ≤ 2 ^ (W k + 2 * ell - 4) := Nat.sub_le _ _
        _ = y * (Q * P) + s * P := by
          rw [hpowWP, hdivmod, Nat.add_mul]
          ac_rfl
        _ ≤ y * (Q * P) + y * (P - 2 * a) := Nat.add_le_add_left hsP _
        _ = y * (Q * P + (P - 2 * a)) := by rw [Nat.mul_add]
    have hsum : (Q * P + (P - 2 * a)) + 2 * a = Q * P + P := by
      have h2a_le_P : 2 * a ≤ P := by
        have h2le : 2 ≤ 2 ^ ell := by
          have : 2 ^ 1 ≤ 2 ^ ell := by
            exact Nat.pow_le_pow_right (by decide : 0 < 2) (by omega)
          simpa using this
        calc
          2 * a ≤ 2 ^ ell * a := by
                simpa [Nat.mul_comm] using Nat.mul_le_mul_right a h2le
          _ = P := hPsplit.symm
      calc
        (Q * P + (P - 2 * a)) + 2 * a = Q * P + ((P - 2 * a) + 2 * a) := by ac_rfl
        _ = Q * P + P := by
              rw [Nat.sub_add_cancel h2a_le_P]
    calc
      M + 2 * a ≤ (Q * P + (P - 2 * a)) + 2 * a := Nat.add_le_add_right hMupper _
      _ = Q * P + P := hsum
      _ = (Q + 1) * P := by rw [Nat.add_mul, one_mul]
  exact q0_late_high_bounds_of_Mupper2 (k := k) (y := y) hk hyW hℓ (by
    simpa [ell, a, P, D, M, Q] using hMupper2)

lemma late_high_nonresonant_proved {k y : ℕ} (hk : 3 ≤ k) (hyW : y < 2 ^ (W k))
    (hℓ : 2 * k + 4 < bw y) (hpow : ¬ ∃ m : ℕ, y = 2 ^ m)
    (hsne : 2 ^ (W k) % y ≠ y - 1) :
    Good k y := by
  have hbounds := late_high_nonresonant_q0_bounds (k := k) (y := y) hk hyW hℓ hsne
  set Q : ℕ := 2 ^ (W k) / y
  have hy0 : 0 < y := by
    cases y with
    | zero =>
        have hell : bw 0 = 1 := by decide
        omega
    | succ _ => positivity
  have hy : 1 < y := by
    have hbw2 : 2 ≤ bw y := by
      omega
    have hbw := bw_bounds (Nat.ne_of_gt hy0)
    have htwo : 2 ≤ 2 ^ (bw y - 1) := by
      have : 2 ^ 1 ≤ 2 ^ (bw y - 1) := by
        exact Nat.pow_le_pow_right (by decide : 0 < 2) (by omega)
      simpa using this
    exact lt_of_lt_of_le (by decide : 1 < 2) (le_trans htwo hbw.1)
  have hlow : Q - 1 ≤ q0 k y := by
    simpa [Q] using hbounds.1
  have hupp : q0 k y ≤ Q := by
    simpa [Q] using hbounds.2
  have hq0 : q0 k y = Q ∨ q0 k y = Q - 1 := by
    omega
  rcases hq0 with hq0 | hq0
  · exact q0_eq_floor_implies_Good hy (by simpa [Q] using hq0)
  · exact q0_eq_floor_pred_implies_Good hy hyW hpow (by simpa [Q] using hq0)

lemma late_high_resonant_largeD_q0_bounds {k y : ℕ} (hk : 3 ≤ k) (hyW : y < 2 ^ (W k))
    (hℓ : 2 * k + 4 < bw y)
    (hs : 2 ^ (W k) % y = y - 1)
    (hDy :
      y ≤ Nat.dist (y * z1 k y) (2 ^ (2 * k + bw y - 2))) :
    2 ^ (W k) / y - 1 ≤ q0 k y ∧ q0 k y ≤ 2 ^ (W k) / y := by
  set ell : ℕ := bw y
  set a : ℕ := 2 ^ (ell - 4)
  set P : ℕ := 2 ^ (2 * ell - 4)
  set D : ℕ := Nat.dist (y * z1 k y) (2 ^ (2 * k + ell - 2))
  set M : ℕ := (2 ^ (W k + 2 * ell - 4) - D ^ 2) / y
  set Q : ℕ := 2 ^ (W k) / y
  set s : ℕ := 2 ^ (W k) % y
  have hDy' : y ≤ D := by simpa [D, ell] using hDy
  have hy0 : 0 < y := by
    cases y with
    | zero =>
        have hell : ell = 1 := by
          decide
        omega
    | succ _ => positivity
  have hyge : 2 ^ (ell - 1) ≤ y := (bw_bounds (Nat.ne_of_gt hy0)).1
  have h2a_le_y : 2 * a ≤ y := by
    dsimp [a]
    calc
      2 * 2 ^ (ell - 4) = 2 ^ (ell - 3) := by
            rw [show ell - 3 = (ell - 4) + 1 by omega, pow_add]
            norm_num
            rw [Nat.mul_comm]
      _ ≤ 2 ^ (ell - 1) := by
            exact Nat.pow_le_pow_right (by decide : 0 < 2) (by omega)
      _ ≤ y := hyge
  have h2ay_le_D2 : y * (2 * a) ≤ D ^ 2 := by
    have hy2a_le_yy : y * (2 * a) ≤ y * y := Nat.mul_le_mul_left _ h2a_le_y
    have hyy_le_D2 : y * y ≤ D * D := Nat.mul_le_mul hDy' hDy'
    exact le_trans hy2a_le_yy (by simpa [pow_two] using hyy_le_D2)
  have hpowWP : 2 ^ (W k + 2 * ell - 4) = 2 ^ (W k) * P := by
    dsimp [P]
    rw [show W k + 2 * ell - 4 = W k + (2 * ell - 4) by omega, pow_add]
  have hdivmod : 2 ^ (W k) = Q * y + s := by
    dsimp [Q, s]
    simpa [Nat.mul_comm] using (Nat.div_add_mod' (2 ^ (W k)) y).symm
  have hMupper2 :
      M + 2 * a ≤ (Q + 1) * P := by
    have hsP : s * P ≤ y * (P - 2 * a) + D ^ 2 := by
      have haux : y * (2 * a) ≤ P + D ^ 2 := le_trans h2ay_le_D2 (Nat.le_add_left _ _)
      have hsP' : y * P - P ≤ y * (P - 2 * a) + D ^ 2 := by
        rw [Nat.mul_sub_left_distrib]
        omega
      simpa [hs, Nat.sub_mul, one_mul] using hsP'
    have hMupper : M ≤ Q * P + (P - 2 * a) := by
      apply Nat.div_le_of_le_mul
      calc
        2 ^ (W k + 2 * ell - 4) - D ^ 2 = y * (Q * P) + s * P - D ^ 2 := by
          rw [hpowWP, hdivmod, Nat.add_mul]
          ac_rfl
        _ ≤ y * (Q * P) + y * (P - 2 * a) := by
              omega
        _ = y * (Q * P + (P - 2 * a)) := by rw [Nat.mul_add]
    have hsum : (Q * P + (P - 2 * a)) + 2 * a = Q * P + P := by
      have h2a_le_P : 2 * a ≤ P := by
        rw [show P = 2 ^ ell * a by
          dsimp [P, a]
          rw [show 2 * ell - 4 = ell + (ell - 4) by omega, pow_add]]
        have h2le : 2 ≤ 2 ^ ell := by
          have : 2 ^ 1 ≤ 2 ^ ell := by
            exact Nat.pow_le_pow_right (by decide : 0 < 2) (by omega)
          simpa using this
        calc
          2 * a ≤ 2 ^ ell * a := by
                simpa [Nat.mul_comm] using Nat.mul_le_mul_right a h2le
          _ = 2 ^ ell * a := rfl
      calc
        (Q * P + (P - 2 * a)) + 2 * a = Q * P + ((P - 2 * a) + 2 * a) := by ac_rfl
        _ = Q * P + P := by rw [Nat.sub_add_cancel h2a_le_P]
    calc
      M + 2 * a ≤ (Q * P + (P - 2 * a)) + 2 * a := Nat.add_le_add_right hMupper _
      _ = Q * P + P := hsum
      _ = (Q + 1) * P := by rw [Nat.add_mul, one_mul]
  exact q0_late_high_bounds_of_Mupper2 (k := k) (y := y) hk hyW hℓ (by
    simpa [ell, a, P, D, M, Q] using hMupper2)

lemma late_high_resonant_largeD_proved {k y : ℕ} (hk : 3 ≤ k) (hyW : y < 2 ^ (W k))
    (hℓ : 2 * k + 4 < bw y) (hpow : ¬ ∃ m : ℕ, y = 2 ^ m)
    (hs : 2 ^ (W k) % y = y - 1)
    (hDy :
      y ≤ Nat.dist (y * z1 k y) (2 ^ (2 * k + bw y - 2))) :
    Good k y := by
  have hbounds := late_high_resonant_largeD_q0_bounds (k := k) (y := y) hk hyW hℓ hs hDy
  set Q : ℕ := 2 ^ (W k) / y
  have hy0 : 0 < y := by
    cases y with
    | zero =>
        have hell : bw 0 = 1 := by decide
        omega
    | succ _ => positivity
  have hy : 1 < y := by
    have hbw2 : 2 ≤ bw y := by omega
    have hbw := bw_bounds (Nat.ne_of_gt hy0)
    have htwo : 2 ≤ 2 ^ (bw y - 1) := by
      have : 2 ^ 1 ≤ 2 ^ (bw y - 1) := by
        exact Nat.pow_le_pow_right (by decide : 0 < 2) (by omega)
      simpa using this
    exact lt_of_lt_of_le (by decide : 1 < 2) (le_trans htwo hbw.1)
  have hlow : Q - 1 ≤ q0 k y := by
    simpa [Q] using hbounds.1
  have hupp : q0 k y ≤ Q := by
    simpa [Q] using hbounds.2
  have hq0 : q0 k y = Q ∨ q0 k y = Q - 1 := by
    omega
  rcases hq0 with hq0 | hq0
  · exact q0_eq_floor_implies_Good hy (by simpa [Q] using hq0)
  · exact q0_eq_floor_pred_implies_Good hy hyW hpow (by simpa [Q] using hq0)

theorem late_high_reduced_to_resonant_small
    (resonant_small :
      ∀ {k y : ℕ},
        3 ≤ k →
        y < 2 ^ (W k) →
        2 * k + 4 < bw y →
        8 ≤ bw y →
        (¬ ∃ m : ℕ, y = 2 ^ m) →
        2 ^ (W k) % y = y - 1 →
        Nat.dist (y * z1 k y) (2 ^ (2 * k + bw y - 2)) < y →
        Good k y)
    {k y : ℕ} :
    3 ≤ k →
    y < 2 ^ (W k) →
    2 * k + 4 < bw y →
    8 ≤ bw y →
    (¬ ∃ m : ℕ, y = 2 ^ m) →
    Good k y := by
  intro hk hyW hℓ hb8 hpow
  by_cases hs : 2 ^ (W k) % y = y - 1
  · by_cases hDy : y ≤ Nat.dist (y * z1 k y) (2 ^ (2 * k + bw y - 2))
    · exact late_high_resonant_largeD_proved hk hyW hℓ hpow hs hDy
    · exact resonant_small hk hyW hℓ hb8 hpow hs (lt_of_not_ge hDy)
  · exact late_high_nonresonant_proved hk hyW hℓ hpow hs

end ReciprocalLateHighProof



open Reciprocal
open ReciprocalLateHighProof

namespace Reciprocal

end Reciprocal



open Reciprocal
open ReciprocalMiddle
open ReciprocalLateHighProof
open ReciprocalLowExactCore

namespace ReciprocalLateHighComplete

/--
TeX correspondence: final closure of the resonant small-distance branch that is
compressed into the paper's high-bit argument.
-/
lemma resonant_q_succ_mul {k y : ℕ} (hy : 0 < y) :
    let Q : ℕ := 2 ^ (W k) / y
    let s : ℕ := 2 ^ (W k) % y
    y * (Q + 1) = 2 ^ (W k) + (y - s) := by
  set Q : ℕ := 2 ^ (W k) / y
  set s : ℕ := 2 ^ (W k) % y
  have hdivmod : 2 ^ (W k) = Q * y + s := by
    dsimp [Q, s]
    simpa [Nat.mul_comm] using (Nat.div_add_mod' (2 ^ (W k)) y).symm
  have hsle : s ≤ y := by
    dsimp [s]
    exact le_of_lt (Nat.mod_lt _ hy)
  calc
    y * (Q + 1) = Q * y + y := by
      rw [Nat.mul_add, Nat.mul_one, Nat.mul_comm]
    _ = Q * y + ((y - s) + s) := by rw [Nat.sub_add_cancel hsle]
    _ = (Q * y + s) + (y - s) := by ac_rfl
    _ = 2 ^ (W k) + (y - s) := by rw [hdivmod]

lemma resonant_small_z1_cases {k y : ℕ} (hk : 3 ≤ k) (hyW : y < 2 ^ (W k))
    (hℓ : 2 * k + 4 < bw y)
    (hs : 2 ^ (W k) % y = y - 1)
    (hDsmall : Nat.dist (y * z1 k y) (2 ^ (2 * k + bw y - 2)) < y) :
    z1 k y = 2 ^ (bw y - 2 * k - 2) * (2 ^ (W k) / y + 1) - 1 ∨
      z1 k y = 2 ^ (bw y - 2 * k - 2) * (2 ^ (W k) / y + 1) := by
  set ell : ℕ := bw y
  set Q : ℕ := 2 ^ (W k) / y
  set c : ℕ := 2 ^ (ell - 2 * k - 2)
  set n : ℕ := c * (Q + 1)
  set S : ℕ := 2 ^ (2 * k + ell - 2)
  have hy0 : 0 < y := by
    cases y with
    | zero =>
        have hell : ell = 1 := by
          decide
        omega
    | succ _ => positivity
  have hc_lt_y : c < y := by
    have hbw := bw_bounds (Nat.ne_of_gt hy0)
    have hc_le : c ≤ 2 ^ (ell - 5) := by
      dsimp [c]
      exact Nat.pow_le_pow_right (by decide : 0 < 2) (by omega)
    calc
      c ≤ 2 ^ (ell - 5) := hc_le
      _ < 2 ^ (ell - 1) := by
            have : 2 ^ (ell - 5) * 16 = 2 ^ (ell - 1) := by
              rw [show ell - 1 = (ell - 5) + 4 by omega, pow_add]
              norm_num
            have h15pos : 0 < 15 * 2 ^ (ell - 5) := by positivity
            calc
              2 ^ (ell - 5) < 2 ^ (ell - 5) * 16 := by
                    calc
                      2 ^ (ell - 5) < 2 ^ (ell - 5) + 15 * 2 ^ (ell - 5) := Nat.lt_add_of_pos_right h15pos
                      _ = 2 ^ (ell - 5) * 16 := by ring
              _ = 2 ^ (ell - 1) := this
      _ ≤ y := hbw.1
  have hQsucc : y * (Q + 1) = 2 ^ (W k) + 1 := by
    have htmp := resonant_q_succ_mul (k := k) (y := y) hy0
    have hysub : y - (y - 1) = 1 := by omega
    simpa [Q, hs, hysub] using htmp
  have hS : S = y * n - c := by
    dsimp [S, n, c]
    have hQsucc' : 2 ^ (W k) = y * (Q + 1) - 1 := by
      omega
    calc
      2 ^ (2 * k + ell - 2) = 2 ^ (ell - 2 * k - 2) * 2 ^ (W k) := by
            rw [show 2 * k + ell - 2 = (ell - 2 * k - 2) + W k by
              unfold W
              omega, pow_add]
      _ = 2 ^ (ell - 2 * k - 2) * (y * (Q + 1) - 1) := by rw [hQsucc']
      _ = y * (2 ^ (ell - 2 * k - 2) * (Q + 1)) - 2 ^ (ell - 2 * k - 2) := by
            rw [Nat.mul_sub_left_distrib, Nat.mul_one]
            ac_rfl
      _ = y * n - c := by rfl
  by_cases hle : y * z1 k y ≤ S
  · have hdist : S - y * z1 k y < y := by
      have h := hDsmall
      rw [Nat.dist_eq_sub_of_le hle] at h
      simpa [S] using h
    have hz1_lt_n : z1 k y < n := by
      have hSy : S < y * n := by
        rw [hS]
        exact Nat.sub_lt (show 0 < y * n by positivity) (by positivity)
      have : y * z1 k y < y * n := lt_of_le_of_lt hle hSy
      exact (Nat.mul_lt_mul_left hy0).1 <| by simpa [Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using this
    have hgap_lt2 : y * (n - z1 k y) < 2 * y := by
      have hz1_le_n : z1 k y ≤ n := le_of_lt hz1_lt_n
      have hc_le_yn : c ≤ y * n := by
        have hcpos : 0 < c := by
          dsimp [c]
          positivity
        have hnpos : 0 < n := by
          dsimp [n]
          exact Nat.mul_pos hcpos (Nat.succ_pos _)
        calc
          c ≤ c * n := Nat.le_mul_of_pos_right _ hnpos
          _ ≤ y * n := Nat.mul_le_mul_right _ (le_of_lt hc_lt_y)
      have hSadd : y * n = S + c := by
        rw [hS]
        exact (Nat.sub_add_cancel hc_le_yn).symm
      calc
        y * (n - z1 k y) = y * n - y * z1 k y := by rw [Nat.mul_sub_left_distrib]
        _ = (S + c) - y * z1 k y := by rw [hSadd]
        _ = c + (S - y * z1 k y) := by
              simpa [Nat.add_comm] using (Nat.add_sub_assoc hle c)
        _ < c + y := Nat.add_lt_add_left hdist _
        _ < 2 * y := by omega
    have hgap_lt2' : n - z1 k y < 2 := by
      have : y * (n - z1 k y) < y * 2 := by simpa [Nat.mul_comm] using hgap_lt2
      exact (Nat.mul_lt_mul_left hy0).1 this
    have hgap_pos : 0 < n - z1 k y := Nat.sub_pos_of_lt hz1_lt_n
    have hgap : n - z1 k y = 1 := by omega
    left
    have hz1eq' : z1 k y = n - 1 := by omega
    simpa [n, Q, c, ell] using hz1eq'
  · have hgt : S < y * z1 k y := lt_of_not_ge hle
    have hz1_ge_n : n ≤ z1 k y := by
      have hlow : y * (n - 1) < S := by
        have hnpos : 0 < n := by
          dsimp [n, c]
          positivity
        have hy_le_yn : y ≤ y * n := by
          simpa [Nat.mul_comm] using (Nat.le_mul_of_pos_right y hnpos)
        rw [show y * (n - 1) = y * n - y by rw [Nat.mul_sub_left_distrib, Nat.mul_one]]
        rw [hS]
        omega
      by_contra hn
      have hz1_lt : z1 k y < n := lt_of_not_ge hn
      have hz1_le : z1 k y ≤ n - 1 := Nat.le_pred_of_lt hz1_lt
      have hyz_le : y * z1 k y ≤ y * (n - 1) := Nat.mul_le_mul_left _ hz1_le
      exact Nat.not_lt_of_ge hyz_le (lt_trans hlow hgt)
    have hdist : y * z1 k y - S < y := by
      have h := hDsmall
      rw [Nat.dist_eq_sub_of_le_right (le_of_lt hgt)] at h
      simpa [S] using h
    have hgap_lt1 : y * (z1 k y - n) < y := by
      calc
        y * (z1 k y - n) = y * z1 k y - y * n := by rw [Nat.mul_sub_left_distrib]
        _ ≤ y * z1 k y - S := by
              rw [hS]
              exact Nat.sub_le_sub_left (Nat.sub_le _ _) _
        _ < y := hdist
    have hgap_lt1' : z1 k y - n < 1 := by
      exact (Nat.mul_lt_mul_left hy0).1 <| by simpa [Nat.mul_comm] using hgap_lt1
    have hgap : z1 k y - n = 0 := by omega
    right
    have hz1eq' : z1 k y = n := by omega
    simpa [n, Q, c, ell] using hz1eq'

lemma resonant_exact_case_q0 {k y : ℕ} (hk : 3 ≤ k) (hyW : y < 2 ^ (W k))
    (hℓ : 2 * k + 4 < bw y)
    (hpow : ¬ ∃ m : ℕ, y = 2 ^ m)
    (hs : 2 ^ (W k) % y = y - 1)
    (hz1eq : z1 k y = 2 ^ (bw y - 2 * k - 2) * (2 ^ (W k) / y + 1)) :
    q0 k y = 2 ^ (W k) / y := by
  set ell : ℕ := bw y
  set Q : ℕ := 2 ^ (W k) / y
  set c : ℕ := 2 ^ (ell - 2 * k - 2)
  set d : ℕ := 2 ^ (ell - 2 * k - 4)
  set B : ℕ := Q + 1
  set m : ℕ := 2 ^ (W k - ell)
  have hy0 : 0 < y := by
    cases y with
    | zero =>
        have hell : ell = 1 := by
          decide
        omega
    | succ _ => positivity
  have hbw := bw_bounds (Nat.ne_of_gt hy0)
  have hygt : 2 ^ (ell - 1) < y := by
    have hyne : y ≠ 2 ^ (ell - 1) := by
      intro hEq
      exact hpow ⟨ell - 1, hEq⟩
    exact lt_of_le_of_ne hbw.1 hyne.symm
  have hB : y * B = 2 ^ (W k) + 1 := by
    have htmp := resonant_q_succ_mul (k := k) (y := y) hy0
    have hysub : y - (y - 1) = 1 := by omega
    simpa [Q, B, hs, hysub] using htmp
  have hc4 : c = 4 * d := by
    dsimp [c, d]
    rw [show ell - 2 * k - 2 = (ell - 2 * k - 4) + 2 by omega, pow_add]
    norm_num
    rw [Nat.mul_comm]
  have hm_pos : 0 < m := by
    dsimp [m]
    positivity
  have hm_le_B : m ≤ B := by
    have hpowW : 2 ^ (W k) = 2 ^ ell * m := by
      calc
        2 ^ (W k) = 2 ^ (ell + (W k - ell)) := by
          congr 1
          have hybw : ell ≤ W k := bw_le_of_lt_pow (Nat.ne_of_gt hy0) hyW
          omega
        _ = 2 ^ ell * 2 ^ (W k - ell) := by rw [pow_add]
    by_contra hmgt
    have hlt : y * B < y * m := Nat.mul_lt_mul_of_pos_left (lt_of_not_ge hmgt) hy0
    have hlt' : y * m < 2 ^ ell * m := Nat.mul_lt_mul_of_pos_right hbw.2 hm_pos
    have : y * B < 2 ^ (W k) := by
      calc
        y * B < y * m := hlt
        _ < 2 ^ ell * m := hlt'
        _ = 2 ^ (W k) := by rw [← hpowW]
    rw [hB] at this
    omega
  have hB_lt_2m : B < 2 * m := by
    have hyge : 2 ^ (ell - 1) + 1 ≤ y := Nat.succ_le_of_lt hygt
    have hpowW : 2 ^ (W k) = 2 ^ ell * m := by
      calc
        2 ^ (W k) = 2 ^ (ell + (W k - ell)) := by
          congr 1
          have hybw : ell ≤ W k := bw_le_of_lt_pow (Nat.ne_of_gt hy0) hyW
          omega
        _ = 2 ^ ell * 2 ^ (W k - ell) := by rw [pow_add]
    by_contra hBge
    have htwom_le : 2 * m ≤ B := le_of_not_gt hBge
    have hyBge : y * (2 * m) ≤ y * B := Nat.mul_le_mul_left _ htwom_le
    have hlower : 2 ^ (W k) + 2 * m ≤ y * B := by
      have hcalc : (2 ^ (ell - 1) + 1) * (2 * m) = 2 ^ (W k) + 2 * m := by
        calc
          (2 ^ (ell - 1) + 1) * (2 * m) = 2 ^ (ell - 1) * (2 * m) + 2 * m := by
            rw [Nat.add_mul, one_mul]
          _ = (2 ^ (ell - 1) * 2) * m + 2 * m := by rw [Nat.mul_assoc]
          _ = 2 ^ ell * m + 2 * m := by
            rw [show 2 ^ (ell - 1) * 2 = 2 ^ ell by
              rw [show ell = (ell - 1) + 1 by omega, pow_add]
              norm_num]
          _ = 2 ^ (W k) + 2 * m := by simpa [Nat.mul_comm] using congrArg (fun t => t + 2 * m) hpowW.symm
      calc
        2 ^ (W k) + 2 * m = (2 ^ (ell - 1) + 1) * (2 * m) := hcalc.symm
        _ ≤ y * (2 * m) := Nat.mul_le_mul_right _ hyge
        _ ≤ y * B := hyBge
    rw [hB] at hlower
    omega
  have hinner : (x k y * z1 k y) / 2 ^ (2 * k - 4) = 4 * (2 ^ (W k) + 1) := by
    calc
      (x k y * z1 k y) / 2 ^ (2 * k - 4) = (y * z1 k y) / d := by
            simpa [d, ell] using (inner_late_high (k := k) (y := y) hyW hℓ)
      _ = (y * (c * B)) / d := by rw [hz1eq]
      _ = (y * ((4 * d) * B)) / d := by rw [hc4]
      _ = (d * (4 * (y * B))) / d := by ac_rfl
      _ = 4 * (y * B) := by rw [Nat.mul_div_right _ (show 0 < d by positivity)]
      _ = 4 * (2 ^ (W k) + 1) := by rw [hB]
  have hfirst :
      2 ^ (2 * k + 3) * z1 k y = 2 ^ (ell + 1) * B := by
    have hsum : (2 * k + 3) + (ell - 2 * k - 2) = ell + 1 := by
      have hle : 2 * k + 2 ≤ ell := by omega
      calc
        (2 * k + 3) + (ell - 2 * k - 2) = 1 + ((2 * k + 2) + (ell - (2 * k + 2))) := by
          rw [show ell - 2 * k - 2 = ell - (2 * k + 2) by omega]
          omega
        _ = 1 + ell := by omega
        _ = ell + 1 := by omega
    calc
      2 ^ (2 * k + 3) * z1 k y = (2 ^ (2 * k + 3) * c) * B := by rw [hz1eq]; ac_rfl
      _ = 2 ^ ((2 * k + 3) + (ell - 2 * k - 2)) * B := by
            dsimp [c]
            rw [← pow_add]
      _ = 2 ^ (ell + 1) * B := by rw [hsum]
  have hsecond :
      (z1 k y * (4 * (2 ^ (W k) + 1))) / 2 ^ (2 * k) = 2 ^ ell * B + 1 := by
    rw [hz1eq]
    have hybw : ell ≤ W k := bw_le_of_lt_pow (Nat.ne_of_gt hy0) hyW
    have hpow2k : 2 ^ (2 * k) = 2 ^ (ell - 2 * k) * m := by
      have hle2k : 2 * k ≤ ell := by omega
      have hexp : (ell - 2 * k) + (W k - ell) = 2 * k := by
        have hsum : ((ell - 2 * k) + (W k - ell)) + 2 * k = W k := by
          calc
            ((ell - 2 * k) + (W k - ell)) + 2 * k
                = (ell - 2 * k + 2 * k) + (W k - ell) := by omega
            _ = ell + (W k - ell) := by rw [Nat.sub_add_cancel hle2k]
            _ = W k := by simpa [Nat.add_comm] using (Nat.sub_add_cancel hybw)
        have hW4 : W k = 2 * k + 2 * k := by
          unfold W
          ring
        have : ((ell - 2 * k) + (W k - ell)) + 2 * k = 2 * k + 2 * k := by
          rw [hsum, hW4]
        exact Nat.add_right_cancel (by simpa [Nat.add_assoc] using this)
      calc
        2 ^ (2 * k) = 2 ^ ((ell - 2 * k) + (W k - ell)) := by rw [hexp]
        _ = 2 ^ (ell - 2 * k) * 2 ^ (W k - ell) := by rw [pow_add]
        _ = 2 ^ (ell - 2 * k) * m := by rfl
    have hc4' : c * 4 = 2 ^ (ell - 2 * k) := by
      dsimp [c]
      rw [show ell - 2 * k = (ell - 2 * k - 2) + 2 by omega, pow_add]
      norm_num
    have hquot :
        ((B * (2 ^ (W k) + 1)) * 2 ^ (ell - 2 * k)) / 2 ^ (2 * k) =
          (B * (2 ^ (W k) + 1)) / m := by
      rw [hpow2k]
      simpa [Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using
        (Nat.mul_div_mul_left (B * (2 ^ (W k) + 1)) m (show 0 < 2 ^ (ell - 2 * k) by positivity))
    calc
      (c * B * (4 * (2 ^ (W k) + 1))) / 2 ^ (2 * k)
          = ((B * (2 ^ (W k) + 1)) * 2 ^ (ell - 2 * k)) / 2 ^ (2 * k) := by
              calc
                (c * B * (4 * (2 ^ (W k) + 1))) / 2 ^ (2 * k)
                    = (B * (2 ^ (W k) + 1) * (c * 4)) / 2 ^ (2 * k) := by ac_rfl
                _ = (B * (2 ^ (W k) + 1) * 2 ^ (ell - 2 * k)) / 2 ^ (2 * k) := by
                      rw [hc4']
                _ = ((B * (2 ^ (W k) + 1)) * 2 ^ (ell - 2 * k)) / 2 ^ (2 * k) := by ac_rfl
      _ = (B * (2 ^ (W k) + 1)) / m := hquot
      _ = 2 ^ ell * B + 1 := by
            have hpowW : 2 ^ (W k) = 2 ^ ell * m := by
              calc
                2 ^ (W k) = 2 ^ (ell + (W k - ell)) := by
                  congr 1
                  omega
                _ = 2 ^ ell * 2 ^ (W k - ell) := by rw [pow_add]
                _ = 2 ^ ell * m := by rfl
            have hrem : B - m < m := by omega
            have hdecomp :
                B * (2 ^ (W k) + 1) = (B - m) + (2 ^ ell * B + 1) * m := by
              calc
                B * (2 ^ (W k) + 1) = B * 2 ^ (W k) + B := by rw [Nat.mul_add, Nat.mul_one]
                _ = B * (2 ^ ell * m) + B := by rw [hpowW]
                _ = (2 ^ ell * B) * m + B := by ac_rfl
                _ = (2 ^ ell * B) * m + ((B - m) + m) := by rw [Nat.sub_add_cancel hm_le_B]
                _ = (B - m) + (2 ^ ell * B + 1) * m := by
                      rw [Nat.add_mul, one_mul]
                      ac_rfl
            calc
              (B * (2 ^ (W k) + 1)) / m = ((B - m) + (2 ^ ell * B + 1) * m) / m := by rw [hdecomp]
              _ = 2 ^ ell * B + 1 := by
                    simpa using (div_eq_of_add_mul_lt (d := m) (q := 2 ^ ell * B + 1) (r := B - m) hm_pos hrem)
  have hz2 :
      z2 k y = Q * 2 ^ ell + (2 ^ ell - 1) := by
    calc
      z2 k y = 2 ^ (ell + 1) * B - (2 ^ ell * B + 1) := by
        simp [z2, hinner, hfirst, hsecond]
      _ = 2 ^ ell * B - 1 := by
        rw [show 2 ^ (ell + 1) * B = 2 ^ ell * B + 2 ^ ell * B by
          rw [show ell + 1 = ell + 1 by rfl, pow_add]
          norm_num
          ring]
        omega
      _ = Q * 2 ^ ell + (2 ^ ell - 1) := by
        dsimp [B]
        have hone : 1 ≤ 2 ^ ell := Nat.succ_le_of_lt (by positivity)
        calc
          2 ^ ell * (Q + 1) - 1 = (2 ^ ell * Q + 2 ^ ell) - 1 := by rw [Nat.mul_add, Nat.mul_one]
          _ = Q * 2 ^ ell + (2 ^ ell - 1) := by
                rw [Nat.add_sub_assoc hone]
                ac_rfl
  unfold q0
  rw [hz2, show Q * 2 ^ ell + (2 ^ ell - 1) = (2 ^ ell - 1) + Q * 2 ^ ell by ac_rfl]
  simpa using
    (div_eq_of_add_mul_lt (d := 2 ^ ell) (q := Q) (r := 2 ^ ell - 1)
      (by positivity) (Nat.pred_lt (Nat.ne_of_gt (show 0 < 2 ^ ell by positivity))))

lemma resonant_exact_case_proved {k y : ℕ} (hk : 3 ≤ k) (hyW : y < 2 ^ (W k))
    (hℓ : 2 * k + 4 < bw y)
    (hpow : ¬ ∃ m : ℕ, y = 2 ^ m)
    (hs : 2 ^ (W k) % y = y - 1)
    (hz1eq : z1 k y = 2 ^ (bw y - 2 * k - 2) * (2 ^ (W k) / y + 1)) :
    Good k y := by
  set ell : ℕ := bw y
  set Q : ℕ := 2 ^ (W k) / y
  have hy0 : 0 < y := by
    cases y with
    | zero =>
        have hell : ell = 1 := by decide
        omega
    | succ _ => positivity
  have hy : 1 < y := by
    have hbw := bw_bounds (Nat.ne_of_gt hy0)
    have htwo : 2 ≤ 2 ^ (ell - 1) := by
      have : 2 ^ 1 ≤ 2 ^ (ell - 1) := by
        exact Nat.pow_le_pow_right (by decide : 0 < 2) (by omega)
      simpa using this
    exact lt_of_lt_of_le (by decide : 1 < 2) (le_trans htwo hbw.1)
  have hq0 : q0 k y = Q := by
    simpa [Q] using resonant_exact_case_q0 hk hyW hℓ hpow hs hz1eq
  exact q0_eq_floor_implies_Good hy (by simpa [Q] using hq0)

lemma resonant_pred_case_q0_bounds {k y : ℕ} (hk : 3 ≤ k) (hyW : y < 2 ^ (W k))
    (hℓ : 2 * k + 4 < bw y)
    (hs : 2 ^ (W k) % y = y - 1)
    (hz1pred : z1 k y = 2 ^ (bw y - 2 * k - 2) * (2 ^ (W k) / y + 1) - 1) :
    2 ^ (W k) / y - 1 ≤ q0 k y ∧ q0 k y ≤ 2 ^ (W k) / y := by
  set ell : ℕ := bw y
  set Q : ℕ := 2 ^ (W k) / y
  set c : ℕ := 2 ^ (ell - 2 * k - 2)
  set a : ℕ := 2 ^ (ell - 4)
  set P : ℕ := 2 ^ (2 * ell - 4)
  set D : ℕ := Nat.dist (y * z1 k y) (2 ^ (2 * k + ell - 2))
  set M : ℕ := (2 ^ (W k + 2 * ell - 4) - D ^ 2) / y
  set s : ℕ := 2 ^ (W k) % y
  have hy0 : 0 < y := by
    cases y with
    | zero =>
        have hell : ell = 1 := by decide
        omega
    | succ _ => positivity
  have hbw := bw_bounds (Nat.ne_of_gt hy0)
  have hc_lt_y : c < y := by
    have hc_le : c ≤ 2 ^ (ell - 5) := by
      dsimp [c]
      exact Nat.pow_le_pow_right (by decide : 0 < 2) (by omega)
    calc
      c ≤ 2 ^ (ell - 5) := hc_le
      _ < 2 ^ (ell - 1) := by
            have : 2 ^ (ell - 5) * 16 = 2 ^ (ell - 1) := by
              rw [show ell - 1 = (ell - 5) + 4 by omega, pow_add]
              norm_num
            have h15pos : 0 < 15 * 2 ^ (ell - 5) := by positivity
            calc
              2 ^ (ell - 5) < 2 ^ (ell - 5) * 16 := by
                    calc
                      2 ^ (ell - 5) < 2 ^ (ell - 5) + 15 * 2 ^ (ell - 5) := Nat.lt_add_of_pos_right h15pos
                      _ = 2 ^ (ell - 5) * 16 := by ring
              _ = 2 ^ (ell - 1) := this
      _ ≤ y := hbw.1
  have hB : y * (Q + 1) = 2 ^ (W k) + 1 := by
    have htmp := resonant_q_succ_mul (k := k) (y := y) hy0
    have hysub : y - (y - 1) = 1 := by omega
    change y * (2 ^ (W k) / y + 1) = 2 ^ (W k) + 1
    simpa [Q, s, hs, hysub] using htmp
  have hS : 2 ^ (2 * k + ell - 2) = y * (c * (Q + 1)) - c := by
    calc
      2 ^ (2 * k + ell - 2) = c * 2 ^ (W k) := by
            dsimp [c]
            rw [show 2 * k + ell - 2 = (ell - 2 * k - 2) + W k by
              unfold W
              omega, pow_add]
      _ = c * (y * (Q + 1) - 1) := by
            have hQsucc' : 2 ^ (W k) = y * (Q + 1) - 1 := by omega
            rw [hQsucc']
      _ = y * (c * (Q + 1)) - c := by
            rw [Nat.mul_sub_left_distrib, Nat.mul_one]
            ac_rfl
  have hD : D = y - c := by
    have hcB_pos : 0 < c * (Q + 1) := by
      dsimp [c]
      positivity
    have hy_le_ycB : y ≤ y * (c * (Q + 1)) := by
      simpa [Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using
        (Nat.le_mul_of_pos_right y hcB_pos)
    have hyz : y * z1 k y = y * (c * (Q + 1)) - y := by
      rw [hz1pred, Nat.mul_sub_left_distrib, Nat.mul_one]
    have hleS : y * z1 k y ≤ 2 ^ (2 * k + ell - 2) := by
      rw [hyz, hS]
      exact Nat.sub_le_sub_left (le_of_lt hc_lt_y) _
    dsimp [D]
    rw [Nat.dist_eq_sub_of_le hleS, hS, hyz]
    omega
  have h2a_leDsq : y * (2 * a) ≤ D ^ 2 := by
    have hDlower : 15 * 2 ^ (ell - 5) ≤ D := by
      have hc_le : c ≤ 2 ^ (ell - 5) := by
        dsimp [c]
        exact Nat.pow_le_pow_right (by decide : 0 < 2) (by omega)
      have hsum_le : 15 * 2 ^ (ell - 5) + c ≤ y := by
        calc
          15 * 2 ^ (ell - 5) + c ≤ 15 * 2 ^ (ell - 5) + 2 ^ (ell - 5) := Nat.add_le_add_left hc_le _
          _ = 16 * 2 ^ (ell - 5) := by ring
          _ = 2 ^ (ell - 1) := by
                rw [show ell - 1 = (ell - 5) + 4 by omega, pow_add]
                norm_num
                rw [Nat.mul_comm]
          _ ≤ y := hbw.1
      rw [hD]
      exact (Nat.le_sub_iff_add_le (le_of_lt hc_lt_y)).2 hsum_le
    have hDsqbig : 2 ^ (2 * ell - 3) ≤ D ^ 2 := by
      have hDsq225 : 225 * 2 ^ (2 * ell - 10) ≤ D ^ 2 := by
        have h15sq : (15 * 2 ^ (ell - 5)) ^ 2 = 225 * 2 ^ (2 * ell - 10) := by
          have hpowsq : 2 ^ (ell - 5) * 2 ^ (ell - 5) = 2 ^ (2 * ell - 10) := by
            rw [← pow_add]
            congr 1
            omega
          calc
            (15 * 2 ^ (ell - 5)) ^ 2 = (15 * 15) * (2 ^ (ell - 5) * 2 ^ (ell - 5)) := by
              rw [pow_two]
              ring
            _ = 225 * 2 ^ (2 * ell - 10) := by
              rw [hpowsq]
        calc
          225 * 2 ^ (2 * ell - 10) = (15 * 2 ^ (ell - 5)) ^ 2 := h15sq.symm
          _ ≤ D ^ 2 := by
                exact Nat.pow_le_pow_left hDlower 2
      have h128 : 2 ^ (2 * ell - 3) = 128 * 2 ^ (2 * ell - 10) := by
        rw [show 2 * ell - 3 = (2 * ell - 10) + 7 by omega, pow_add]
        norm_num
        ac_rfl
      calc
        2 ^ (2 * ell - 3) = 128 * 2 ^ (2 * ell - 10) := h128
        _ ≤ 225 * 2 ^ (2 * ell - 10) := by
              exact Nat.mul_le_mul_right _ (show 128 ≤ 225 by decide)
        _ ≤ D ^ 2 := hDsq225
    have hy2a : y * (2 * a) < 2 ^ (2 * ell - 3) := by
      dsimp [a]
      calc
        y * (2 * 2 ^ (ell - 4)) < 2 ^ ell * (2 * 2 ^ (ell - 4)) := Nat.mul_lt_mul_of_pos_right hbw.2 (by positivity)
        _ = 2 ^ (2 * ell - 3) := by
              rw [show 2 * 2 ^ (ell - 4) = 2 ^ (ell - 3) by
                rw [show ell - 3 = (ell - 4) + 1 by omega, pow_add]
                norm_num
                rw [Nat.mul_comm]]
              rw [← pow_add]
              congr 1
              omega
    exact le_trans (le_of_lt hy2a) hDsqbig
  have hpowWP : 2 ^ (W k + 2 * ell - 4) = 2 ^ (W k) * P := by
    dsimp [P]
    rw [show W k + 2 * ell - 4 = W k + (2 * ell - 4) by omega, pow_add]
  have hdivmod : 2 ^ (W k) = Q * y + s := by
    dsimp [Q, s]
    simpa [Nat.mul_comm] using (Nat.div_add_mod' (2 ^ (W k)) y).symm
  have hMupper2 :
      M + 2 * a ≤ (Q + 1) * P := by
    have hsP : s * P ≤ y * (P - 2 * a) + D ^ 2 := by
      have haux : y * (2 * a) ≤ P + D ^ 2 := le_trans h2a_leDsq (Nat.le_add_left _ _)
      have hsP' : y * P - P ≤ y * (P - 2 * a) + D ^ 2 := by
        rw [Nat.mul_sub_left_distrib]
        omega
      simpa [s, hs, Nat.sub_mul, one_mul] using hsP'
    have hMupper : M ≤ Q * P + (P - 2 * a) := by
      apply Nat.div_le_of_le_mul
      calc
        2 ^ (W k + 2 * ell - 4) - D ^ 2 = y * (Q * P) + s * P - D ^ 2 := by
          rw [hpowWP, hdivmod, Nat.add_mul]
          ac_rfl
        _ ≤ y * (Q * P) + y * (P - 2 * a) := by omega
        _ = y * (Q * P + (P - 2 * a)) := by rw [Nat.mul_add]
    have h2a_le_P : 2 * a ≤ P := by
      dsimp [a, P]
      rw [show 2 * ell - 4 = ell + (ell - 4) by omega, pow_add]
      have h2le : 2 ≤ 2 ^ ell := by
        have : 2 ^ 1 ≤ 2 ^ ell := by
          exact Nat.pow_le_pow_right (by decide : 0 < 2) (by omega)
        simpa using this
      calc
        2 * 2 ^ (ell - 4) ≤ 2 ^ ell * 2 ^ (ell - 4) := by
              simpa [Nat.mul_comm] using Nat.mul_le_mul_right (2 ^ (ell - 4)) h2le
        _ = 2 ^ ell * 2 ^ (ell - 4) := rfl
    have hsum : (Q * P + (P - 2 * a)) + 2 * a = Q * P + P := by
      calc
        (Q * P + (P - 2 * a)) + 2 * a = Q * P + ((P - 2 * a) + 2 * a) := by ac_rfl
        _ = Q * P + P := by rw [Nat.sub_add_cancel h2a_le_P]
    calc
      M + 2 * a ≤ (Q * P + (P - 2 * a)) + 2 * a := Nat.add_le_add_right hMupper _
      _ = Q * P + P := hsum
      _ = (Q + 1) * P := by rw [Nat.add_mul, one_mul]
  exact q0_late_high_bounds_of_Mupper2 (k := k) (y := y) hk hyW hℓ (by
    simpa [ell, a, P, D, M, Q] using hMupper2)

lemma resonant_pred_case_proved {k y : ℕ} (hk : 3 ≤ k) (hyW : y < 2 ^ (W k))
    (hℓ : 2 * k + 4 < bw y)
    (hpow : ¬ ∃ m : ℕ, y = 2 ^ m)
    (hs : 2 ^ (W k) % y = y - 1)
    (hz1pred : z1 k y = 2 ^ (bw y - 2 * k - 2) * (2 ^ (W k) / y + 1) - 1) :
    Good k y := by
  set ell : ℕ := bw y
  set Q : ℕ := 2 ^ (W k) / y
  have hy0 : 0 < y := by
    cases y with
    | zero =>
        have hell : ell = 1 := by decide
        omega
    | succ _ => positivity
  have hy : 1 < y := by
    have hbw := bw_bounds (Nat.ne_of_gt hy0)
    have htwo : 2 ≤ 2 ^ (ell - 1) := by
      have : 2 ^ 1 ≤ 2 ^ (ell - 1) := by
        exact Nat.pow_le_pow_right (by decide : 0 < 2) (by omega)
      simpa using this
    exact lt_of_lt_of_le (by decide : 1 < 2) (le_trans htwo hbw.1)
  have hbounds := resonant_pred_case_q0_bounds hk hyW hℓ hs hz1pred
  have hlow : Q - 1 ≤ q0 k y := by simpa [Q] using hbounds.1
  have hupp : q0 k y ≤ Q := by simpa [Q] using hbounds.2
  have hq0 : q0 k y = Q ∨ q0 k y = Q - 1 := by omega
  rcases hq0 with hq0 | hq0
  · exact q0_eq_floor_implies_Good hy (by simpa [Q] using hq0)
  · exact q0_eq_floor_pred_implies_Good hy hyW hpow (by simpa [Q] using hq0)

theorem late_high_resonant_small_proved {k y : ℕ}
    (hk : 3 ≤ k)
    (hyW : y < 2 ^ (W k))
    (hℓ : 2 * k + 4 < bw y)
    (_hb8 : 8 ≤ bw y)
    (hpow : ¬ ∃ m : ℕ, y = 2 ^ m)
    (hs : 2 ^ (W k) % y = y - 1)
    (hDsmall : Nat.dist (y * z1 k y) (2 ^ (2 * k + bw y - 2)) < y) :
    Good k y := by
  have hz1cases := resonant_small_z1_cases hk hyW hℓ hs hDsmall
  rcases hz1cases with hz1 | hz1
  · exact resonant_pred_case_proved hk hyW hℓ hpow hs hz1
  · exact resonant_exact_case_proved hk hyW hℓ hpow hs hz1

end ReciprocalLateHighComplete

namespace Reciprocal

open ReciprocalLateHighComplete

/--
TeX correspondence: final proof assembly.

`late_high_proved` closes the last late-high branch, and
`thm_kge5_complete` / `thm_all_k_complete` correspond to the paper's
Theorem `kge5` and Theorem `all-k`.
-/
theorem late_high_proved {k y : ℕ} :
    3 ≤ k →
    y < 2 ^ (W k) →
    2 * k + 4 < bw y →
    8 ≤ bw y →
    (¬ ∃ m : ℕ, y = 2 ^ m) →
    Good k y := by
  intro hk hyW hℓ hb8 hpow
  exact late_high_reduced_to_resonant_small
    (fun hk3 hyW3 hℓ3 hb83 hpow3 hs hDsmall =>
      late_high_resonant_small_proved hk3 hyW3 hℓ3 hb83 hpow3 hs hDsmall)
    hk hyW hℓ hb8 hpow

theorem thm_kge5_complete {k y : ℕ} :
    5 ≤ k → 1 ≤ y → y < 2 ^ (W k) → Good k y := by
  intro hk hy1 hy2
  exact thm_kge5 tiny_proved
    (fun hk3 hygt1 hyk hb4 hpow => ReciprocalLowExact.low_exact_proved hk3 hygt1 hyk hb4 hpow)
    (fun hk3 hyW hreg => mid_high_reduced_to_late_high late_high_proved hk3 hyW hreg)
    hk hy1 hy2

theorem thm_all_k_complete {k y : ℕ} :
    3 ≤ k → 1 ≤ y → y < 2 ^ (W k) → Good k y := by
  intro hk hy1 hy2
  exact thm_all_k tiny_proved
    (fun hk3 hygt1 hyk hb4 hpow => ReciprocalLowExact.low_exact_proved hk3 hygt1 hyk hb4 hpow)
    (fun hk3 hyW hreg => mid_high_reduced_to_late_high late_high_proved hk3 hyW hreg)
    hk hy1 hy2

end Reciprocal
