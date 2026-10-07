import Mathlib

namespace LUTNR

/-
Lean companion for `scripts/lean/LUTNR.tex`.

The paper's parameter `L` is specialized to the even-width case `L = 2 * n`,
so every occurrence of `L / 2` in the TeX becomes `n` definitionally in Lean.
-/

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

lemma bw_bounds {y : ℕ} (hy : y ≠ 0) : 2 ^ (bw y - 1) ≤ y ∧ y < 2 ^ bw y := by
  constructor
  · simpa [bw] using Nat.log2_self_le hy
  · simpa [bw] using (Nat.lt_log2_self (n := y))

/-- The TeX parameter `L` is written as `2 * n` so that `L / 2 = n` definitionally. -/
def L (n : ℕ) : ℕ := 2 * n

/-- TeX: `clz = L - (⌊log₂ y⌋ + 1) = L - bw y`. -/
def clz (n y : ℕ) : ℕ := L n - bw y

/-- TeX: `yy = y * 2^clz`. -/
def yy (n y : ℕ) : ℕ := y * 2 ^ clz n y

/-- TeX: `hy = ⌊yy / 2^(L/2)⌋`, specialized to `⌊yy / 2^n⌋`. -/
def hy (n y : ℕ) : ℕ := yy n y / 2 ^ n

/-- Low bits discarded when passing from `yy` to `hy`. -/
def rem (n y : ℕ) : ℕ := yy n y % 2 ^ n

/-- TeX proof notation: `z' = ⌊hz / 2^(L - clz)⌋`. -/
def zPre (y hz : ℕ) : ℕ := hz / 2 ^ bw y

/-- Final LUTNR output after the one-step downward correction. -/
def zOut (n y hz : ℕ) : ℕ :=
  let z' := zPre y hz
  if y * z' < 2 ^ L n then z' else z' - 1

lemma L_pos {n : ℕ} (hn : 0 < n) : 0 < L n := by
  unfold L
  omega

lemma bw_le_L {n y : ℕ} (hy1 : 1 ≤ y) (hy2 : y < 2 ^ L n) : bw y ≤ L n := by
  have hy0 : y ≠ 0 := by omega
  exact bw_le_of_lt_pow hy0 hy2

lemma clz_add_bw {n y : ℕ} (hy1 : 1 ≤ y) (hy2 : y < 2 ^ L n) : clz n y + bw y = L n := by
  unfold clz
  have hbw : bw y ≤ L n := bw_le_L hy1 hy2
  omega

/-- TeX rearrangement used repeatedly: `L - clz = bw y`. -/
lemma L_sub_clz_eq_bw {n y : ℕ} (hy1 : 1 ≤ y) (hy2 : y < 2 ^ L n) :
    L n - clz n y = bw y := by
  unfold clz
  have hbw : bw y ≤ L n := bw_le_L hy1 hy2
  omega

lemma yy_eq_mul_pow {n y : ℕ} : yy n y = y * 2 ^ clz n y := rfl

lemma hy_def {n y : ℕ} : hy n y = yy n y / 2 ^ n := rfl

lemma rem_def {n y : ℕ} : rem n y = yy n y % 2 ^ n := rfl

lemma yy_eq_hy_mul_pow_add_rem {n y : ℕ} :
    yy n y = hy n y * 2 ^ n + rem n y := by
  unfold hy rem
  simpa [Nat.add_comm, Nat.mul_comm] using (Nat.mod_add_div (yy n y) (2 ^ n)).symm

lemma rem_lt_pow {n y : ℕ} : rem n y < 2 ^ n := by
  unfold rem
  exact Nat.mod_lt _ (by positivity)

lemma yy_bounds {n y : ℕ} (hy1 : 1 ≤ y) (hy2 : y < 2 ^ L n) :
    2 ^ (L n - 1) ≤ yy n y ∧ yy n y < 2 ^ L n := by
  have hy0 : y ≠ 0 := by omega
  have hbw := bw_bounds hy0
  have hleL : bw y ≤ L n := bw_le_L hy1 hy2
  have hadd : clz n y + bw y = L n := clz_add_bw hy1 hy2
  have hbwpos : 0 < bw y := bw_pos y
  constructor
  · calc
      2 ^ (L n - 1) = 2 ^ (clz n y + (bw y - 1)) := by
        have hEq : L n - 1 = clz n y + (bw y - 1) := by
          omega
        rw [hEq]
      _ = 2 ^ ((bw y - 1) + clz n y) := by rw [Nat.add_comm]
      _ = 2 ^ (bw y - 1) * 2 ^ clz n y := by rw [pow_add]
      _ ≤ y * 2 ^ clz n y := Nat.mul_le_mul_right _ hbw.1
      _ = yy n y := rfl
  · calc
      yy n y = y * 2 ^ clz n y := rfl
      _ < 2 ^ bw y * 2 ^ clz n y := Nat.mul_lt_mul_of_pos_right hbw.2 (by positivity)
      _ = 2 ^ (bw y + clz n y) := by rw [← pow_add]
      _ = 2 ^ L n := by rw [Nat.add_comm, hadd]

lemma hy_mul_pow_le_yy {n y : ℕ} : hy n y * 2 ^ n ≤ yy n y := by
  rw [yy_eq_hy_mul_pow_add_rem]
  exact Nat.le_add_right _ _

lemma yy_lt_hy_succ_mul_pow {n y : ℕ} : yy n y < (hy n y + 1) * 2 ^ n := by
  calc
    yy n y = hy n y * 2 ^ n + rem n y := yy_eq_hy_mul_pow_add_rem
    _ < hy n y * 2 ^ n + 2 ^ n := Nat.add_lt_add_left rem_lt_pow _
    _ = (hy n y + 1) * 2 ^ n := by rw [Nat.add_mul, one_mul]

lemma hy_lower {n y : ℕ} (hn : 0 < n) (hy1 : 1 ≤ y) (hy2 : y < 2 ^ L n) :
    2 ^ (n - 1) ≤ hy n y := by
  have hlt : 2 ^ (L n - 1) < (hy n y + 1) * 2 ^ n := by
    exact lt_of_le_of_lt (yy_bounds hy1 hy2).1 yy_lt_hy_succ_mul_pow
  have hlt' : 2 ^ (n - 1) * 2 ^ n < (hy n y + 1) * 2 ^ n := by
    simpa [show L n - 1 = (n - 1) + n by
      unfold L
      omega, pow_add]
      using hlt
  have hlt'' : 2 ^ (n - 1) < hy n y + 1 := Nat.lt_of_mul_lt_mul_right hlt'
  omega

lemma hy_pos {n y : ℕ} (hn : 0 < n) (hy1 : 1 ≤ y) (hy2 : y < 2 ^ L n) : 0 < hy n y := by
  exact lt_of_lt_of_le (by positivity : 0 < 2 ^ (n - 1)) (hy_lower hn hy1 hy2)

lemma hy_hz_lt_pow {n y hz : ℕ} (hupper : hy n y * 2 ^ n * hz < 2 ^ (2 * L n)) :
    hy n y * hz < 2 ^ (3 * n) := by
  have h' : (hy n y * hz) * 2 ^ n < 2 ^ (3 * n) * 2 ^ n := by
    simpa [Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm, show 2 * L n = 3 * n + n by
      unfold L
      omega, pow_add]
      using hupper
  exact Nat.lt_of_mul_lt_mul_right h'

lemma rem_eq_zero_of_n_le_clz {n y : ℕ} (hclz : n ≤ clz n y) : rem n y = 0 := by
  unfold rem yy
  have hdvd : 2 ^ n ∣ 2 ^ clz n y := by
    refine ⟨2 ^ (clz n y - n), ?_⟩
    rw [← pow_add]
    congr 1
    omega
  exact Nat.mod_eq_zero_of_dvd (dvd_mul_of_dvd_right hdvd y)

lemma hy_eq_div_of_clz_lt {n y : ℕ} (hclz : clz n y < n) :
    hy n y = y / 2 ^ (n - clz n y) := by
  unfold hy yy
  have hpow : 2 ^ n = 2 ^ (n - clz n y) * 2 ^ clz n y := by
    rw [← pow_add]
    congr 1
    omega
  rw [hpow]
  simpa using
    (Nat.mul_div_mul_right (m := 2 ^ clz n y) y (2 ^ (n - clz n y)) (by positivity : 0 < 2 ^ clz n y))

lemma rem_eq_mod_mul_pow_of_clz_lt {n y : ℕ} (hclz : clz n y < n) :
    rem n y = (y % 2 ^ (n - clz n y)) * 2 ^ clz n y := by
  let d := n - clz n y
  have hpow : 2 ^ n = 2 ^ d * 2 ^ clz n y := by
    rw [← pow_add]
    congr 1
    dsimp [d]
    omega
  have hydecomp : y * 2 ^ clz n y = ((y % 2 ^ d) * 2 ^ clz n y) + (y / 2 ^ d) * 2 ^ n := by
    calc
      y * 2 ^ clz n y = (y % 2 ^ d + 2 ^ d * (y / 2 ^ d)) * 2 ^ clz n y := by
        rw [Nat.mod_add_div]
      _ = ((y % 2 ^ d) * 2 ^ clz n y) + (2 ^ d * (y / 2 ^ d)) * 2 ^ clz n y := by
        rw [Nat.add_mul]
      _ = ((y % 2 ^ d) * 2 ^ clz n y) + (y / 2 ^ d) * 2 ^ n := by
        congr 1
        calc
          (2 ^ d * (y / 2 ^ d)) * 2 ^ clz n y = (y / 2 ^ d) * (2 ^ d * 2 ^ clz n y) := by
            ac_rfl
          _ = (y / 2 ^ d) * 2 ^ n := by rw [show 2 ^ d * 2 ^ clz n y = 2 ^ n by simpa [d] using hpow.symm]
  calc
    rem n y = yy n y % 2 ^ n := rfl
    _ = (y * 2 ^ clz n y) % 2 ^ n := rfl
    _ = (((y % 2 ^ d) * 2 ^ clz n y) + (y / 2 ^ d) * 2 ^ n) % 2 ^ n := by
      rw [hydecomp]
    _ = ((y % 2 ^ d) * 2 ^ clz n y) % 2 ^ n := by
      rw [Nat.mul_comm (y / 2 ^ d) (2 ^ n), Nat.add_mul_mod_self_left]
    _ = (y % 2 ^ d) * 2 ^ clz n y := by
      apply Nat.mod_eq_of_lt
      have hmod : y % 2 ^ d < 2 ^ d := Nat.mod_lt _ (by positivity)
      have hlt := Nat.mul_lt_mul_of_pos_right hmod (by positivity : 0 < 2 ^ clz n y)
      have hpow' : 2 ^ d * 2 ^ clz n y = 2 ^ n := by
        rw [← pow_add]
        congr 1
        dsimp [d]
        omega
      simpa [Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc, hpow'] using hlt

lemma rem_bound_of_clz_lt {n y : ℕ} (hclz : clz n y < n) :
    rem n y ≤ 2 ^ n - 2 ^ clz n y := by
  let d := n - clz n y
  rw [rem_eq_mod_mul_pow_of_clz_lt hclz]
  have hmod : y % 2 ^ d ≤ 2 ^ d - 1 := Nat.le_pred_of_lt (Nat.mod_lt _ (by positivity))
  calc
    (y % 2 ^ d) * 2 ^ clz n y ≤ (2 ^ d - 1) * 2 ^ clz n y := Nat.mul_le_mul_right _ hmod
    _ = 2 ^ n - 2 ^ clz n y := by
      rw [Nat.sub_mul, one_mul, show 2 ^ d * 2 ^ clz n y = 2 ^ n by
        rw [← pow_add]
        congr 1
        dsimp [d]
        omega]

lemma pow_sub_one_le_pow_double_sub_two (d : ℕ) (hd : 0 < d) :
    2 ^ d - 1 ≤ 2 ^ (2 * d - 2) := by
  cases d with
  | zero =>
      cases Nat.lt_asymm hd hd
  | succ d =>
      cases d with
      | zero =>
          norm_num
      | succ d' =>
          have hpow : 2 ^ (Nat.succ (Nat.succ d')) ≤ 2 ^ (Nat.succ (Nat.succ d') + d') := by
            exact Nat.pow_le_pow_right (by positivity) (Nat.le_add_right _ _)
          have hpow' : 2 ^ (Nat.succ (Nat.succ d')) ≤ 2 ^ (2 * Nat.succ (Nat.succ d') - 2) := by
            rw [show 2 * Nat.succ (Nat.succ d') - 2 = Nat.succ (Nat.succ d') + d' by omega]
            exact hpow
          exact le_trans (Nat.sub_le _ _) hpow'

lemma rem_mul_pow_le_y_mul_hy {n y : ℕ} (hn : 0 < n) (hy1 : 1 ≤ y) (hy2 : y < 2 ^ L n) :
    rem n y * 2 ^ n ≤ y * hy n y := by
  by_cases hclz : clz n y < n
  · let d := n - clz n y
    have hrempow : rem n y * 2 ^ n ≤ 2 ^ (2 * n + d - 2) := by
      have hremrepr : rem n y = (y % 2 ^ d) * 2 ^ clz n y := by
        simpa [d] using rem_eq_mod_mul_pow_of_clz_lt (n := n) (y := y) hclz
      have hmod : y % 2 ^ d ≤ 2 ^ d - 1 := Nat.le_pred_of_lt (Nat.mod_lt _ (by positivity))
      calc
        rem n y * 2 ^ n = (y % 2 ^ d) * 2 ^ (2 * n - d) := by
          rw [hremrepr]
          calc
            (y % 2 ^ d) * 2 ^ clz n y * 2 ^ n = (y % 2 ^ d) * (2 ^ clz n y * 2 ^ n) := by
              ac_rfl
            _ = (y % 2 ^ d) * 2 ^ (2 * n - d) := by
              congr 1
              rw [← pow_add]
              congr 1
              dsimp [d]
              omega
        _ ≤ (2 ^ d - 1) * 2 ^ (2 * n - d) := Nat.mul_le_mul_right _ hmod
        _ ≤ 2 ^ (2 * d - 2) * 2 ^ (2 * n - d) := by
          exact Nat.mul_le_mul_right _ (pow_sub_one_le_pow_double_sub_two d (by
            dsimp [d]
            omega))
        _ = 2 ^ (2 * n + d - 2) := by
          rw [← pow_add]
          congr 1
          dsimp [d]
          omega
    have hy0 : y ≠ 0 := by omega
    have hd : 0 < d := by
      dsimp [d]
      omega
    have hy_low : 2 ^ (n + d - 1) ≤ y := by
      have hbw := bw_bounds hy0
      have hbweq : bw y = n + d := by
        have hclzle : clz n y ≤ n := Nat.le_of_lt hclz
        have hsum : clz n y + bw y = 2 * n := by simpa [L] using clz_add_bw hy1 hy2
        dsimp [d]
        omega
      simpa [hbweq] using hbw.1
    have hhy_low : 2 ^ (n - 1) ≤ hy n y := hy_lower hn hy1 hy2
    have hprod : 2 ^ (2 * n + d - 2) ≤ y * hy n y := by
      calc
        2 ^ (2 * n + d - 2) = 2 ^ (n + d - 1) * 2 ^ (n - 1) := by
          rw [← pow_add]
          congr 1
          omega
        _ ≤ y * hy n y := Nat.mul_le_mul hy_low hhy_low
    exact le_trans hrempow hprod
  · have hclz' : n ≤ clz n y := by omega
    simp [rem_eq_zero_of_n_le_clz hclz']

def RequirementHyp (n y hz : ℕ) : Prop :=
  2 ^ (2 * L n) - hy n y * 2 ^ n ≤ hy n y * 2 ^ n * hz ∧
    hy n y * 2 ^ n * hz < 2 ^ (2 * L n)

/--
TeX hypothesis after Lemma `lem:requirement`:
`2^(2L) - y * 2^clz ≤ y * 2^clz * hz < 2^(2L) + y * 2^L`.
-/
def ApproxHyp (n y hz : ℕ) : Prop :=
  2 ^ (2 * L n) - yy n y ≤ yy n y * hz ∧
    yy n y * hz < 2 ^ (2 * L n) + y * 2 ^ L n

/-- TeX conclusion of Theorem `thm:lutnr`: `z ∈ [1, 2^L)` and `2^L - y ≤ y*z < 2^L`. -/
def GoodOutput (n y z : ℕ) : Prop :=
  1 ≤ z ∧ z < 2 ^ L n ∧ 2 ^ L n - y ≤ y * z ∧ y * z < 2 ^ L n

/-- Division with remainder for the pre-correction candidate. -/
lemma hz_eq_zPre_mul_pow_add_mod (y hz : ℕ) :
    hz = zPre y hz * 2 ^ bw y + hz % 2 ^ bw y := by
  unfold zPre
  simpa [Nat.add_comm, Nat.mul_comm] using (Nat.mod_add_div hz (2 ^ bw y)).symm

/-- Exact TeX denominator: `2^(L - clz) = 2^(bw y)`. -/
lemma zPre_eq_tex_div {n y hz : ℕ} (hy1 : 1 ≤ y) (hy2 : y < 2 ^ L n) :
    zPre y hz = hz / 2 ^ (L n - clz n y) := by
  rw [L_sub_clz_eq_bw hy1 hy2]
  rfl

/--
TeX-style decomposition
`hz = 2^(L-clz) * z' + r` with `z' = zPre y hz`.

This is the exact bridge between the theorem proof in `LUTNR.tex` and the Lean
definition `zPre`.
-/
lemma hz_eq_tex_shift_mul_zPre_add_mod {n y hz : ℕ} (hy1 : 1 ≤ y) (hy2 : y < 2 ^ L n) :
    hz = 2 ^ (L n - clz n y) * zPre y hz + hz % 2 ^ (L n - clz n y) := by
  rw [L_sub_clz_eq_bw hy1 hy2]
  simpa [Nat.mul_comm] using hz_eq_zPre_mul_pow_add_mod y hz

lemma zOut_eq_of_lt {n y hz : ℕ} (h : y * zPre y hz < 2 ^ L n) :
    zOut n y hz = zPre y hz := by
  simp [zOut, h]

lemma zOut_eq_of_not_lt {n y hz : ℕ} (h : ¬ y * zPre y hz < 2 ^ L n) :
    zOut n y hz = zPre y hz - 1 := by
  simp [zOut, h]

/--
TeX Lemma `lem:requirement`, specialized to the even-width case `L = 2 * n`.

This is the first arithmetic goal to discharge in Lean. The statement is kept
close to the paper, while `RequirementHyp` and `ApproxHyp` package the two
interval conditions.
-/
theorem requirement {n y hz : ℕ} (hn : 0 < n) (hy1 : 1 ≤ y) (hy2 : y < 2 ^ L n)
    (hreq : RequirementHyp n y hz) : ApproxHyp n y hz := by
  rcases hreq with ⟨hlower, hupper⟩
  constructor
  · have hsub : 2 ^ (2 * L n) - yy n y ≤ 2 ^ (2 * L n) - hy n y * 2 ^ n := by
      exact Nat.sub_le_sub_left hy_mul_pow_le_yy _
    exact le_trans hsub <| le_trans hlower <|
      Nat.mul_le_mul_right _ hy_mul_pow_le_yy
  · have hhyhz : hy n y * hz < 2 ^ (3 * n) := hy_hz_lt_pow hupper
    have hrem : rem n y * hz < y * 2 ^ L n := by
      have hmul :
          (rem n y * hz) * 2 ^ n ≤ y * (hy n y * hz) := by
        calc
          (rem n y * hz) * 2 ^ n = (rem n y * 2 ^ n) * hz := by ac_rfl
          _ ≤ (y * hy n y) * hz := Nat.mul_le_mul_right _ (rem_mul_pow_le_y_mul_hy hn hy1 hy2)
          _ = y * (hy n y * hz) := by ac_rfl
      have hyhz_mul : y * (hy n y * hz) < y * 2 ^ (3 * n) := by
        exact Nat.mul_lt_mul_of_pos_left hhyhz hy1
      have hlt : (rem n y * hz) * 2 ^ n < (y * 2 ^ L n) * 2 ^ n := by
        exact lt_of_le_of_lt hmul (by
          simpa [show 3 * n = L n + n by
            unfold L
            omega, pow_add, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
            using hyhz_mul)
      exact Nat.lt_of_mul_lt_mul_right hlt
    have hsplit : yy n y * hz = hy n y * 2 ^ n * hz + rem n y * hz := by
      calc
        yy n y * hz = (hy n y * 2 ^ n + rem n y) * hz := by rw [yy_eq_hy_mul_pow_add_rem]
        _ = hy n y * 2 ^ n * hz + rem n y * hz := by ring
    calc
      yy n y * hz = hy n y * 2 ^ n * hz + rem n y * hz := hsplit
      _ < 2 ^ (2 * L n) + y * 2 ^ L n := add_lt_add_of_lt_of_lt hupper hrem

/-- TeX label `lem:requirement`. -/
theorem lem_requirement {n y hz : ℕ} (hn : 0 < n) (hy1 : 1 ≤ y) (hy2 : y < 2 ^ L n)
    (hreq : RequirementHyp n y hz) : ApproxHyp n y hz :=
  requirement hn hy1 hy2 hreq

/--
TeX Theorem `thm:lutnr`, again written with `L = 2 * n`.

The output is `zOut n y hz`, which is the shifted candidate `zPre y hz`
followed by the final downward correction from the pseudocode.
-/
theorem lutnr_correct {n y hz : ℕ} (hn : 0 < n) (hy1 : 1 ≤ y) (hy2 : y < 2 ^ L n)
    (_hhz1 : 2 ^ n ≤ hz) (hhz2 : hz < 2 ^ (L n + 1))
    (happrox : ApproxHyp n y hz) :
    GoodOutput n y (zOut n y hz) := by
  rcases happrox with ⟨happrox1, happrox2⟩
  let z' := zPre y hz
  let r := hz % 2 ^ bw y
  have hhz : hz = z' * 2 ^ bw y + r := by
    simpa [z', r] using hz_eq_zPre_mul_pow_add_mod y hz
  have hr_lt : r < 2 ^ bw y := by
    dsimp [r]
    exact Nat.mod_lt _ (by positivity)
  have hclz : clz n y + bw y = L n := clz_add_bw hy1 hy2
  have hsplit : yy n y * hz = 2 ^ L n * (y * z') + y * 2 ^ clz n y * r := by
    calc
      yy n y * hz = (y * 2 ^ clz n y) * (z' * 2 ^ bw y + r) := by simp [yy, hhz]
      _ = y * z' * (2 ^ clz n y * 2 ^ bw y) + y * 2 ^ clz n y * r := by ring
      _ = y * z' * 2 ^ L n + y * 2 ^ clz n y * r := by rw [← pow_add, hclz]
      _ = 2 ^ L n * (y * z') + y * 2 ^ clz n y * r := by ac_rfl
  have hr_le : r ≤ 2 ^ bw y - 1 := Nat.le_pred_of_lt hr_lt
  have hterm_le : y * 2 ^ clz n y * r ≤ y * 2 ^ L n - yy n y := by
    calc
      y * 2 ^ clz n y * r ≤ y * 2 ^ clz n y * (2 ^ bw y - 1) := Nat.mul_le_mul_left _ hr_le
      _ = y * 2 ^ L n - yy n y := by
        calc
          y * 2 ^ clz n y * (2 ^ bw y - 1) = y * (2 ^ clz n y * (2 ^ bw y - 1)) := by
            ac_rfl
          _ = y * (2 ^ L n - 2 ^ clz n y) := by
            congr 1
            rw [Nat.mul_sub_left_distrib, mul_one, ← pow_add, hclz]
          _ = y * 2 ^ L n - yy n y := by
            rw [Nat.mul_sub_left_distrib, yy]
  have hyy_le : yy n y ≤ y * 2 ^ L n := by
    rw [yy]
    apply Nat.mul_le_mul_left
    exact Nat.pow_le_pow_right (by positivity) (by
      unfold clz
      exact Nat.sub_le _ _)
  have hlower_mul :
      2 ^ L n * 2 ^ L n ≤ 2 ^ L n * (y * z' + y) := by
    have hyy_le_big : yy n y ≤ 2 ^ (2 * L n) := by
      exact le_trans (le_of_lt (yy_bounds hy1 hy2).2)
        (Nat.pow_le_pow_right (by positivity) (by
          have hLpos : 0 < L n := L_pos hn
          omega))
    have hsum : 2 ^ (2 * L n) ≤ yy n y * hz + yy n y := by
      simpa [Nat.sub_add_cancel hyy_le_big] using Nat.add_le_add_right happrox1 (yy n y)
    have hsum' : yy n y * hz + yy n y ≤ 2 ^ L n * (y * z' + y) := by
      calc
        yy n y * hz + yy n y
            = (2 ^ L n * (y * z') + y * 2 ^ clz n y * r) + yy n y := by rw [hsplit]
        _ ≤ (2 ^ L n * (y * z') + (y * 2 ^ L n - yy n y)) + yy n y := by
          exact Nat.add_le_add_right (Nat.add_le_add_left hterm_le _) _
        _ = 2 ^ L n * (y * z') + y * 2 ^ L n := by
          rw [Nat.add_assoc, Nat.sub_add_cancel hyy_le]
        _ = 2 ^ L n * (y * z' + y) := by rw [Nat.mul_add, Nat.mul_comm (2 ^ L n) y]
    exact (by
      simpa [show 2 * L n = L n + L n by omega, pow_add] using le_trans hsum hsum')
  have hyz_lower : 2 ^ L n - y ≤ y * z' := by
    have htmp : 2 ^ L n ≤ y * z' + y := Nat.le_of_mul_le_mul_left hlower_mul (by positivity)
    omega
  have hyz_upper : y * z' < 2 ^ L n + y := by
    have hupper_mul : 2 ^ L n * (y * z') ≤ yy n y * hz := by
      rw [hsplit]
      exact Nat.le_add_right _ _
    have hupper_mul' : 2 ^ L n * (y * z') < 2 ^ L n * (2 ^ L n + y) := by
      exact lt_of_le_of_lt hupper_mul (by
        simpa [Nat.mul_add, show 2 * L n = L n + L n by omega, pow_add,
          Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm] using happrox2)
    have hupper_mul'' : (y * z') * 2 ^ L n < (2 ^ L n + y) * 2 ^ L n := by
      simpa [Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm] using hupper_mul'
    exact Nat.lt_of_mul_lt_mul_right hupper_mul''
  have hz'_lt : z' < 2 ^ L n := by
    dsimp [z', zPre]
    apply (Nat.div_lt_iff_lt_mul (by positivity : 0 < 2 ^ bw y)).2
    calc
      hz < 2 ^ (L n + 1) := hhz2
      _ = 2 ^ L n * 2 ^ 1 := by rw [pow_add]
      _ ≤ 2 ^ L n * 2 ^ bw y := Nat.mul_le_mul_left _
        (Nat.pow_le_pow_right (by positivity) (bw_pos y))
  by_cases hcorr : y * z' < 2 ^ L n
  · have hzout_eq : zOut n y hz = z' := by
      simpa [z'] using zOut_eq_of_lt (n := n) (y := y) (hz := hz) hcorr
    refine ⟨?_, ?_, ?_, ?_⟩
    · have hrespos : 0 < y * z' := lt_of_lt_of_le (Nat.sub_pos_of_lt hy2) hyz_lower
      have hzpos : 0 < z' := Nat.pos_of_mul_pos_left hrespos
      rw [hzout_eq]
      exact Nat.succ_le_of_lt hzpos
    · rw [hzout_eq]
      exact hz'_lt
    · rw [hzout_eq]
      exact hyz_lower
    · rw [hzout_eq]
      exact hcorr
  · have hyz_ge : 2 ^ L n ≤ y * z' := by omega
    have hzout_eq : zOut n y hz = z' - 1 := by
      simpa [z'] using zOut_eq_of_not_lt (n := n) (y := y) (hz := hz) hcorr
    have hz_gt1 : 1 < z' := by
      by_contra hz_le
      have hz_le' : z' ≤ 1 := Nat.not_lt.mp hz_le
      have hyz_le : y * z' ≤ y := by
        simpa using (Nat.mul_le_mul_left y hz_le')
      exact hcorr (lt_of_le_of_lt hyz_le hy2)
    refine ⟨?_, ?_, ?_, ?_⟩
    · have hz_ge2 : 2 ≤ z' := Nat.succ_le_of_lt hz_gt1
      have : 1 ≤ z' - 1 := by omega
      rw [hzout_eq]
      exact this
    · have hzpos : 0 < z' := lt_trans zero_lt_one hz_gt1
      have : z' - 1 < 2 ^ L n := lt_of_lt_of_le (Nat.sub_lt hzpos zero_lt_one) (Nat.le_of_lt hz'_lt)
      rw [hzout_eq]
      exact this
    · rw [hzout_eq, Nat.mul_sub_left_distrib, mul_one]
      exact Nat.sub_le_sub_right hyz_ge _
    · rw [hzout_eq, Nat.mul_sub_left_distrib, mul_one]
      omega

/-- TeX label `thm:lutnr`. -/
theorem thm_lutnr {n y hz : ℕ} (hn : 0 < n) (hy1 : 1 ≤ y) (hy2 : y < 2 ^ L n)
    (hhz1 : 2 ^ n ≤ hz) (hhz2 : hz < 2 ^ (L n + 1))
    (happrox : ApproxHyp n y hz) :
    GoodOutput n y (zOut n y hz) :=
  lutnr_correct hn hy1 hy2 hhz1 hhz2 happrox

/-- A concrete reciprocal candidate built from the truncated normalized input. -/
def hzFloor (n y : ℕ) : ℕ :=
  (2 ^ (2 * L n) - 1) / (hy n y * 2 ^ n)

lemma hzFloor_requirement {n y : ℕ} (hn : 0 < n) (hy1 : 1 ≤ y)
    (hy2 : y < 2 ^ L n) : RequirementHyp n y (hzFloor n y) := by
  have hD : 0 < hy n y * 2 ^ n := Nat.mul_pos (hy_pos hn hy1 hy2) (by positivity)
  have hT : 0 < (2 : ℕ) ^ (2 * L n) := by positivity
  have hprod : hy n y * 2 ^ n * hzFloor n y ≤ 2 ^ (2 * L n) - 1 := by
    simpa [hzFloor, Nat.mul_comm] using
      (Nat.div_mul_le_self (2 ^ (2 * L n) - 1) (hy n y * 2 ^ n))
  have hsplit := Nat.mod_add_div (2 ^ (2 * L n) - 1) (hy n y * 2 ^ n)
  have hrem := Nat.mod_lt (2 ^ (2 * L n) - 1) hD
  change (2 ^ (2 * L n) - 1) % (hy n y * 2 ^ n) +
    hy n y * 2 ^ n * hzFloor n y = 2 ^ (2 * L n) - 1 at hsplit
  constructor
  · omega
  · exact lt_of_le_of_lt hprod (Nat.sub_lt hT zero_lt_one)

lemma hzFloor_approx {n y : ℕ} (hn : 0 < n) (hy1 : 1 ≤ y)
    (hy2 : y < 2 ^ L n) : ApproxHyp n y (hzFloor n y) :=
  requirement hn hy1 hy2 (hzFloor_requirement hn hy1 hy2)

lemma hzFloor_bounds {n y : ℕ} (hn : 0 < n) (hy1 : 1 ≤ y)
    (hy2 : y < 2 ^ L n) : 2 ^ n ≤ hzFloor n y ∧ hzFloor n y < 2 ^ (L n + 1) := by
  have hD : 0 < hy n y * 2 ^ n := Nat.mul_pos (hy_pos hn hy1 hy2) (by positivity)
  have hDupper : hy n y * 2 ^ n < 2 ^ L n :=
    lt_of_le_of_lt hy_mul_pow_le_yy (yy_bounds hy1 hy2).2
  have hDlower : 2 ^ (L n - 1) ≤ hy n y * 2 ^ n := by
    calc
      2 ^ (L n - 1) = 2 ^ (n - 1) * 2 ^ n := by
        rw [← pow_add]
        congr 1
        unfold L
        omega
      _ ≤ hy n y * 2 ^ n := Nat.mul_le_mul_right _ (hy_lower hn hy1 hy2)
  constructor
  · unfold hzFloor
    apply (Nat.le_div_iff_mul_le hD).2
    have hlt : 2 ^ n * (hy n y * 2 ^ n) < 2 ^ (2 * L n) := by
      calc
        2 ^ n * (hy n y * 2 ^ n) < 2 ^ n * 2 ^ L n :=
          Nat.mul_lt_mul_of_pos_left hDupper (by positivity)
        _ = 2 ^ (n + L n) := by rw [pow_add]
        _ ≤ 2 ^ (2 * L n) := Nat.pow_le_pow_right (by positivity) (by unfold L; omega)
    omega
  · unfold hzFloor
    apply (Nat.div_lt_iff_lt_mul hD).2
    have htarget : 2 ^ (2 * L n) ≤ 2 ^ (L n + 1) * (hy n y * 2 ^ n) := by
      calc
        2 ^ (2 * L n) = 2 ^ (L n + 1) * 2 ^ (L n - 1) := by
          rw [← pow_add]
          congr 1
          have hL := L_pos hn
          omega
        _ ≤ 2 ^ (L n + 1) * (hy n y * 2 ^ n) := Nat.mul_le_mul_left _ hDlower
    have hT : 0 < (2 : ℕ) ^ (2 * L n) := by positivity
    exact lt_of_lt_of_le (Nat.sub_lt hT zero_lt_one) htarget

/-- Correctness for the concrete candidate, without an approximation premise. -/
theorem thm_lutnr_hzFloor {n y : ℕ} (hn : 0 < n) (hy1 : 1 ≤ y)
    (hy2 : y < 2 ^ L n) : GoodOutput n y (zOut n y (hzFloor n y)) := by
  have hbounds := hzFloor_bounds hn hy1 hy2
  exact thm_lutnr hn hy1 hy2 hbounds.1 hbounds.2 (hzFloor_approx hn hy1 hy2)

end LUTNR
