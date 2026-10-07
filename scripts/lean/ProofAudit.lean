import ReducedNR
import LUTNR
import Lean.Util.CollectAxioms
import Lean.Elab.Command

/- Check every bundled declaration, including transitive dependencies. -/
open Lean in
run_cmd do
  let env ← getEnv
  let moduleNames := env.header.moduleNames
  let allowed := #[`propext, `Classical.choice, `Quot.sound]
  let mut checked : Nat := 0
  for (name, _) in env.constants.toList do
    let some idx := env.getModuleIdxFor? name | continue
    let moduleName := moduleNames[idx.toNat]!
    unless moduleName == `ReducedNR || moduleName == `LUTNR do
      continue
    for axiomName in ← collectAxioms name do
      unless allowed.contains axiomName do
        throwError "{name} depends on disallowed axiom {axiomName}"
    checked := checked + 1
  if checked == 0 then
    throwError "No bundled declarations were audited"
  logInfo m!"Audited {checked} declarations: only standard logical axioms are used."

#print axioms Reciprocal.thm_all_k_complete
#print axioms LUTNR.thm_lutnr
#print axioms LUTNR.thm_lutnr_hzFloor
