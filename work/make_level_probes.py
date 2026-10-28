"""Option 2 level probes on top of P2-S13 (teammate's best, 0.15577).

Probe B: buildings new-business rows x0.85 (renewals unchanged). Evidence: NB actual/expected vs renewals 0.76 (grouped)
and 0.72 (time fold), and 2025 weather about -6% for buildings with the developed market's per-peril weather effects.
Probe C: contents new-business rows x1.15. Evidence: NB vs renewals 1.24 (grouped) and 1.49 (time fold); weather flat.
Each probe changes one line only, so their public scores can be read separately.
"""
import io
import zipfile

import pandas as pd

D = r"D:/Part 2/aiiaa/data/iaa-ai-assisted-hackathon-2026-part-2/emerging/"
W = r"D:/Part 2/aiiaa/work/"
with zipfile.ZipFile(W + "P2-S13_p2_c047.zip") as z:
    base = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
info = pd.concat([pd.read_parquet(D + f"test_{l}.parquet", columns=["row_id", "new_business"]).assign(line=l)
                  for l in ["motor", "buildings", "contents"]])
base = base.merge(info, on="row_id", how="left", validate="one_to_one")
assert base.line.notna().all() and len(base) == 209063
print("P2-S13 mean", round(base.prediction.mean(), 2), base.groupby(["line", "new_business"]).prediction.mean().round(1).to_dict())

for name, line, factor in [("p2_s13_bnb085", "buildings", 0.85), ("p2_s13_cnb115", "contents", 1.15)]:
    s = base.copy()
    sel = (s.line == line) & s.new_business.astype(bool)
    s.loc[sel, "prediction"] *= factor
    out = s[["row_id", "prediction"]]
    out.to_csv(W + name + ".csv", index=False)
    with zipfile.ZipFile(W + name + ".zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(name + ".csv", out.to_csv(index=False))
    print(f"{name}: {sel.sum():,} rows x{factor}, mean {out.prediction.mean():.2f}, {line} total x{s[s.line==line].prediction.sum()/base[base.line==line].prediction.sum():.3f}")
