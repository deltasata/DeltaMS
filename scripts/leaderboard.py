"""Summarise the public MSSEG-2 test-set results (Zenodo 10.5281/zenodo.5775523 v3, CC-BY-4.0).

Each team file has a table for the 32 patients with new lesions, then a table for the 28 without.
Output: results/leaderboard.csv with mean F1 / Dice (lesion cases) and mean false lesions (empty cases).
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reference/zenodo_5775523"


def read_team(path: Path):
    raw = pd.read_excel(path, header=None)
    hdr = [i for i, v in raw[0].items() if str(v).strip().lower().startswith("patient")]
    if len(hdr) < 2:
        raise ValueError(f"{path.name}: found {len(hdr)} header rows")

    def block(start, stop):
        t = raw.iloc[start + 1:stop].copy()
        t.columns = [str(c).strip() for c in raw.iloc[start]]
        t = t.dropna(how="all")
        t = t[pd.to_numeric(t.iloc[:, 0], errors="coerce").notna()]
        return t.apply(pd.to_numeric, errors="coerce")

    return block(hdr[0], hdr[1]), block(hdr[1], len(raw))


def main():
    rows = []
    for f in sorted(SRC.glob("*.xlsx")):
        if f.name.startswith("MS_challenge"):
            continue
        les, emp = read_team(f)
        rows.append(dict(
            team=f.stem, n_les=len(les), n_emp=len(emp),
            F1=les["F1_score"].mean(), F1_sd=les["F1_score"].std(),
            Dice=les["Dice"].mean(), Dice_sd=les["Dice"].std(),
            SensL=les["SensL"].mean(), PPVL=les["PPVL"].mean(),
            FP_empty=emp["NbTestedLesions"].mean(),
            frac_empty_with_FP=(emp["NbTestedLesions"] > 0).mean(),
        ))
    df = pd.DataFrame(rows).sort_values("F1", ascending=False)
    out = ROOT / "results/leaderboard.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
