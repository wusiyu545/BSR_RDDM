from pathlib import Path

ROOT = Path("./data/DIV2K_x4_min")
VAL_INPUT_DIR = ROOT / "val_LQ_x4_blind"
OUT_FLIST = ROOT / "test_input.flist"

files = sorted(list(VAL_INPUT_DIR.glob("*.png")) + list(VAL_INPUT_DIR.glob("*.jpg")) + list(VAL_INPUT_DIR.glob("*.jpeg")))

with open(OUT_FLIST, "w", encoding="utf-8") as f:
    for p in files:
        f.write(p.as_posix() + "\n")

print(f"[DONE] wrote {len(files)} lines to {OUT_FLIST}")