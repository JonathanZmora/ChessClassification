import argparse
import csv
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


OUT_DIR_RE = re.compile(r'^\s*OUT_DIR\s*=\s*"([^"]+)"\s*$', re.MULTILINE)


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def extract_out_dir(blender_script_path: Path) -> Path | None:
    """
    chess_position_api_v4.py has a hardcoded:
      OUT_DIR = "C:\\...\\renders"
    We parse it so we know where Blender actually writes.
    """
    text = blender_script_path.read_text(encoding="utf-8", errors="ignore")
    m = OUT_DIR_RE.search(text)
    if not m:
        return None
    return Path(m.group(1))


def safe_clean_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    for p in d.glob("*"):
        if p.is_file():
            p.unlink()


def read_processed_frames(out_csv: Path) -> set[int]:
    if not out_csv.exists():
        return set()

    processed = set()
    with out_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                processed.add(int(row["frame_number"]))
            except Exception:
                continue
    return processed


def iter_frames_from_csv(csv_path: Path):
    """
    Supports either:
      - from_frame,to_frame,fen   (your uploaded all_games.csv style)
      - frame_number,fen         (single-frame rows)
    """
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = {c.strip().lower() for c in reader.fieldnames or []}

        fen_col = "fen" if "fen" in cols else ("fEN" if "fEN" in cols else None)
        if fen_col is None:
            # try case-insensitive lookup
            for c in reader.fieldnames or []:
                if c.strip().lower() == "fen":
                    fen_col = c
                    break
        if fen_col is None:
            raise ValueError("CSV must contain a 'fen' column.")

        has_range = ("from_frame" in cols and "to_frame" in cols)
        has_single = ("frame_number" in cols or "frame" in cols)

        if not has_range and not has_single:
            raise ValueError("CSV must contain either (from_frame,to_frame) or (frame_number/frame).")

        for row in reader:
            game_num = row["game"].strip()
            fen = row[fen_col].strip()

            if has_range:
                start = int(row["from_frame"])
                end = int(row["to_frame"])
                for frame_num in range(start, end + 1):
                    yield game_num, frame_num, fen
            else:
                key = "frame_number" if "frame_number" in cols else "frame"
                frame_num = int(row[key])
                yield game_num, frame_num, fen


def run_blender(blender_exe: Path, blend_file: Path, blender_script: Path, fen: str, view: str, resolution: int, cwd: Path):
    cmd = [
        str(blender_exe),
        str(blend_file),
        "--background",
        "--python",
        str(blender_script),
        "--",
        "--fen",
        fen,
        "--view",
        view,
        "--resolution",
        str(resolution),
    ]
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_perspective(python_exe: Path, perspective_script: Path, cwd: Path):
    subprocess.run([str(python_exe), str(perspective_script)], cwd=str(cwd), check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV path (e.g. all_games.csv)")
    parser.add_argument("--out", default="synthetic_index.csv", help="Output index CSV name")
    parser.add_argument("--view", default="black")
    parser.add_argument("--resolution", type=int, default=1600)
    parser.add_argument("--skip-existing", action="store_true", help="Skip frame if already in output CSV")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent

    blender_exe = Path(r"/home/zmoraj/blender-5.0.1-linux-x64/blender")
    blend_file = project_dir / "chess-set.blend"
    blender_script = project_dir / "chess_position_api_v4.py"
    perspective_script = project_dir / "perspective_transform.py"

    input_csv = (project_dir / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    out_csv = (project_dir / args.out).resolve()
    ready_dir = project_dir / "synthetic_data_ready"
    ready_dir.mkdir(parents=True, exist_ok=True)

    if not blender_exe.exists():
        raise FileNotFoundError(f"Blender exe not found: {blender_exe}")
    if not blend_file.exists():
        raise FileNotFoundError(f"Missing blend file: {blend_file}")
    if not blender_script.exists():
        raise FileNotFoundError(f"Missing blender script: {blender_script}")
    if not perspective_script.exists():
        raise FileNotFoundError(f"Missing perspective script: {perspective_script}")
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")

    # Detect where Blender outputs renders
    renders_dir = extract_out_dir(blender_script) or (project_dir / "renders")
    renders_parent = renders_dir.parent  # perspective_transform uses "./renders", so cwd must be parent
    if renders_dir.name.lower() != "renders":
        # If OUT_DIR is something else, perspective_transform won't find it unless it is literally named "renders".
        # Most people have OUT_DIR ending with "...\\renders". This is just a safety note.
        print(f"[WARN] OUT_DIR in blender script is '{renders_dir}'. perspective_transform expects a folder named 'renders'.")
    renders_dir.mkdir(parents=True, exist_ok=True)

    processed = read_processed_frames(out_csv) if args.skip_existing else set()

    # open output CSV
    new_file = not out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        if new_file:
            writer.writerow([
                "game_number",
                "frame_number",
                "view",
                "original_name",
                "warped_overhead_name",
                "warped_east_name",
                "warped_west_name",
                "FEN",
            ])

        total = 0
        for game_num, frame_num, fen in iter_frames_from_csv(input_csv):
            total += 1

            if args.skip_existing and frame_num in processed:
                print(f"[{now()}] [SKIP] frame {frame_num} already in {out_csv.name}")
                continue

            print(f"[{now()}] [RUN] frame {frame_num}  fen={fen}")

            # Clean renders to avoid old json/png being reprocessed
            safe_clean_dir(renders_dir)

            try:
                # 1) Blender render
                run_blender(
                    blender_exe=blender_exe,
                    blend_file=blend_file,
                    blender_script=blender_script,
                    fen=fen,
                    view=args.view,
                    resolution=args.resolution,
                    cwd=project_dir,
                )

                # 2) Perspective transform (expects ./renders)
                run_perspective(
                    python_exe=Path(sys.executable),
                    perspective_script=perspective_script,
                    cwd=renders_parent,
                )

                # 3) Rename warped files + move only warped files
                src_overhead = renders_dir / "1_overhead_warped.png"
                
                if args.view == 'white':
                    # In white view, Blender outputs: 2_east, 3_west
                    src_east = renders_dir / "2_east_warped.png"
                    src_west = renders_dir / "3_west_warped.png"
                else:
                    # In black view, Blender outputs: 2_west, 3_east
                    src_west = renders_dir / "2_west_warped.png"
                    src_east = renders_dir / "3_east_warped.png"

                if not src_overhead.exists() or not src_west.exists() or not src_east.exists():
                    raise FileNotFoundError("One or more warped outputs are missing in renders/.")

                dst_overhead = ready_dir / f"game{game_num}_{frame_num}_{args.view[0]}_overhead_warped.png"
                dst_west = ready_dir / f"game{game_num}_{frame_num}_{args.view[0]}_west_warped.png"
                dst_east = ready_dir / f"game{game_num}_{frame_num}_{args.view[0]}_east_warped.png"

                shutil.move(str(src_overhead), str(dst_overhead))
                shutil.move(str(src_west), str(dst_west))
                shutil.move(str(src_east), str(dst_east))

                # 4) Write output row
                original_name = f"frame_{frame_num:06d}.jpg"
                writer.writerow([
                    game_num,
                    frame_num,
                    args.view,
                    original_name,
                    dst_overhead.name,
                    dst_east.name,
                    dst_west.name,
                    fen,
                ])
                f_out.flush()

                if args.skip_existing:
                    processed.add(frame_num)

                print(f"[{now()}] [OK] moved -> {dst_overhead.name}, {dst_east.name}, {dst_west.name}")

            except subprocess.CalledProcessError as e:
                print(f"[{now()}] [ERROR] subprocess failed for frame {frame_num}: {e}")
                continue
            except Exception as e:
                print(f"[{now()}] [ERROR] frame {frame_num}: {e}")
                continue

    print(f"[{now()}] Done. Output CSV: {out_csv}")
    print(f"[{now()}] Ready data dir: {ready_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
