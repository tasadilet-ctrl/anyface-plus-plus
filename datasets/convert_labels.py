"""Convert AnyFace++ labels to YOLO pose format with valid placeholder landmarks."""
import os
from pathlib import Path
import shutil

def convert_label(input_file, output_file):
    """Convert a single label file."""
    with open(input_file) as f:
        content = f.read().strip()
    if not content:
        return False

    parts = content.split()
    if len(parts) < 17:
        return False

    face_type = int(parts[0])
    xc = float(parts[1])
    yc = float(parts[2])
    w = float(parts[3])
    h = float(parts[4])

    # Read landmarks from input (5 points)
    landmarks = []
    for i in range(5):
        x = float(parts[5 + i*2])
        y = float(parts[6 + i*2])
        if x > 0 and y > 0:
            landmarks.append(f"{x:.6f} {y:.6f} 1.0")
        else:
            # Use face center as placeholder (YOLO rejects negative coords)
            landmarks.append(f"{xc:.6f} {yc:.6f} 1.0")

    out = f"{face_type} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} " + " ".join(landmarks)

    with open(output_file, "w") as f:
        f.write(out + "\n")
    return True

base = Path("/data/home/<user>/srp/anyface-orig/datasets/dataset/RAFdbex")
for split in ["train", "val"]:
    label_dir = base / split / "labels"
    out_dir = base / split / "labels_tmp"
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for txt in label_dir.glob("*.txt"):
        if convert_label(txt, out_dir / txt.name):
            count += 1

    # Replace
    for txt in label_dir.glob("*.txt"):
        txt.unlink()
    for txt in out_dir.glob("*.txt"):
        shutil.move(txt, label_dir / txt.name)
    out_dir.rmdir()

    print(f"{split}: converted {count} labels")

# Cleanup cache
for f in Path("/data/home/<user>/srp/anyface-orig/datasets/dataset/RAFdbex").glob("**/*.cache"):
    f.unlink()

print("Done!")
