import os
import argparse
from typing import Optional, Tuple


def read_loss(file_path: str) -> Optional[float]:
    """Read the first float-like token from the file, return None if not found."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                token = line.strip().split()[0] if line.strip() else ""
                try:
                    return float(token)
                except Exception:
                    continue
    except Exception:
        return None
    return None


def find_best(folder: str, filename: str = "best_val_loss", recursive: bool = False) -> Optional[Tuple[str, float]]:
    """Return (subfolder_path, loss) with the lowest loss found, or None if none found."""
    best_path = None
    best_loss = None

    if recursive:
        for root, dirs, _ in os.walk(folder):
            for d in dirs:
                candidate = os.path.join(root, d, filename)
                loss = read_loss(candidate)
                if loss is not None and (best_loss is None or loss < best_loss):
                    best_loss = loss
                    best_path = os.path.join(root, d)
    else:
        for name in os.listdir(folder):
            sub = os.path.join(folder, name)
            if not os.path.isdir(sub):
                continue
            candidate = os.path.join(sub, filename)
            loss = read_loss(candidate)
            if loss is not None and (best_loss is None or loss < best_loss):
                best_loss = loss
                best_path = sub

    if best_path is None:
        return None
    return best_path, best_loss


def main():
    p = argparse.ArgumentParser(description="Find subfolder with lowest validation loss file.")
    p.add_argument("folder", nargs="?", default=".", help="Parent folder to scan")
    p.add_argument("--filename", default="best_val_loss.txt", help="Name of the loss file (default: best_val_loss)")
    p.add_argument("--recursive", action="store_true", help="Scan subfolders recursively")
    args = p.parse_args()

    res = find_best(args.folder, filename=args.filename, recursive=args.recursive)
    if res is None:
        print("No valid loss files found.")
    else:
        path, loss = res
        print(f"Best folder: {path}")
        print(f"Validation loss: {loss}")


if __name__ == "__main__":
    main()