from pathlib import Path

root = Path(".")
max_depth = 3

print(f"Root: {root.resolve()}\n")

for path in sorted(root.rglob("*")):
    depth = len(path.relative_to(root).parts)

    if depth <= max_depth:
        kind = "DIR " if path.is_dir() else "FILE"

        if path.is_file():
            size = path.stat().st_size / (1024 ** 2)
        else:
            size = 0

        print(f"{kind} | {size:8.2f} MB | {path}")