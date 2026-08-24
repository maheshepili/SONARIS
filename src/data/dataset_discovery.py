from collections import Counter
from pathlib import Path


DATASET_PATH = Path("data/raw/SubPipeMiniSSS")

print("=" * 50)
print("SUBPIPE DATASET DISCOVERY")
print("=" * 50)

if not DATASET_PATH.exists():
    print("Dataset not found!")
    print(DATASET_PATH)
    raise SystemExit(1)

# 1. Count all files and directories.
all_files = [path for path in DATASET_PATH.rglob("*") if path.is_file()]
all_dirs = [path for path in DATASET_PATH.rglob("*") if path.is_dir()]

print(f"\nTotal files: {len(all_files)}")
print(f"Total directories: {len(all_dirs)}")

# 2. File extensions.
extensions = Counter(
    path.suffix.lower() if path.suffix else "[no extension]"
    for path in all_files
)

print("\nFiles by extension:")
for extension, count in sorted(extensions.items()):
    print(f"  {extension}: {count}")

# 3. Inspect the image folders.
image_folders = [
    "Cam0_images",
    "Cam1_images",
    "SSS_HF_images",
    "SSS_LF_images",
]

print("\nImage directories:")
for folder_name in image_folders:
    folder = DATASET_PATH / "DATA" / folder_name

    if not folder.exists():
        print(f"\n{folder_name}: NOT FOUND")
        continue

    images = [path for path in folder.iterdir() if path.is_file()]
    image_extensions = Counter(
        path.suffix.lower() if path.suffix else "[no extension]"
        for path in images
    )

    print(f"\n{folder_name}")
    print(f"  Number of files: {len(images)}")
    print("  Extensions:")
    for extension, count in sorted(image_extensions.items()):
        print(f"    {extension}: {count}")

    print("  First 5 files:")
    for image in sorted(images)[:5]:
        print(f"    {image.name}")

    print("  Last 5 files:")
    for image in sorted(images)[-5:]:
        print(f"    {image.name}")

# 4. CSV files.
csv_files = list(DATASET_PATH.rglob("*.csv"))

print("\nCSV files:")
print(f"Total CSV files: {len(csv_files)}")
for csv_file in sorted(csv_files):
    size_mb = csv_file.stat().st_size / (1024 * 1024)
    print(f"  {csv_file.name} ({size_mb:.2f} MB)")

print("\n" + "=" * 50)
print("DISCOVERY COMPLETE")
print("=" * 50)
