from pathlib import Path

DATASET_PATH = Path("data/raw/SubPipeMiniSSS")

pbm_files = sorted(DATASET_PATH.rglob("*.pbm"))

print("=" * 60)
print("PBM INSPECTION")
print("=" * 60)

print(f"Total PBM files: {len(pbm_files)}")

if not pbm_files:
    raise SystemExit("No PBM files found.")

sample = pbm_files[0]

print(f"\nSample file:")
print(sample)

print(f"\nFile size:")
print(f"{sample.stat().st_size / (1024 * 1024):.2f} MB")

with open(sample, "rb") as file:
    header = file.read(100)

print("\nFirst 100 bytes:")
print(header)

print("\n" + "=" * 60)