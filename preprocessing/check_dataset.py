from pathlib import Path
from collections import Counter

DATASET_DIR = Path("data/raw/Autism emotion recogition dataset")

TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(directory):
    counts = Counter()

    if not directory.exists():
        print(f"ERROR: Directory not found: {directory}")
        return counts

    for class_dir in sorted(directory.iterdir()):

        if not class_dir.is_dir():
            continue

        count = sum(
            1
            for file in class_dir.iterdir()
            if file.is_file()
            and file.suffix.lower() in VALID_EXTENSIONS
        )

        counts[class_dir.name] = count

    return counts


print("=" * 60)
print("AUTISM EMOTION DATASET VALIDATION")
print("=" * 60)

print(f"\nDataset path:")
print(DATASET_DIR.resolve())

print("\nTRAINING DATA")
print("-" * 60)

train_counts = count_images(TRAIN_DIR)

for emotion, count in train_counts.items():
    print(f"{emotion:15} : {count}")

print(f"\nTotal training images: {sum(train_counts.values())}")


print("\nTEST DATA")
print("-" * 60)

test_counts = count_images(TEST_DIR)

for emotion, count in test_counts.items():
    print(f"{emotion:15} : {count}")

print(f"\nTotal test images: {sum(test_counts.values())}")


print("\nCLASS CHECK")
print("-" * 60)

train_classes = set(train_counts.keys())
test_classes = set(test_counts.keys())

print("Training classes:", sorted(train_classes))
print("Testing classes :", sorted(test_classes))

if train_classes == test_classes:
    print("\n✓ Training and testing classes match.")
else:
    print("\nWARNING: Training and testing classes do not match.")


print("\nDATASET SUMMARY")
print("-" * 60)

print(f"Training images : {sum(train_counts.values())}")
print(f"Testing images  : {sum(test_counts.values())}")
print(f"Total images    : {sum(train_counts.values()) + sum(test_counts.values())}")
print(f"Number of classes: {len(train_classes)}")

print("\nValidation complete.")