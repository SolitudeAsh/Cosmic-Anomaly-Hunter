import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = "/Users/ashwi/Downloads/MQAIS/Events/Cosmic hunter"

VOTES_FILE = os.path.join(
    PROJECT_ROOT,
    "Data",
    "zoo2MainSpecz.csv"
)

MAPPING_FILE = os.path.join(
    PROJECT_ROOT,
    "Data",
    "gz2_filename_mapping.csv"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "Data",
    "metadata.csv"
)

IMAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "Data",
    "images_gz2",
    "images"
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading Galaxy Zoo data...")

votes = pd.read_csv(VOTES_FILE)

mapping = pd.read_csv(MAPPING_FILE)

print("Votes:", len(votes))
print("Mapping:", len(mapping))


# ============================================================
# CREATE LABELS
# ============================================================

def get_label(row):

    # Spiral
    spiral = row[
        "t04_spiral_a08_spiral_fraction"
    ]

    # Elliptical / smooth
    smooth = row[
        "t01_smooth_or_features_a01_smooth_fraction"
    ]

    # Merger
    merger = row[
        "t08_odd_feature_a24_merger_fraction"
    ]

    # Require strong evidence
    if spiral > 0.5:
        return 1

    elif smooth > 0.5:
        return 0

    elif merger > 0.5:
        return 2

    else:
        return 2


votes["label"] = votes.apply(
    get_label,
    axis=1
)


# ============================================================
# MATCH DR7 OBJID → ASSET ID
# ============================================================

print("\nMatching DR7 object IDs...")

# Make sure both columns are strings
votes["dr7objid"] = (
    votes["dr7objid"]
    .astype(str)
)

mapping["objid"] = (
    mapping["objid"]
    .astype(str)
)

# Rename mapping column
mapping = mapping.rename(
    columns={
        "objid": "dr7objid"
    }
)

# Merge
metadata = votes[
    ["dr7objid", "label"]
].merge(
    mapping[
        ["dr7objid", "asset_id"]
    ],
    on="dr7objid",
    how="inner"
)


print(
    "Successfully matched:",
    len(metadata)
)


# ============================================================
# CREATE IMAGE FILENAMES
# ============================================================

metadata["asset_id"] = (
    metadata["asset_id"]
    .astype(int)
)

metadata["filename"] = (
    metadata["asset_id"]
    .astype(str)
    + ".jpg"
)


# ============================================================
# CHECK IMAGE EXISTENCE
# ============================================================

print("\nChecking image files...")

metadata["image_exists"] = metadata[
    "filename"
].apply(
    lambda filename:
        os.path.exists(
            os.path.join(
                IMAGE_DIR,
                filename
            )
        )
)


existing = metadata[
    metadata["image_exists"]
].copy()

missing = metadata[
    ~metadata["image_exists"]
].copy()


# ============================================================
# REPORT
# ============================================================

print("\n==============================")
print("DATASET REPORT")
print("==============================")

print(
    "Total metadata entries:",
    len(metadata)
)

print(
    "Images found:",
    len(existing)
)

print(
    "Images missing:",
    len(missing)
)

print(
    "Percentage available:",
    f"{100 * len(existing) / len(metadata):.2f}%"
)


if len(missing) > 0:

    print("\nFirst 20 missing images:")

    print(
        missing[
            ["dr7objid", "asset_id", "filename"]
        ].head(20).to_string(
            index=False
        )
    )


# ============================================================
# REMOVE CHECK COLUMN
# ============================================================

existing = existing[
    [
        "filename",
        "dr7objid",
        "asset_id",
        "label"
    ]
]


# ============================================================
# SAVE
# ============================================================

existing.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f"\nSaved metadata to:\n{OUTPUT_FILE}"
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nClass distribution:")

print(
    existing["label"]
    .value_counts()
    .sort_index()
)


print("\nLabels:")

print("0 = Elliptical / Smooth")
print("1 = Spiral")
print("2 = Merger / Other")