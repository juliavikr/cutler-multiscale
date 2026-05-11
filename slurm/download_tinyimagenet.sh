#!/bin/bash

#SBATCH --job-name=cutler-download-tinyimagenet
# TODO: set your SLURM account â€” export SBATCH_ACCOUNT=<your_number>
#       or pass --account=<your_number> to sbatch at submission time.
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/download_tinyimagenet_%j.out
#SBATCH --error=logs/download_tinyimagenet_%j.err

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-${HOME}/data}"
ARCHIVE_DIR="${DATA_ROOT}/tiny-imagenet"
DATASET_DIR="${DATA_ROOT}/tiny-imagenet-200"
mkdir -p "${ARCHIVE_DIR}"

echo "=== Downloading Tiny ImageNet 200 ==="
if [ ! -d "${DATASET_DIR}/train" ]; then
    wget -c --progress=bar:force \
        http://cs231n.stanford.edu/tiny-imagenet-200.zip \
        -O "${ARCHIVE_DIR}/tiny-imagenet-200.zip"

    echo "=== Unzipping ==="
    unzip -q "${ARCHIVE_DIR}/tiny-imagenet-200.zip" -d "${DATA_ROOT}/"

    echo "=== Cleaning up zip ==="
    rm "${ARCHIVE_DIR}/tiny-imagenet-200.zip"
else
    echo "Tiny ImageNet already exists at ${DATASET_DIR}; skipping download."
fi

make_subset() {
    local subset_dir="$1"
    shift
    local classes=("$@")

    mkdir -p "${subset_dir}/train" "${subset_dir}/train_flat"
    for class_id in "${classes[@]}"; do
        mkdir -p "${subset_dir}/train" "${subset_dir}/train_flat/${class_id}"
        cp -R "${DATASET_DIR}/train/${class_id}" "${subset_dir}/train/"
        cp "${DATASET_DIR}/train/${class_id}/images/"* "${subset_dir}/train_flat/${class_id}/"
    done
}

CLASSES_10=(
    n01443537
    n02123045
    n02281406
    n02410509
    n02906734
    n03100240
    n03444034
    n04067472
    n04254777
    n07711569
)

CLASSES_5=(
    n01443537
    n02123045
    n02281406
    n02410509
    n02906734
)

echo "=== Creating 10-class subset ==="
make_subset "${DATA_ROOT}/tiny-imagenet-10classes" "${CLASSES_10[@]}"

echo "=== Creating 5-class subset ==="
make_subset "${DATA_ROOT}/tiny-imagenet-5" "${CLASSES_5[@]}"

echo "=== Train folder structure (first 10 classes) ==="
ls "${DATASET_DIR}/train/" | head -10

echo "=== Class folder count ==="
N=$(ls "${DATASET_DIR}/train/" | wc -l)
echo "Found ${N} class folders in train/ (expected 200)"

echo ""
echo "Done."
echo "Tiny ImageNet full: ${DATASET_DIR}"
echo "10-class subset:    ${DATA_ROOT}/tiny-imagenet-10classes"
echo "5-class subset:     ${DATA_ROOT}/tiny-imagenet-5"
