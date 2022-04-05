#!/usr/bin/env bash

echo "Downloading Task1 Dataset..."

gdown --id 1cqUQyALFSzSNAwx1NMgc0bInmbXObLLA

echo "Unzipping Task1 Dataset..."

unzip dataset_task1.zip >/dev/null

echo "Downloading Task1 Checkpoints..."

gdown --id 11rX4a6vkD76mYViT5NAdzVE7KG9QuoZ7

echo "Unzipping Task1 Checkpoints..."

unzip checkpoints_task1.zip >/dev/null

echo "Deleting Archives..."

rm dataset_task1.zip checkpoints_task1.zip

echo "Done..."