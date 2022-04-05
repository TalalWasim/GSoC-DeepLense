#!/usr/bin/env bash

echo "Downloading Task5 Dataset..."

gdown --id 1K3Cqv9thq6tocLa8doejSrmMUeqLFWfE

echo "Unzipping Task5 Dataset..."

tar zxvf lenses.tgz >/dev/null

echo "Downloading Task5 Checkpoints..."

gdown --id 1ZfQD_h-3BD2b6Pqp-mIPnariAxgnue-z

echo "Unzipping Task5 Checkpoints..."

unzip checkpoints_task5.zip >/dev/null

echo "Deleting Archives..."

rm lenses.tgz checkpoints_task5.zip

echo "Done..."