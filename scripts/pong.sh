#!/bin/bash

export PYTHONPATH="$(pwd)"

python src/pong.py \
  --reset_output_dir=False \
  --output_dir="outputs" \
  --log_every=10 \
  --reset_every=1 \
  --update_every=1 \
  --n_eps=2000 \
  --bl_dec=0.99 \
  --num_samples=1\
  --discount=0.99\
  --restore=True\
  --render=True\
  "$@"

