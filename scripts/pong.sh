#!/bin/bash

export PYTHONPATH="$(pwd)"

python src/pong.py \
  --reset_output_dir=False \
  --output_dir="outputs" \
  --log_every=1 \
  --update_every=1 \
  --n_eps=2000 \
  --bl_dec=0.99 \
  --discount=0.99\
  --restore=True\
  --render=True\
  --render_every=1\
  --render_speed=6\
  "$@"

