from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import cPickle as pickle
import shutil
from skimage.color import rgb2gray


import gym
import numpy as np
import tensorflow as tf

from src.utils import *


DEFINE_boolean("reset_output_dir", False, "Delete output_dir if exists.")
DEFINE_boolean("restore", False, "Restore from checkpoint")
DEFINE_boolean("render", False, "Show Game")
DEFINE_integer("render_every", 1, "How often to render the game")
DEFINE_integer("render_speed", 1, "How fast to render the game")
DEFINE_string("data_path", "", "")
DEFINE_string("output_dir", "", "")
DEFINE_integer("num_samples", 1, "How many screens to group")

DEFINE_integer("log_every", 1, "How many episodes to log")
DEFINE_integer("reset_every", 10, "How many episodes to reset")
DEFINE_integer("update_every", 1, "How many episodes to update params")

DEFINE_integer("n_eps", 100, "Number of training episodes")
DEFINE_integer("n_hidden", 20, "Hidden dimension")
DEFINE_float("lr", 1e-5, "Learning rate")
DEFINE_float("discount", 0.99, "Discount factor")
DEFINE_float("grad_bound", 5.0, "Gradient clipping threshold")
DEFINE_float("bl_dec", 0.99, "Baseline moving average")
DEFINE_float("beta", 1e-4, "Regularization Factor")


flags = tf.app.flags
FLAGS = flags.FLAGS

### Preprocess and hidden units size sourced from github: https://github.com/mrahtz/tensorflow-rl-pong/blob/master/pong.py
def build_tf_graph(hparams):
  print("-" * 80)
  print("Building TF graph")

  states = tf.placeholder(tf.float32, [None, 6400], name="states")
  sampled_actions = tf.placeholder(tf.float32, [None, 1], name="sampled_actions")
  rewards = tf.placeholder(tf.float32, [None, 1], name="rewards")

  feed = tf.layers.dense( inputs = states, 
                          units = 256,
                          activation = tf.nn.relu,
                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                          name = 'feed')
  logits = tf.layers.dense( inputs=feed,
                            units=1,
                            activation=tf.sigmoid,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            name = 'softmax')
  loss = tf.losses.log_loss(labels=sampled_actions,
                            predictions=logits,
                            weights=rewards)

  optimizer = tf.train.AdamOptimizer(hparams.lr)
  global_step = tf.train.get_or_create_global_step()
  train_op = optimizer.minimize(loss, global_step=global_step)

  # print(count_model_params(tf.trainable_variables()))

  ops = {
    "states": states,
    "rewards": rewards,
    "sampled_actions": sampled_actions,
    "logits": logits,
    "loss": loss,
    "train_op": train_op,
  }
  return ops

def preProcess(image):
  image = image[35:195]
  image = image[::2, ::2, 0]
  image[image == 144] = 0
  image[image == 109] = 0
  image[image != 0] = 1
  return image.astype(np.float).ravel()

def getDiscountedRewards(discount, rewards):
  scorepoints = np.nonzero(rewards)[0]
  discrewards = np.zeros_like(rewards)
  prev = 0
  for k in scorepoints:
    discrewards[prev:k+1] = rewards[k]*np.power(discount, np.arange(k-prev+1)[::-1])
    prev = k+1
  discrewards = (discrewards-np.mean(discrewards))/np.std(discrewards)
  return discrewards

def train(hparams):
  print("-" * 80)
  print("Building OpenAI gym environment.")
  env = gym.make("Pong-v0")
  env.reset()

  n_actions = env.action_space.n
  inp_shape = list(env.observation_space.shape)
  print("Game has {0} actions".format(n_actions))
  print("Input shape: {0}".format(inp_shape))

  hparams.add_hparam("inp_shape", inp_shape)
  hparams.add_hparam("n_actions", n_actions)

  g = tf.Graph()
  with g.as_default():
    ops = build_tf_graph(hparams)

    # TF session
    saver = tf.train.Saver(max_to_keep=10)
    chkpt = tf.train.get_checkpoint_state(hparams.output_dir)

    checkpoint_saver_hook = tf.train.CheckpointSaverHook(
      hparams.output_dir, save_steps=500, saver=saver)
    hooks = [checkpoint_saver_hook]
    sess = tf.train.SingularMonitoredSession(
      hooks=hooks, checkpoint_dir=hparams.output_dir)
    if chkpt != None and hparams.restore:
      saver.restore(sess, chkpt.model_checkpoint_path)

    actionmap = {0:3, 1:2}

    # RL
    baseline = 0
    for eps in range(1, 1 + hparams.n_eps):
      states, actions, rewards = [], [], []
      done = False

      prevstate = env.reset()
      prevstate = preProcess(prevstate)
      curstate, _, _, _ = env.step(0)
      curstate = preProcess(curstate)

      while not done:
        if hparams.render and eps%hparams.render_every == 0:
          env.render()  
          time.sleep(0.1/hparams.render_speed)

        state = curstate-prevstate
        prevstate = curstate

        state = state.reshape([1, -1])
        logits = sess.run(ops["logits"], feed_dict={ops["states"]: state})[0][0]

        if np.random.uniform() < logits:
          action = 1
        else:
          action = 0

        curstate, reward, done, _ = env.step(actionmap[action])
        curstate = preProcess(curstate)

        states.append(state)
        actions.append(action)
        rewards.append(reward)

      rewards = np.array(rewards, dtype=np.float32)
      actions = np.array(actions, dtype=np.float32)

      if eps % hparams.update_every == 0:
        discrewards = getDiscountedRewards(hparams.discount, rewards)
        baseline = 0.99 * baseline + 0.01 * np.sum(rewards)

        run_ops = [
          ops["loss"],
          ops["train_op"],
        ]
        feed_dict = {
          ops["states"]: np.vstack(states),
          ops["rewards"]: discrewards.reshape((-1,1)),
          ops["sampled_actions"]: actions.reshape((-1,1)),
        }
        (loss, _) = sess.run(run_ops, feed_dict=feed_dict)


      if eps % hparams.log_every == 0:
        log_string = "eps={0:<4d}".format(eps)
        log_string += " loss={0:<g}".format(loss)
        log_string += " len={0:<g}".format(len(rewards))
        log_string += " modelscore={0:<g}".format(np.sum(rewards>0))
        print(log_string)

    sess.close()

def main(_):
  print("-" * 80)
  if not os.path.isdir(FLAGS.output_dir):
    print("Path {} does not exist. Creating.".format(FLAGS.output_dir))
    os.makedirs(FLAGS.output_dir)
  elif FLAGS.reset_output_dir and not FLAGS.restore:
    print("Path {} exists. Remove and remake.".format(FLAGS.output_dir))
    shutil.rmtree(FLAGS.output_dir)
    os.makedirs(FLAGS.output_dir)

  print("-" * 80)
  log_file = os.path.join(FLAGS.output_dir, "stdout")
  print("Logging to {}".format(log_file))
  sys.stdout = Logger(log_file)

  print_user_flags()

  hparams = tf.contrib.training.HParams(
    n_eps=FLAGS.n_eps,
    n_hidden=FLAGS.n_hidden,
    lr=FLAGS.lr,
    grad_bound=FLAGS.grad_bound,
    bl_dec=FLAGS.bl_dec,
    discount=FLAGS.discount,
    log_every=FLAGS.log_every,
    reset_every=FLAGS.reset_every,
    num_samples=FLAGS.num_samples,
    update_every=FLAGS.update_every,
    output_dir=FLAGS.output_dir,
    restore=FLAGS.restore,
    beta=FLAGS.beta,
    render=FLAGS.render,
    render_every=FLAGS.render_every,
    render_speed=FLAGS.render_speed
  )
  train(hparams)


if __name__ == "__main__":
  tf.app.run()