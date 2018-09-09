# PongAI


A reinforcement learning approach to training an AI in Pong to consistently defeat the in game opponent.

To begin training, you will need to make sure the following packages and tools are installed:
* Python 2.7
* Tensorflow 1.4.1
* Pong-V0 via OpenAI-Gym


The following settings can be adjusted in the pong.sh file found in the scripts folder:
* reset_output_dir (boolean): True clears the current output directory 
* output_dir (String): location of the output directory
* log_every (int): Number of episodes between displaying the current loss and score of the model
* update_every (int): How many episodes to run before updating the loss and baseline
* n_eps (int): Number of episodes to train for
* bl_dec (float): Decay factor fo the baseline
* discount (float): Decay factor for the rewards
* restore (bool): True indicates restoring the model from a checkpoint, False otherwise
* render (bool): Whether or not to render the game during each episode
* render_every (int): How often to render the game played by the model

The model can be started by running scripts/pong.sh from the root directory. A trained checkpoint file is included in the outputs folder in which the AI has been trained for 12000 episodes.
