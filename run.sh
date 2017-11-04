trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
python3 examples/inverted_pendulum.py &
rm -rf ./train
tensorboard --logdir=./train
