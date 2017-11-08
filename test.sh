trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
#python3 examples/inverted_pendulum.py &
python3 test/test_simple.py &
rm -rf ./train
tensorboard --logdir=./train
