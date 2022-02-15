nohup python train.py </dev/null >output.txt 2>&1 &
echo $! > save_pid.txt