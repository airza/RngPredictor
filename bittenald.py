import subprocess
for i in range(64):
	print("calling setupForwardPass.py on bit %d"%i)
	subprocess.call(["python3","setupForwardPass.py",str(i)])