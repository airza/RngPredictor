import subprocess
for i in range(64):
	print("!!!Running %d!!!"%i)
	subprocess.run(["python","trainer.py",str(i)])