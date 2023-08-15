import subprocess
for i in range(64):
	with subprocess.Popen(["python3","forwardPassTrainer.py",str(i)], stdout=subprocess.PIPE, text=True) as proc:
		for line in proc.stdout:
			print(line, end='')
	# Wait for the subprocess to finish
	proc.wait()