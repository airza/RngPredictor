import subprocess
for i in range(12,23):
	import_count = 2**i
	print("doing it with import count %d"%import_count)
	with subprocess.Popen(["python3","forwardPassTrainer.py","33",str(import_count)], stdout=subprocess.PIPE, text=True) as proc:
		for line in proc.stdout:
			print(line, end='')
	# Wait for the subprocess to finish
	proc.wait()