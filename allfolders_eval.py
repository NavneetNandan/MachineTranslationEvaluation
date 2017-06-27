import os
import subprocess
for folderName, subfolders, filenames in os.walk('Data'):
	print('The current folder is ' + folderName)
	if 'hyp.txt' in filenames:
		if 'text.txt' in filenames:
			h = os.path.abspath(os.path.join(folderName, 'hyp.txt')).replace(' ','\ ')
			t = os.path.abspath(os.path.join(folderName, 'text.txt')).replace(' ','\ ')
			# h=os.path.relpath('hyp.txt').replace(' ','\ ')
			print(h)
			# t=os.path.relpath('text.txt').replace(' ','\ ')
			if not os.path.exists(os.path.abspath(os.path.join(folderName,'sentence_eval.csv'))) :
				subprocess.run("~/anaconda3/bin/python evaluate.py "+h+" "+t+" -s",shell=True)
			else:print("sentence already done")
			if not os.path.exists(os.path.abspath(os.path.join(folderName,'corpus_eval.txt'))):
				subprocess.run("~/anaconda3/bin/python evaluate.py "+h+" "+t+" -c",shell=True)
			else:print("corpus already done")