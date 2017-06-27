This folder contains scripts and supporting tools for automatic machine translation evaluation. Evaluation is done using following evaluation metrics:
1. Bleu
2. lBleu
3. meteor
4. WER
5. TER
6. rouge(1,2,3,4,4su)
7. NIST
8. ribes

dependencies:
java 1.7+
python-Levenshtein 
nltk == 3.2.4
numpy == 1.13.0
setuptools == 33.1.1

To run the script on an hypothesis and reference text file pair, use evaluate.py script using following command:
	python3 evaluate.py hypothesis_file_name reference_file _name [-c|-s]
	        -c: calculate score whole corpus wise
	        -s: calculate score sentence by sentence
python3.5+ is necessary
After evaluation sentence wise score is stored in sentence_eval.csv file and corpus wise score is stored in corpus_eval.txt in same folder of hypothesis text file.
sentence_eval.csv is tab seperated valid csv file having ordering as:
hypothesis	text	bleu	lbleu	ribes	nist	wer	ter	meteor	rouge_1	rouge_2	rouge_3	rouge_4	rouge_su4
corpus_eval.txt is valid json file and values of scores calculated for whole corpus.

To evaluate all file pairs inside a folder you can use all_folder_eval.py script. 
	Usage: 
		python3 allfolder_eval.py [path of directory]
This script will call above mentioned evaluate.py for every hypothesis and reference text file pair found. Make sure that all hypothesis text file is named as hyp.txt, reference file as text.txt and corresponding pair is stored in same folder. If already corpus_eval.txt or sentence_eva.csv is found in a folder then it will not execute corpus wise evaluation and sentence wise evalution respectively on hypothesis and text file pair of that folder. It browses folder in BFS manner.
Output of each pair is stored in folder containing that pair.
Make sure that no folder name contains whitespaces.
