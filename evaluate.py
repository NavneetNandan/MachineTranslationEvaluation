import os
import sys

from nltk import RegexpTokenizer
from nltk.translate import bleu_score
import lebleu.lebleu.lebleu as lb


def errmsg():
    print("""Invalid command.
        Usage: 
        python3 evaluate.py hypothesis_file_name refernce_file _name [-c|-s]
            -c: calculate score whole corpus wise
            -s: calculate score sentence by sentence 
        """)
    sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) == 4:
        hypothesis_file_path = sys.argv[1]
        reference_file_path = sys.argv[2]
        mode = sys.argv[3]
        tokenizer = RegexpTokenizer(r'\w+')
        leblue_scorer = lb.LeBLEU()
        if os.path.exists(hypothesis_file_path) and os.path.exists(reference_file_path):
            if mode in ["-c", "-s"]:
                with open(hypothesis_file_path) as hypothesis_file:
                    with open(reference_file_path) as reference_file:
                        hypothesis_sentences = [hypothesis_sentence.rstrip()
                                                for hypothesis_sentence in hypothesis_file.readlines()]
                        reference_sentences = [reference_sentence.rstrip()
                                               for reference_sentence in reference_file.readlines()]
                        if mode == '-s':
                            evaluation_list = []
                            count = 0

                            for i, hypothesis_sentence in enumerate(hypothesis_sentences):
                                evaluation_list[i] = {}
                                reference_sentence = reference_sentences[i]
                                evaluation_list[i]['bleuscore'] = bleu_score.sentence_bleu(
                                    [tokenizer.tokenize(reference_sentence)], tokenizer.tokenize(hypothesis_sentences))
                                evaluation_list[i]['lbleuscore'] = leblue_scorer.eval_single(hypothesis_sentence,
                                                                                             reference_sentence)

                        else:
                            print("""Number of sentences in hypothesis file and reference file is not equal""")
                            sys.exit(1)
            else:
                errmsg()
        else:
            print("Hypothesis file or Reference file does not exist at specified location")
            errmsg()
    else:
        errmsg()
