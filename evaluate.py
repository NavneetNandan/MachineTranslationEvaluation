import os
import subprocess
import sys
from pythonrouge.pythonrouge import Pythonrouge
from nltk import RegexpTokenizer
import lebleu.lebleu.lebleu as lb


def errmsg():
    print("""Invalid command.
Usage: 
    python3 evaluate.py hypothesis_file_name reference_file _name [-c|-s]
        -c: calculate score whole corpus wise
        -s: calculate score sentence by sentence""")
    sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) == 4:
        hypothesis_file_path = sys.argv[1]
        reference_file_path = sys.argv[2]
        mode = sys.argv[3]
        tokenizer = RegexpTokenizer(r'\w+')
        ROUGE_path = "/home/navneet/pythonrouge/pythonrouge/RELEASE-1.5.5/ROUGE-1.5.5.pl" # ROUGE-1.5.5.pl
        data_path = "/home/navneet/pythonrouge/pythonrouge/RELEASE-1.5.5/data"  # data folder in RELEASE-1.5.5
        leblue_scorer = lb.LeBLEU()
        if os.path.exists(hypothesis_file_path) and os.path.exists(reference_file_path):
            if mode in ["-c", "-s"]:
                with open(hypothesis_file_path, "r") as hypothesis_file:
                    with open(reference_file_path, "r") as reference_file:
                        hypothesis_sentences = [hypothesis_sentence.rstrip()
                                                for hypothesis_sentence in hypothesis_file.readlines()]
                        reference_sentences = [reference_sentence.rstrip()
                                               for reference_sentence in reference_file.readlines()]
                        if len(reference_sentences) == len(hypothesis_sentences):
                            with open("hyp.temp", "w") as temp_hyp:
                                for i, hypothesis_sentence in enumerate(hypothesis_sentences):
                                    temp_hyp.write('{0} ({1})\n'.format(hypothesis_sentence, str(i)))
                            with open("ref.temp", "w") as temp_ref:
                                for i, reference_sentence in enumerate(reference_sentences):
                                    temp_ref.write('{0} ({1})\n'.format(reference_sentence, str(i)))
                            subprocess.run(
                                "java -jar tercom-0.7.25/tercom.7.25.jar -r ref.temp -h hyp.temp -n out -o sum",
                                stdout=subprocess.PIPE, shell=True)
                            with open("out.sum", "r") as eval_out:
                                out_sum_lines = eval_out.readlines()[5:len(reference_sentences) + 5]
                            meteor_evaluation_output = subprocess.run(
                                "java -Xmx2G -jar ~/meteor_test/meteor-*.jar {0} {1} -l en -norm".format(
                                    hypothesis_file_path, reference_file_path),
                                stdout=subprocess.PIPE, shell=True)
                            meteor_evaluation_output = meteor_evaluation_output.stdout.decode("UTF-8")
                            meteor_evaluation_sentence_wise = meteor_evaluation_output.split('\n')[
                                                              11:11 + len(reference_sentences)]

                            # rouge = Pythonrouge(n_gram=2, ROUGE_SU4=True, ROUGE_L=True, stemming=True, stopwords=True,
                            #                     word_level=True, length_limit=True, length=50, use_cf=False, cf=95,
                            #                     scoring_formula="average", resampling=True, samples=1000, favor=True,
                            #                     p=0.5)
                            if mode == '-s':
                                evaluation_list = list(range(0, len(hypothesis_sentences)))
                                count = 0
                                evaluation_output = subprocess.run(
                                    "./mteval/build/bin/mteval-sentence -e BLEU RIBES NIST WER -r " +
                                    reference_file_path + " -h " + hypothesis_file_path, stdout=subprocess.PIPE,
                                    shell=True)
                                evaluation_output = evaluation_output.stdout.decode("UTF-8")
                                # print(evaluation_output)
                                evaluation_sentence_wise = evaluation_output.split("\n")
                                for i, hypothesis_sentence in enumerate(hypothesis_sentences):
                                    evaluation_list[i] = {}
                                    reference_sentence = reference_sentences[i]
                                    evaluation_list[i]['lbleu'] = leblue_scorer.eval_single(hypothesis_sentence,
                                                                                            reference_sentence)
                                    results = evaluation_sentence_wise[i].split("\t")
                                    evaluation_list[i]['bleu'] = float(results[0].split("=")[1])
                                    evaluation_list[i]['ribes'] = float(results[1].split("=")[1])
                                    evaluation_list[i]['nist'] = float(results[2].split("=")[1])
                                    evaluation_list[i]['wer'] = float(results[3].split("=")[1])
                                    evaluation_list[i]['ter'] = float(out_sum_lines[i].split("|")[8].lstrip().rstrip()) \
                                                                / 100
                                    evaluation_list[i]['meteor'] = float(
                                        meteor_evaluation_sentence_wise[i].split(":")[1].rstrip().lstrip())
                                    # setting_file = rouge.setting(files=False, summary=hypothesis_sentence,
                                    #                              reference=reference_sentence)
                                    # result = rouge.eval_rouge(setting_file, recall_only=True, ROUGE_path=ROUGE_path,
                                    #                           data_path=data_path)
                                    # print(result)
                                print(evaluation_list)
                            if mode == '-c':
                                lbleu_score = leblue_scorer.eval(hypothesis_sentences,reference_sentences)
                                print(lbleu_score)
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
