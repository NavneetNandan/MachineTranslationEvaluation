import os
import subprocess
import sys
from pyrouge.pyrouge.rouge import Rouge155
from pprint import pprint
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
        leblue_scorer = lb.LeBLEU()
        rouge = Rouge155(n_words=100)
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
                                out_sum_lines = eval_out.readlines()[5:len(reference_sentences) + 7]
                            meteor_evaluation_output = subprocess.run(
                                "java -Xmx2G -jar ~/meteor_test/meteor-*.jar {0} {1} -l en -norm".format(
                                    hypothesis_file_path, reference_file_path),
                                stdout=subprocess.PIPE, shell=True)
                            meteor_evaluation_output = meteor_evaluation_output.stdout.decode("UTF-8")
                            meteor_evaluation_sentence_wise = meteor_evaluation_output.split('\n')[
                                                              11:11 + len(reference_sentences)]

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
                                    evaluation_list[i]['ter'] = float(out_sum_lines[i].split("|")[8].lstrip()
                                                                      .rstrip()) / 100
                                    evaluation_list[i]['meteor'] = float(
                                        meteor_evaluation_sentence_wise[i].split(":")[1].rstrip().lstrip())
                                    evaluation_list[i]['rouge_1'] = \
                                        rouge.score_summary(hypothesis_sentence, {'a': reference_sentence})[
                                            'rouge_1_f_score']
                                    evaluation_list[i]['rouge_2'] = \
                                        rouge.score_summary(hypothesis_sentence, {'a': reference_sentence})[
                                            'rouge_2_f_score']
                                    evaluation_list[i]['rouge_3'] = \
                                        rouge.score_summary(hypothesis_sentence, {'a': reference_sentence})[
                                            'rouge_3_f_score']
                                    evaluation_list[i]['rouge_4'] = \
                                        rouge.score_summary(hypothesis_sentence, {'a': reference_sentence})[
                                            'rouge_4_f_score']
                                    evaluation_list[i]['rouge_su4'] = \
                                        rouge.score_summary(hypothesis_sentence, {'a': reference_sentence})[
                                            'rouge_su4_f_score']
                                pprint(evaluation_list)
                            if mode == '-c':
                                corpus_scores = {'lbleu': leblue_scorer.eval(hypothesis_sentences, reference_sentences)}
                                evaluation_output = subprocess.run(
                                    "./mteval/build/bin/mteval-corpus -e BLEU RIBES NIST WER -r " +
                                    reference_file_path + " -h " + hypothesis_file_path, stdout=subprocess.PIPE,
                                    shell=True)
                                evaluation_output = evaluation_output.stdout.decode("UTF-8")
                                results = evaluation_output.split("\t")
                                corpus_scores['bleu'] = float(results[0].split("=")[1])
                                corpus_scores['ribes'] = float(results[1].split("=")[1])
                                corpus_scores['nist'] = float(results[2].split("=")[1])
                                corpus_scores['wer'] = float(results[3].split("=")[1])
                                corpus_scores['ter'] = float(out_sum_lines[-1].split("|")[8].lstrip().rstrip()) / 100
                                corpus_scores['meteor'] = float(
                                    meteor_evaluation_output.split("\n")[-2].split(":")[1].rstrip().lstrip())
                                rouge1_sum = 0
                                rouge2_sum = 0
                                rouge3_sum = 0
                                rouge4_sum = 0
                                rougesu4_sum = 0
                                for i, hypothesis_sentence in enumerate(hypothesis_sentences):
                                    rouge1_sum += rouge.score_summary(hypothesis_sentence, {'a': reference_sentence})[
                                        'rouge_1_f_score']
                                    rouge2_sum += rouge.score_summary(hypothesis_sentence, {'a': reference_sentence})[
                                        'rouge_2_f_score']
                                    rouge3_sum += rouge.score_summary(hypothesis_sentence, {'a': reference_sentence})[
                                        'rouge_3_f_score']
                                    rouge4_sum += rouge.score_summary(hypothesis_sentence, {'a': reference_sentence})[
                                        'rouge_4_f_score']
                                    rougesu4_sum += rouge.score_summary(hypothesis_sentence, {'a': reference_sentence})[
                                        'rouge_su4_f_score']
                                corpus_scores['rouge_1'] = rouge1_sum / len(hypothesis_sentences)
                                corpus_scores['rouge_2'] = rouge2_sum / len(hypothesis_sentences)
                                corpus_scores['rouge_3'] = rouge3_sum / len(hypothesis_sentences)
                                corpus_scores['rouge_4'] = rouge4_sum / len(hypothesis_sentences)
                                corpus_scores['rouge_su4'] = rougesu4_sum / len(hypothesis_sentences)
                                pprint(corpus_scores)

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
