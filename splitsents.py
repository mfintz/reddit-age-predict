import nltk
# nltk.download()
from nltk.tokenize import TweetTokenizer, sent_tokenize
import os
import codecs
import re




WHITE_SPACE = [' ', '\n', '\r', '\t']

# own split sentences , might be too tuned
def splitSent(input_path):
    dir_files = os.listdir(input_path)


    #TODO: also , take care of skipping the folder - it is considered as a file now
    if not os.path.exists(input_path + "\\splitsents\\"):
        os.makedirs(input_path + "\\splitsents\\")
    for file in dir_files:
        if file.endswith(".txt"):
            # new file
            sents = codecs.open(input_path + "\\splitsents\\" + file + ".splitsentences", 'w', "utf-8")
            # source file
            src = codecs.open(input_path + "\\" + file, 'r',"utf-8")
            # for line in src:
            #     print()
            #
            # sents.close()

            sentences = []
            final_text = ""
            for line in src:
                final_text += line

            i = 0
            for j in range(len(final_text)):
                if i < j:
                    # any line break (we entered) ends a sentence
                    if final_text[j] in ['\r', '\n']:
                        sentences.append(final_text[i:j])
                        i = j + 1
                    # look for characters which signify end of sentence
                    elif final_text[j] in ['.', ';', '?', '!']:
                        # a '.'
                        if final_text[j] == '.':
                            k = j + 1
                            # a '.' with whitespace afterwards
                            if (k < len(final_text)) and (final_text[k] in WHITE_SPACE):
                                sentences.append(final_text[i:k])
                                i = k + 1
                            # a '.' with quotes afterwards
                            elif (k < len(final_text)) and (final_text[k] in ['"', '״', '\'']):
                                if (k + 1 < len(final_text)) and (final_text[k + 1] in WHITE_SPACE):
                                    sentences.append(final_text[i:k + 1])
                                    i = k + 2
                            else:
                                # look for more '.'s
                                while (k < len(final_text)) and (final_text[k] == '.'):
                                    k += 1
                                # many '.'s. otherwise - a letter after a '.' - not the end of the sentence
                                if k>= len(final_text):
                                    print("debug")
                                if final_text[k] in WHITE_SPACE:
                                    sentences.append(final_text[i:k])
                                    i = k + 1
                        # a '?' or a '!'
                        elif final_text[j] in ['?', '!']:
                            k = j + 1
                            while (k < len(final_text)) and (final_text[k] in ['?', '!']):
                                k += 1
                            # a whitespace should appear after them
                            if final_text[k] in WHITE_SPACE:
                                sentences.append(final_text[i:k])
                                i = k + 1
                        # a ';'
                        else:
                            k = j + 1
                            if (k < len(final_text)) and (final_text[k] in WHITE_SPACE):
                                sentences.append(final_text[i:k])
                                i = k + 1
            first_sentence = True
            for sent in sentences:
                if first_sentence:
                    first_sentence = False
                else:
                    sents.write('\r\n')
                sents.write(sent)

        sents.close()


# general splitter, using NLTK
def nltkSplitSents(input_path):
    dir_files = os.listdir(input_path)


    if not os.path.exists(input_path + "\\splitsentsNLTKstyle\\"):
        os.makedirs(input_path + "\\splitsentsNLTKstyle\\")
    for file in dir_files:
        if file.endswith(".txt"):
            # new file
            sents = codecs.open(input_path + "\\splitsentsNLTKstyle\\" + file + ".splitsentencesNLTK", 'w', "utf-8")
            # source file
            src = codecs.open(input_path + "\\" + file, 'r',"utf-8")

            final_text = ""
            for line in src:
                final_text += line

            sent_text = nltk.sent_tokenize(final_text)

            first_sentence = True
            for sent in sent_text:
                if first_sentence:
                    first_sentence = False
                else:
                    sents.write('\r\n')
                sents.write(sent)



            sents.close()



def cleanSents(input_path):
    dir_files = os.listdir(input_path)


    if not os.path.exists(input_path + "\\cleanSents\\"):
        os.makedirs(input_path + "\\cleanSents\\")
    for file in dir_files:
        if file.endswith(".splitsentencesNLTK") or file.endswith(".splitsentences"):
            # new file
            sents = codecs.open(input_path + "\\cleanSents\\" + file + ".clean", 'w', "utf-8")
            # source file
            src = codecs.open(input_path + "\\" + file, 'r',"utf-8")

            url_reg = r"http\S+"
            url_reg2 = r"\(http\S+"

            clean_lines = []
            for line in src:
                newline = re.sub(url_reg, "", line)
                newline = re.sub(url_reg2, "", newline)
                newline.replace("\r","")
                clean_lines.append(newline)




            first_sentence = True
            for sent in clean_lines:
                # if first_sentence:
                #     first_sentence = False
                # else:
                #     sents.write('\r\n')
                sents.write(sent)



            sents.close()




# splitSent("jsons\\texts")

# nltkSplitSents("jsons\\texts")

cleanSents("jsons\\texts\\splitsentsNLTKstyle")