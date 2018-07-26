import math
import os
import codecs
import pickle
import shutil
import string
from collections import OrderedDict

import matplotlib.pyplot as plt
import nltk
import operator
from numpy.random import choice
from random import shuffle
import importlib

import numpy as np
from nltk import TweetTokenizer
from sklearn import svm, cross_validation, metrics, naive_bayes, tree, neighbors, linear_model
from sklearn import feature_extraction
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.feature_selection import SelectKBest, chi2
from en_shell_nouns import SHELL_NOUNS
from en_function_words import FUNCTION_WORDS

NUM_OF_SAMPLES = 1000
BATCH_SIZE = 1500
NUM_OF_CLASSES = 8
MERGED_FILES_LOCATION = "jsons\\texts\\splitsentsNLTKstyle\\cleanSents\\mergedFiles\\"
BATCHED_FILES_LOCATION = "jsons\\texts\\splitsentsNLTKstyle\\cleanSents\\mergedFiles\\batched\\"

debug_file = open("debug.txt", 'w')
top_500_file = codecs.open("top500words.txt",'w','utf-8')


# simple helper function to remove indices from a list
def removeIndices(input_list:list , indices_to_remove:list):
    for i in indices_to_remove:
        #if i >= len(input_list):
            # print()
        input_list.remove(i)

def getInd(src:list,num):
    for ind,val in src:
        if num in val:
            return ind

# from each class , choose NUM_OF_SAMPLES  - each sample is a batch at size BATCH_SIZE (BATCH_SIZE sentences)
# def pickSamples(input_path):
#     dir_files = os.listdir(input_path)
#     tknzr = TweetTokenizer()
#
#     if os.path.exists(input_path + "\\batched\\"):
#         shutil.rmtree(input_path + "\\batched\\")
#         os.makedirs(input_path + "\\batched\\")
#     else:
#         os.makedirs(input_path + "\\batched\\")
#     for file in dir_files:
#         batches = []
#         if file.endswith(".merged"):
#             # saving to file , mainly for debug
#
#             # new file - each line at the file will contain a batch (certain number of sentences)
#             sents = codecs.open(input_path + "\\batched\\" + file + ".batched", 'w', "utf-8")
#             # source file
#             src = codecs.open(input_path + "\\" + file, 'r',"utf-8")
#
#             file_len = 0
#             batch_ind = 0
#             all_batch_ind = 0
#             for line in src:
#                 file_len += 1
#                 if file_len == 99242:
#                     print()
#
#             t = [i for i in range(file_len)]
#
#             num_of_possible_batches = int(file_len / BATCH_SIZE)
#
#             # generate a list (of lists) holding the lines to be batched
#             for i in range(num_of_possible_batches):
#                 # choose sentences indices from file, uniformly
#                 indices_to_choose = sorted(choice(t, BATCH_SIZE, replace=False))
#                 batches.append(indices_to_choose)
#                 removeIndices(t, indices_to_choose)
#
#             batched_lines = [""]*num_of_possible_batches
#             for index, line in enumerate(src):
#                 current_batch_line_number = getInd(batches,index)
#                 batched_lines[current_batch_line_number].join(line)
#
#
#             for i in range(len(batched_lines)):
#                 sents.write(batched_lines[i])
#
#             src.close()


# pickSamples("jsons\\texts\\splitsentsNLTKstyle\\cleanSents\\mergedFiles\\")


# making batches and returning the lowest number of batches generated for downsampling later
def makeBatches(input_path):
    dir_files = os.listdir(input_path)
    shortest = 999999
    if os.path.exists(input_path + "\\batched\\"):
        shutil.rmtree(input_path + "\\batched\\")
        os.makedirs(input_path + "\\batched\\")
    else:
        os.makedirs(input_path + "\\batched\\")
    for file in dir_files:
        batches = []
        if file.endswith(".merged"):
            # saving to file , mainly for debug

            # new file - each line at the file will contain a batch (certain number of sentences)
            sents = codecs.open(input_path + "\\batched\\" + file + ".batched", 'w', "utf-8")
            # source file
            src = codecs.open(input_path + "\\" + file, 'r', "utf-8")

            source_text = []
            for line in src:
                source_text.append(line)
            shuffle(source_text)

            # source file's text is now shuffled so we can choose consecutive BATCH_SIZE of lines

            num_of_possible_batches = int(len(source_text) / BATCH_SIZE)

            if num_of_possible_batches < shortest:
                shortest = num_of_possible_batches

            j = 0
            while num_of_possible_batches > 0 :
                batch = []
                for i in range(BATCH_SIZE):
                    batch.append(source_text[j])
                    j += 1
                cur_b = " ".join(batch)
                batches.append(cur_b)
                num_of_possible_batches -= 1

            for b in batches:
                sents.write(b.replace("\n","").replace("\r",""))
                sents.write("\n")
    return shortest



# returning equally divided batches (according to the shortest batch that was created earlier)
def getBatches(input_path, shortest):
    dir_files = os.listdir(input_path)

    batches = []
    for file in dir_files:
        batch = []
        if file.endswith(".batched"):
            # saving to file , mainly for debug

            # new file - each line at the file will contain a batch (certain number of sentences), divided equally
            sents = codecs.open(input_path + "\\" + file + ".Equally", 'w', "utf-8")
            # source file
            src = codecs.open(input_path + "\\" + file, 'r', "utf-8")

            count = 0
            for line in src:
                # if len(line) < 200:
                #     continue
                batch.append(line)
                sents.write(line)
                count += 1
                if count == shortest:
                    break


            sents.close()
        batches.append(batch)
    return batches


def makeWordsBatches(input_path):
    dir_files = os.listdir(input_path)
    shortest = 999999
    if os.path.exists(input_path + "\\batched\\"):
        shutil.rmtree(input_path + "\\batched\\")
        os.makedirs(input_path + "\\batched\\")
    else:
        os.makedirs(input_path + "\\batched\\")
    master_of_batches = []
    for file in dir_files:
        batches = []
        if file.endswith(".merged"):
            # saving to file , mainly for debug

            # new file - each line at the file will contain a batch (certain number of sentences)
            sents = codecs.open(input_path + "\\batched\\" + file + ".batched", 'w', "utf-8")
            # source file
            src = codecs.open(input_path + "\\" + file, 'r', "utf-8")

            # tokenize , without ","
            source_text = []
            # exclude = set(string.punctuation)
            word_count = 0
            for line in src:
                tokenized = line.replace(",", "").split()
                if len(tokenized) > 0:
                    source_text.append(tokenized)
                    word_count += len(tokenized)
            shuffle(source_text)

            # source file's text is now shuffled so we can choose consecutive BATCH_SIZE of lines

            num_of_possible_batches = int(word_count / BATCH_SIZE)

            if num_of_possible_batches < shortest:
                shortest = num_of_possible_batches

            row = []
            row_len = 0
            for i in range(len(source_text)):
                if row_len < BATCH_SIZE:
                    row_len += len(source_text[i])
                    row.append(" ".join(source_text[i]))
                else:
                    batches.append(row)
                    row = []
                    row_len = 0
                    row.append(" ".join(source_text[i]))
                    row_len += len(source_text[i])

            str_batches = []
            for b in batches:
                cur = " ".join(b).replace("\n", "").replace("\r", "")
                sents.write(cur)
                sents.write("\n")
                str_batches.append(cur)
                print("current batch size:", len(cur.split()))
            master_of_batches.append(str_batches)

    return master_of_batches


def creatPlot(info):
    # data to plot
    n_groups = 3

    means_13 = [math.log(y, 10) for y in info[0]]
    means_14 = [math.log(y, 10) for y in info[1]]
    means_15 = [math.log(y, 10) for y in info[2]]
    means_16 = [math.log(y, 10) for y in info[3]]
    means_17 = [math.log(y, 10) for y in info[4]]
    means_18 = [math.log(y, 10) for y in info[5]]
    means_19 = [math.log(y, 10) for y in info[6]]
    means_20 = [math.log(y, 10) for y in info[7]]

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.1
    opacity = 0.8

    rects1 = plt.bar(index, means_13, bar_width,
                     alpha=opacity,
                     color='blue',
                     label='13')

    rects2 = plt.bar(index + bar_width, means_14, bar_width,
                     alpha=opacity,
                     color='green',
                     label='14')

    rects3 = plt.bar(index + bar_width * 2, means_15, bar_width,
                     alpha=opacity,
                     color='red',
                     label='15')
    rects4 = plt.bar(index + bar_width * 3, means_16, bar_width,
                     alpha=opacity,
                     color='pink',
                     label='16')

    rects5 = plt.bar(index + bar_width * 4, means_17, bar_width,
                     alpha=opacity,
                     color='brown',
                     label='17')

    rects6 = plt.bar(index + bar_width * 5, means_18, bar_width,
                     alpha=opacity,
                     color='orange',
                     label='18')

    rects7 = plt.bar(index + bar_width * 6, means_19, bar_width,
                     alpha=opacity,
                     color='y',
                     label='19')

    rects8 = plt.bar(index + bar_width * 7, means_20, bar_width,
                     alpha=opacity,
                     color='purple',
                     label='20')

    plt.xlabel('Age')
    plt.ylabel('Scores (logged) ')
    plt.title('Scores by age')
    plt.xticks(index + bar_width, ('word count', '# of lines', 'avg sent len'))
    plt.legend()

    plt.tight_layout()
    plt.show()



def filesInfo(input_path):
    dir_files = os.listdir(input_path)
    all_info = []
    for file in dir_files:
        cur_info = []
        if file.find(".") == -1:
            continue
        # source file
        src = codecs.open(input_path + "\\" + file, 'r', "utf-8")
        word_count = 0
        line_sum = 0
        number_of_lines = 0
        for line in src:
            if len(line) < 2:
                continue
            exclude = set(string.punctuation)
            s = ''.join(ch for ch in line if ch not in exclude)
            line_length = len(s.split())
            word_count += line_length
            line_sum += line_length
            number_of_lines += 1
        average_sent_len = line_sum / number_of_lines
        print(file[:2], "has", word_count,"words")
        cur_info.append(word_count)
        print(file[:2], "has", number_of_lines, "sentences")
        cur_info.append(number_of_lines)
        print(file[:2], "has", average_sent_len, "average sentence length")
        cur_info.append(average_sent_len)

        all_info.append(cur_info)
    return all_info



def createDict(input_file):

    # source file
    src = codecs.open(input_file, 'r', "utf-8")
    all_mighty_dict = {}
    for line in src:
        for word in line.split():
            if word in all_mighty_dict:
                all_mighty_dict[word] += 1
            else:
                all_mighty_dict[word] = 1
    return all_mighty_dict


def get_tokens(text):
    one_line = text.replace('\n', ' ')      # Replace linebreaks with spaces
    return one_line.split()

def analyze_text(input_file):
    sentences = codecs.open(input_file, 'r', "utf-8")




    # sentences = text.split("\n")  # split the text into sentences to avoid line breaks

    for s in sentences:
        sent_tokens = get_tokens(s)

        for tok_ind in range(len(sent_tokens)):
            unigram = sent_tokens[tok_ind]
            if unigram not in analyze_text.unigrams_dict:
                analyze_text.unigrams_dict[unigram] = 1  # Seeing it for the first time
            else:
                analyze_text.unigrams_dict[unigram] += 1  # already saw, increasing the count

            # if tok_ind < (len(sent_tokens) - 1):
            #     bigram = sent_tokens[tok_ind] + " " + sent_tokens[tok_ind + 1]
            #     if bigram not in analyze_text.bigrams_dict:
            #         analyze_text.bigrams_dict[bigram] = 1
            #     else:
            #         analyze_text.bigrams_dict[bigram] += 1

            # if tok_ind < (len(sent_tokens) - 2):
            #     trigram = sent_tokens[tok_ind] + " " + sent_tokens[tok_ind + 1] + " " + sent_tokens[tok_ind + 2]
            #     if trigram not in analyze_text.trigrams_dict:
            #         analyze_text.trigrams_dict[trigram] = 1
            #     else:
            #         analyze_text.trigrams_dict[trigram] += 1

        analyze_text.total_num_unigrams += len(sent_tokens)
        # analyze_text.total_num_bigrams += ((len(sent_tokens) - 1) if (len(sent_tokens) >= 2) else 0)
        # analyze_text.total_num_trigrams += ((len(sent_tokens) - 2) if (len(sent_tokens) >= 3) else 0)

    return analyze_text.unigrams_dict, analyze_text.bigrams_dict, analyze_text.trigrams_dict, analyze_text.total_num_unigrams, analyze_text.total_num_bigrams, analyze_text.total_num_trigrams

analyze_text.unigrams_dict = {}
analyze_text.bigrams_dict = {}
analyze_text.trigrams_dict = {}
analyze_text.total_num_unigrams = 0
analyze_text.total_num_bigrams = 0
analyze_text.total_num_trigrams = 0


# info = filesInfo("jsons\\texts\\splitsentsNLTKstyle\\cleanSents\\mergedFiles\\")
# creatPlot(info)
############################### SENTENCES BATCHES #####################################################
# shortest_batch = makeBatches("jsons\\texts\\splitsentsNLTKstyle\\cleanSents\\mergedFiles\\")
# print(shortest_batch)
# all_ages_batched = getBatches("jsons\\texts\\splitsentsNLTKstyle\\cleanSents\\mergedFiles\\batched",shortest_batch)
# #
# pickle.dump(all_ages_batched, open('all_ages_batched.p', 'wb'))

# use this line when working only , to use previous pickle. otherwise , comment it and uncomment previous lines (not pickle)
# all_ages_batched = pickle.load(open('all_ages_batched.p', "rb"))
#######################################################################################################


############################### WORDS BATCHES #####################################################
master_of_batches = makeWordsBatches("jsons\\texts\\splitsentsNLTKstyle\\cleanSents\\mergedFiles\\")
shortest_batch = min([len(master_of_batches[i]) for i in range(len(master_of_batches))])
print(shortest_batch)
all_ages_batched = getBatches("jsons\\texts\\splitsentsNLTKstyle\\cleanSents\\mergedFiles\\batched",shortest_batch)

# pickle.dump(all_ages_batched, open('all_ages_batched.p', 'wb'))

# use this line when working only , to use previous pickle. otherwise , comment it and uncomment previous lines (not pickle)
# all_ages_batched = pickle.load(open('all_ages_batched.p', "rb"))
#######################################################################################################



# 19 isn't merged with 20 for now
train_labels = np.asarray([13]*len(all_ages_batched[0]) +
                           [14]*len(all_ages_batched[0]) +
                           [15]*len(all_ages_batched[0]) +
                           [16]*len(all_ages_batched[0]) +
                           [17]*len(all_ages_batched[0]) +
                           [18]*len(all_ages_batched[0]) +
                           [19]*len(all_ages_batched[0]) +
                           [20]*len(all_ages_batched[0])
                            )

# prepare n-grams data
n_grams_all_ages = []
dir_files = os.listdir(BATCHED_FILES_LOCATION)

for file in dir_files:
    n_age = []
    if file.__contains__(".Equally"):
        unigrams_dict, bigrams_dict, trigrams_dict, total_num_unigrams, total_num_bigrams, total_num_trigrams = analyze_text(BATCHED_FILES_LOCATION + file)
    else:
        unigrams_dict, bigrams_dict, trigrams_dict, total_num_unigrams, total_num_bigrams, total_num_trigrams = analyze_text(BATCHED_FILES_LOCATION + file + ".Equally")
    n_age.append(unigrams_dict)
    n_age.append(bigrams_dict)
    n_age.append(trigrams_dict)
    n_age.append(total_num_unigrams)
    n_age.append(total_num_bigrams)
    n_age.append(total_num_trigrams)
    n_grams_all_ages.append(n_age)

top_500_in_dict = (sorted(analyze_text.unigrams_dict.items(), key=operator.itemgetter(1), reverse=True))


all_mighty_dict_size = len(analyze_text.unigrams_dict)
feature_len = 6 + len(top_500_in_dict)
feat_vec_mat = np.zeros((len(all_ages_batched[0])*NUM_OF_CLASSES, feature_len))




REFERENCES = ["I", "me", "mine", "you", "your", "we", "our", "him", "her", "they"]



def featureVecotrize(batch):
    # global debug_procces_words
    tknzr = TweetTokenizer()
    vect = [0] * (feature_len - 6)
    tokenized_batch = tknzr.tokenize(batch)
    sent_text = nltk.sent_tokenize(batch)
    # ordered_dict_listed = tuple(OrderedDict(analyze_text.unigrams_dict).keys())
    ordered_dict_listed = tuple(OrderedDict(top_500_in_dict).keys())
    len_sum = 0
    num_of_words = 0
    shell_nouns_count = 0
    references_count = 0
    function_words_count = 0
    for word in tokenized_batch:
        if word in ordered_dict_listed:
            vect[ordered_dict_listed.index(word)] += 1 / len(all_ages_batched[0])
        if word in SHELL_NOUNS:
            shell_nouns_count += 1
        if word in REFERENCES:
            references_count += 1
        if word in FUNCTION_WORDS:
            function_words_count += 1
        if len(word) < 2 or (len(word) == 1 and "." in word or "," in word):
            continue
        else:
            num_of_words +=1
            len_sum += len(word)


    avg_sent = sum([len(sent.replace(","," ").split()) for sent in sent_text])/ len(sent_text)
    batch_wo_punc = batch.replace(",","")
    unique_words = len(set((batch_wo_punc.split())))

    avg_word = len_sum/num_of_words

    vect.append(avg_sent)
    vect.append(avg_word)
    vect.append(shell_nouns_count/len(all_ages_batched[0]))
    vect.append(references_count/len(all_ages_batched[0]))
    vect.append(function_words_count/len(all_ages_batched[0]))
    vect.append(unique_words/len(all_ages_batched[0]))
    # print(avg_sent,avg_word,shell_nouns_count,references_count,function_words_count,unique_words,file=debug_file)

    return vect







vect_num = 0
# fill feature matrix
for age,batch_age in enumerate(all_ages_batched):
    # all_mighty_dict,bigrams_dict, trigrams_dict, total_num_unigrams, total_num_bigrams, total_num_trigrams = analyze_text("jsons\\texts\\splitsentsNLTKstyle\\cleanSents\\mergedFiles\\batched\\" + str(age + 13) + ".txt.splitsentencesNLTK.clean.merged.batched")
    for batch_index,batch in enumerate(batch_age):
        vect = featureVecotrize(batch)
        vect_num += 1
        print("vect #", vect_num)
        for ft_index in range(len(vect)):
            feat_vec_mat[batch_index + len(all_ages_batched[0])*age][ft_index] = vect[ft_index]


# X_new = SelectKBest(chi2, k=20).fit_transform(feat_vec_mat, train_labels)


# 10-fold cross validation
kf = cross_validation.KFold(len(all_ages_batched)*len(all_ages_batched[0]), n_folds=10, shuffle=True)
avg_acc_SVC = avg_acc_NB = avg_acc_DT = avg_acc_KNN = avg_acc_regr = avg_r2_regr = mae_regr = 0
avg_acc_lasso_regr = avg_r2_lasso_regr = mae_lasso_regr = avg_acc_ridge_regr = avg_r2_ridge_regr = mae_ridge_regr = 0

alpha_lasso = 0.1
alpha_ridge = .5

for train_ind, test_ind in kf:

    clf_SVC = OneVsOneClassifier(svm.LinearSVC())
    clf_NB = OneVsOneClassifier(naive_bayes.MultinomialNB())
    # clf_DT = OneVsOneClassifier(tree.DecisionTreeClassifier())
    # clf_KNN = OneVsOneClassifier(neighbors.KNeighborsClassifier())
    regr = linear_model.LinearRegression()


    ridge_regr = linear_model.Ridge(alpha=alpha_ridge)

    lasso_regr = linear_model.Lasso(alpha=alpha_lasso)

    # Split the labels into training and test subsets
    curr_train_labels = train_labels[train_ind]
    curr_test_labels = train_labels[test_ind]

    # Train the classifiers on 9/10 of the samples
    clf_SVC.fit(feat_vec_mat[train_ind, :], curr_train_labels)
    print("fitting SVM\n")
    clf_NB.fit(feat_vec_mat[train_ind, :], curr_train_labels)
    print("fitting NB\n")
    # clf_DT.fit(feat_vec_mat[train_ind, :], curr_train_labels)
    # print("fitting DT\n")
    # clf_KNN.fit(feat_vec_mat[train_ind, :], curr_train_labels)
    # print("fitting KNN\n")
    regr.fit(feat_vec_mat[train_ind, :], curr_train_labels)
    print("fitting LR\n")
    ridge_regr.fit(feat_vec_mat[train_ind, :], curr_train_labels)
    print("fitting ridge LR\n")
    lasso_regr.fit(feat_vec_mat[train_ind, :], curr_train_labels)
    print("fitting ridge LR\n")

    # Test the classifiers on the remaining 1/10 of the samples
    # SVC prediction
    curr_pred_labels_SVC = clf_SVC.predict(feat_vec_mat[test_ind, :])
    acc = metrics.accuracy_score(curr_test_labels, curr_pred_labels_SVC)
    avg_acc_SVC += acc

    # NaiveBayes prediction
    curr_pred_labels_NB = clf_NB.predict(feat_vec_mat[test_ind, :])
    acc = metrics.accuracy_score(curr_test_labels, curr_pred_labels_NB)
    avg_acc_NB += acc

    # DecisionTree prediction
    # curr_pred_labels_DT = clf_DT.predict(feat_vec_mat[test_ind, :])
    # acc = metrics.accuracy_score(curr_test_labels, curr_pred_labels_DT)
    # avg_acc_DT += acc

    # KNN prediction
    # curr_pred_labels_KNN = clf_KNN.predict(feat_vec_mat[test_ind, :])
    # acc = metrics.accuracy_score(curr_test_labels, curr_pred_labels_KNN)
    # avg_acc_KNN += acc

    # Ridge Linear Regression prediction
    curr_pred_labels_ridge_regr = ridge_regr.predict(feat_vec_mat[test_ind, :])
    acc = mean_squared_error(curr_test_labels, curr_pred_labels_ridge_regr)
    ridge_r2 = r2_score(curr_test_labels, curr_pred_labels_ridge_regr)
    ridge_mae = mean_absolute_error(curr_test_labels, curr_pred_labels_ridge_regr)
    mae_ridge_regr += ridge_mae
    avg_acc_ridge_regr += acc
    avg_r2_ridge_regr += ridge_r2

    # The coefficients
    print('RIDGE:\n')
    print('Coefficients: \n', ridge_regr.coef_)
    print('Intercept: ',ridge_regr.intercept_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % acc)
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % ridge_r2)

    # MAE
    print('MAE score: %.2f' % ridge_mae)
    print('#########')




    # Ridge Linear Regression prediction
    curr_pred_labels_lasso_regr = lasso_regr.predict(feat_vec_mat[test_ind, :])
    acc = mean_squared_error(curr_test_labels, curr_pred_labels_lasso_regr)
    lasso_r2 = r2_score(curr_test_labels, curr_pred_labels_lasso_regr)
    lasso_mae = mean_absolute_error(curr_test_labels, curr_pred_labels_lasso_regr)
    mae_lasso_regr += lasso_mae
    avg_acc_lasso_regr += acc
    avg_r2_lasso_regr += lasso_r2

    # The coefficients
    print('LASSO:\n')
    print('Coefficients: \n', lasso_regr.coef_)
    print('Intercept: ',lasso_regr.intercept_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % acc)
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % lasso_r2)

    # MAE
    print('MAE score: %.2f' % lasso_mae)
    print('#########')






    # Linear Regression prediction
    curr_pred_labels_regr = regr.predict(feat_vec_mat[test_ind, :])
    acc = mean_squared_error(curr_test_labels, curr_pred_labels_regr)
    r2 = r2_score(curr_test_labels, curr_pred_labels_regr)
    mae = mean_absolute_error(curr_test_labels, curr_pred_labels_regr)
    mae_regr += mae
    avg_acc_regr += acc
    avg_r2_regr += r2

    print('Linear Regression:\n')
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % acc)
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2)

    # MAE
    print('MAE score: %.2f' % mae)
    print('#########')

    # Plot outputs
    # plt.scatter(feat_vec_mat[test_ind, 0], curr_test_labels, color='black')
    # plt.plot(feat_vec_mat[test_ind, :], curr_pred_labels_regr, color='blue', linewidth=3)

    # plt.xticks(())
    # plt.yticks(())
    #
    # plt.show()




print("--------------")
print("Batch size (words) is : ",BATCH_SIZE)
print("SVC Accuracy: " + str(avg_acc_SVC / 10))
print("NB Accuracy: " + str(avg_acc_NB / 10))
# print("DT Accuracy: " + str(avg_acc_DT / 10))
# print("KNN Accuracy: " + str(avg_acc_KNN / 10))
print("Linear Regression mse: " + str(avg_acc_regr / 10))
print("Linear Regression variance (r2) : " + str(avg_r2_regr / 10))
print("Mean Absolute error : " + str(mae_regr / 10))
print("###########\n")
print("LR WITH REG:\n")
print("Ridge alpha = ", alpha_ridge)
print("Ridge Linear Regression mse: " + str(avg_acc_ridge_regr / 10))
print("Ridge Linear Regression variance (r2) : " + str(avg_r2_ridge_regr / 10))
print("Ridge Mean Absolute error : " + str(mae_ridge_regr / 10))
print("--------")
print("Lasso alpha = ", alpha_lasso)
print("Lasso Linear Regression mse: " + str(avg_acc_lasso_regr / 10))
print("Lasso Linear Regression variance (r2) : " + str(avg_r2_lasso_regr / 10))
print("Lasso Mean Absolute error : " + str(mae_lasso_regr / 10))
print("--------")
# -----------------------------------------------------------------------------
