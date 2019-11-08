import random
import math

spam_ratio, words, spamicity = 0,[],[]
spam_ratio_bis, words_bis, spamicity_bis, spamicity_no, spamicity_inv, product_word_dic = 0,0,0,0,0,0
oracle = 0
seuil = 0.8
seuil_precision = 1
seed = 123
seuil_recall = 1

def split_lines(input, seed, output1, output2):
    """Distributes the lines of 'input' to 'output1' and 'output2' pseudo-randomly.

    The output files should be approximately balanced (50/50 chance for each line
    to go either to output1 or output2).

    Args:
      input: a string, the name of the input file.
      seed: an integer, the seed of the pseudo-random generator used. The split
          should be different with different seeds. Conversely, using the same
          seed and the same input should yield exactly the same outputs.
      output1: a string, the name of the first output file.
      output2: a string, the name of the second output file.
    """

    f = open(output1, "w")
    f2 = open(output2, "w")
    random.seed(seed)
    for line in open(input, 'r').readlines():

        if random.randint(1, 100) > 50:
            f2.write(line)
        else:
            f.write(line)


split_lines('input.txt', 500, 'test1.txt', 'test2.txt')


def tokenize_and_split(sms_file):
  
    """Parses and tokenizes the sms data, splitting 'spam' and 'ham' messages.

  Args:
    sms_file: a string, the name of the input SMS data file.

  Returns:
    A triple (words, l0, l1):
    - words is a dictionary mapping each word to a unique, dense 'word index'.
      The word indices must be in [0...len(words)-1].
    - l0 is a list of the 'spam' messages, encoded as lists of word indices.
    - l1 is like l0, but for 'ham' messages.
  """
    
    dic = {}
    list1 = []
    list2 = []
    i = -1
    ham = True
    for line in open(sms_file, 'r').readlines():
        w = []
        for word in line.split():
          i = i + 1
          if word == "ham":
            ham = True
            i = i - 1
          elif word == "spam":
            ham = False
            i = i - 1
          else:
            if word not in dic:
              dic[word] = i
              w.append(dic[word])
            else : 
              i = i - 1
              w.append(dic[word])
        if ham and w !=[]:
          list2.append(w)
        elif ham == False and w !=[]:
          list1.append(w)
  
    return dic,list1,list2

def tokenize_and_split_bis(sms_file):
  
    """Parses and tokenizes the sms data, splitting 'spam' and 'ham' messages.
       calculate tfIdf and can delete some words with modulate global variable oracle (not recommended because there are few words)

  Args:
    sms_file: a string, the name of the input SMS data file.

  Returns:
    A triple (words, l0, l1):
    - words is a dictionary mapping each word to a unique, dense 'word index'.
      The word indices must be in [0...len(words)-1].
    - l0 is a list of the 'spam' messages, encoded as lists of word indices.
    - l1 is like l0, but for 'ham' messages.
  """
    
    dic = {}
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    i = -1
    document = 0
    terms = 0
    new_document = True
    ham = True
    for line in open(sms_file, 'r').readlines():
        w = []
        document += 1
        new_document = True
        for word in line.split():
          i = i + 1
          if word == "ham":
            ham = True
            i = i - 1
          elif word == "spam":
            ham = False
            i = i - 1
          else:
            if word not in dic:
              dic[word] = i
              w.append(dic[word])
              list3.append(1)
              list4.append(1)
              new_document = False
              terms += 1
            else : 
              i = i - 1
              w.append(dic[word])
              list4[dic[word]] += 1
              terms += 1
              if new_document:              
                list3[dic[word]] += 1
                new_document = False
                
        if ham and w !=[]:
          list2.append(w)
        elif ham == False and w !=[]:
          list1.append(w)

    moy = 0
    len_dic = len(dic.keys())
    list5 = [0 for x in range(len_dic)]
    for key in dic.keys():
        if list4[dic[key]] > 0:
          tf = list4[dic[key]] / terms
          idf = math.log(document / list3[dic[key]])
          tfIdf = tf * idf
          list5[dic[key]] = tfIdf
          # print("the word " + str(key) + " appairs " + str(list4[dic[key]]) + " times.")
          # print("his frequency is " + str(list4[dic[key]] / terms) )
          # print("the word " + str(key) + " appairs " + str(list3[dic[key]]) + " times in each document.")
          # print("his frequency is " + str(idf))
          # print("utility " + str(tfIdf))
          moy += tfIdf
          
    moy = moy / len_dic      
    # print(moy)
    dic_bis = {}
    i = -1
    for key in dic.keys():
      value = list5[dic[key]]
      # print(str(value))
      if (value > oracle * moy):
        i += 1
        dic_bis[key] = i
      # else:
      #   print("not pass " + key + " " + str(value))
    
    
    # print(dic_bis == dic)
    # print(dic)
    return dic_bis,list1,list2

# print(tokenize_and_split_bis("SMSSpamCollection"))

def compute_frequencies(num_words, documents):
  """Computes the frequency of words in a corpus of documents.
    
  Args:
    num_words: the number of words that exist. Words will be integers in
        [0..num_words-1].
    documents: a list of lists of integers. Like the l0 or l1 output of
        tokenize_and_split().

  Returns:
    A list of floats of length num_words: element #i will be the ratio
    (in [0..1]) of documents containing i, i.e. the ratio of indices j
    such that "i in documents[j]".
    If index #i doesn't appear in any document, its frequency should be zero.
  """
  res = [0 for i in range(num_words)]
  sum = 0
  for word in documents:
    sum += 1
    tmp = set(word)
    for number in tmp:
      res[number] += 1
  
  res = [i / sum for i in res]
  return res

# print(compute_frequencies(6, [[0, 1, 1], [0, 4, 0]]))


def naive_bayes_train(sms_file):
  """Performs the "training" phase of the Naive Bayes estimator.
    
  Args:
    sms_file: a string, the name of the input SMS data file.

  Returns:
    A triple (spam_ratio, words, spamicity) where:
    - spam_ratio is a float in [0..1] and is the ratio of SMS marked as 'spam'.
    - words is the dictionary output by tokenize_and_split().
    - spamicity is a list of num_words floats, where num_words = len(words) and
      spamicity[i] = (ratio of spams containing word #i) /
                     (ratio of SMS (spams and hams) containing word #i)
  """
  dic, list1, list2 = tokenize_and_split_bis(sms_file)
  nbr_words = len(list1) + len(list2)
  spam_ratio = len(list1) / nbr_words
  document = list1 + list2

  nbr_spam = 0
  for line in list1:
    for word in line:
      nbr_spam += 1
  
  nbr_ham = 0
  for line in list2:
    for word in line:
      nbr_ham += 1
  
  nbr_words = nbr_ham + nbr_spam
  sms_ratio_list = compute_frequencies(nbr_words, document)
  spam_ratio_list = compute_frequencies(nbr_words, list1)
  spamicity = [0. for i in range(nbr_words)]

  # print(nbr_words)

  for i in range(nbr_words):
    if sms_ratio_list[i] != 0:
      spamicity[i] = spam_ratio_list[i] / sms_ratio_list[i]

  return spam_ratio, dic, spamicity
    
# print(naive_bayes_train('test.txt'))

def naive_bayes_train_bis(sms_file):
  """Performs the "training" phase of the Naive Bayes estimator.
    
  Args:
    sms_file: a string, the name of the input SMS data file.

  Returns:
    A triple (spam_ratio, words, spamicity) where:
    - spam_ratio is a float in [0..1] and is the ratio of SMS marked as 'spam'.
    - words is the dictionary output by tokenize_and_split().
    - spamicity is a list of num_words floats, where num_words = len(words) and
      spamicity[i] = (ratio of spams containing word #i) /
                     (ratio of SMS (spams and hams) containing word #i)
    - spamicity_no is the 1-spamicity[i]
    - spamicity_inv is the 1/spamicity[i]
    - product_word_dic is the product of the word in the dict (product spamicity_inv for all the i)
  """
  dic, list1, list2 = tokenize_and_split_bis(sms_file)
  nbr_words = len(list1) + len(list2)
  spam_ratio = len(list1) / nbr_words
  document = list1 + list2

  nbr_spam = 0
  for line in list1:
    for word in line:
      nbr_spam += 1
  
  nbr_ham = 0
  for line in list2:
    for word in line:
      nbr_ham += 1
  
  nbr_words = nbr_ham + nbr_spam
  sms_ratio_list = compute_frequencies(nbr_words, document)
  spam_ratio_list = compute_frequencies(nbr_words, list1)
  spamicity = [0. for i in range(nbr_words)]
  # print(sms_ratio_list)
  # print(spam_ratio_list)
  spamicity_no = [0. for i in range(nbr_words)]
  spamicity_inv = [0. for i in range(nbr_words)]

  product_word_dic = 1
  for i in range(nbr_words):
    if sms_ratio_list[i] != 0:
      spamicity[i] = ((spam_ratio_list[i]) / sms_ratio_list[i])
      spamicity_no[i] = 1 - ((spam_ratio_list[i]) / sms_ratio_list[i])
      spamicity_inv[i] = ((1 - (spam_ratio_list[i])) /  (1 - sms_ratio_list[i]))
      # print(spamicity_inv[i])
      # if spamicity_inv[i] != 0 :
      product_word_dic *= spamicity_inv[i]
      
  return spam_ratio, dic, spamicity, spamicity_no, spamicity_inv, product_word_dic

def naive_bayes_predict(spam_ratio, words, spamicity, sms):
  """Performs the "prediction" phase of the Naive Bayes estimator.

  You should use the simple formula:
  P(spam|words in sms) = spam_ratio * Product[word in sms]( P(word|spam) / P(word) )
  Make sure you skip (i.e. ignore) the SMS words that are unknown (not in 'words').
  BE CAREFUL: if a word is repeated in the sms, it shouldn't appear twice here!
    
  Args:
    spam_ratio: see output of naive_bayes_train
    words: see output of naive_bayes_train
    spamicity: see output of naive_bayes_train
    sms: a string (which you can tokenize to obtain a list of words)

  Returns:
    The estimated probability that the given sms is a spam.
  """
  res = set(sms.split())

  product = 1
  for word in res:
    if word in words:
      heur = spamicity[words[word]]
      product *= heur
  
  is_spam = spam_ratio * product
  # print(is_spam)
  return is_spam

def naive_bayes_predict_bis(spam_ratio, words, spamicity, spamicity_no, spamicity_inv, product_word_dic, sms):
  """Performs the "prediction" phase of the Naive Bayes estimator but in a better way.

  You should use the complex formula:
  P(spam|words in sms) = spam_ratio * (Product[word in sms]( P(word|spam) * (1-p(mot)) / P(word) * 1-p(mot|spam)) * Product[word in dic] (1-p(word|spam)) / 1 - p(word))
  Make sure you skip (i.e. ignore) the SMS words that are unknown (not in 'words').
  BE CAREFUL: if a word is repeated in the sms, it shouldn't appear twice here!
  Some enhancements has been made...
    
  Args:
    spam_ratio: see output of naive_bayes_train_bis
    words: see output of naive_bayes_train_bis
    spamicity: see output of naive_bayes_train_bis
    spamicity_no: see output of naive_bayes_train_bis,
    spamicity_inv: see output of naive_bayes_train_bis
    product_word_dic: see output of naive_bayes_train_bis
    sms: a string (which you can tokenize to obtain a list of words)

  Returns:
    The estimated probability that the given sms is a spam.
  """
  res = set(sms.split())

  product_word_mess = 1
  for word in res:
    if word in words:
      heur = spamicity[words[word]]
      if heur > 0.8 or heur < 0.2:
        if heur == 0:
          heur = 1
        # print(word + " " + str(heur))
        product_word_mess *= ( heur ) * ( 1 / ( spamicity_inv[words[word]] ))
      # product_word_mess *= heur
  # print(product_word_dic)
  is_spam = spam_ratio * product_word_mess * product_word_dic 
  # print(is_spam)
  return is_spam

def naive_bayes_eval(test_sms_file, f):
  """Evaluates a spam classifier.
  
  Args:
    test_sms_file: a string, the name of the input 'test' SMS data file.
    f: a function. f(sms), where sms is a string (like "Hi. Where are you?",
        should return 1 if sms is classified as spam, and 0 otherwise.
  
  Returns:
    A pair of floats (recall, precision): 'recall' is the ratio (in [0,1]) of
    spams in the test file that were successfully identified as spam, and
    'Precision' is the ratio of predicted spams that were actually spam.
  """
  list1 = []
  list2 = []
  spam = False
  i = 0
  for line in open(test_sms_file, 'r').readlines():
    words = line.split()
    spam = (words[0] == "spam")
    if spam :
      list1.append(line[len("spam") + 1:-1:])
    else:
      list2.append(line[len("ham") + 1:-1:])
          
  # print(list1)
  # print(list2)
  wlist1 = []
  wlist2 = []
      
  for word in list1:
    # print( str(f(word)) + " " + word)              
    if f(word):
      # print("f(word)= True\n on ajoute 1 à wlist 1")
      wlist1.append(1)
    else: 
      # print("f(word)= False\n on ajoute 0 à wlist 1")
      wlist1.append(0)
    
  for word in list2:
    # print(str(f(word)) + " " + word)          
    if f(word):
      # print("f(word)= True\n on ajoute 1 à wliste 2")      
      wlist2.append(1)
    else: 
      # print("f(word)= False\n on ajoute 0 à wlist 2")
      wlist2.append(0)
    
  # print(wlist1)
  # print(wlist2)
  res = 0
  for ratio in wlist1:
    res += ratio
  
  recall = res / len(wlist1)
  res2 = 0
  for ratio in wlist2:
    res2 += ratio
  
  if res + res2 == 0:
    res = 1
    
  precision = res / (res + res2)
  
  return (recall,precision)

def classify_spam(sms):
  """Returns True is the message 'sms' is predicted to be a spam."""
  return naive_bayes_predict(spam_ratio, words, spamicity, sms) > seuil

def classify_spam_precision(sms):
  """Like classify_spam(), but guaranteed to have precision > 0.9."""
  # return naive_bayes_predict(spam_ratio, words, spamicity, sms) >= seuil
  # print(naive_bayes_predict_bis(spam_ratio_bis, words_bis, spamicity_bis,spamicity_no, spamicity_inv, product_word_dic, sms)  > seuil_precision)
  return naive_bayes_predict_bis(spam_ratio_bis, words_bis, spamicity_bis,spamicity_no, spamicity_inv, product_word_dic, sms)  > seuil_precision

def classify_spam_recall(sms):
  """Like classify_spam(), but guaranteed to have recall > 0.9."""
  # return naive_bayes_predict(spam_ratio, words, spamicity, sms) >= seuil
  return naive_bayes_predict_bis(spam_ratio_bis, words_bis, spamicity_bis,spamicity_no, spamicity_inv, product_word_dic, sms) > seuil_recall


split_lines("SMSSpamCollection", seed, "train.txt", "test.txt")
print("environment ready")
spam_ratio, words, spamicity = naive_bayes_train("train.txt")
print("program trained with spam")
# print(naive_bayes_eval("test.txt", classify_spam))
spam_ratio_bis, words_bis, spamicity_bis, spamicity_no, spamicity_inv, product_word_dic = naive_bayes_train_bis("train.txt")
print("program trained with spam_bis")

# Found the best seuil

max_precision_for_recall, ind_rec = 0,0
max_recall_for_precision, ind_prec = 0,0

for i in range(1, 11):
  seuil_precision_ = i / 10
  seuil_precision = seuil_precision_
  recall, precision = naive_bayes_eval("test.txt", classify_spam_precision)
  # print(str(recall) + " " + str(precision))
  if recall >= 0.9:
    if max_precision_for_recall < precision:
      max_precision_for_recall, ind_rec = precision, seuil_precision_
  
  if precision >= 0.9 :
    if max_recall_for_precision < recall:
      max_recall_for_precision, ind_prec = recall, seuil_precision_

print(str(max_precision_for_recall)+ " " + str(ind_rec))
print(str(max_recall_for_precision)+ " " + str(ind_prec))
  # return ind_rec, ind_prec

seuil_recall, seuil_precision = ind_rec, ind_prec
# print("nice seuil defined " + str(seuil_recall) + " for recall and "+ str(seuil_precision) + " for precision")

# print("better recall is : ")
# print(naive_bayes_eval("test.txt", classify_spam_recall))
# print("better precision is : ")
# print(naive_bayes_eval("test.txt", classify_spam_precision))
