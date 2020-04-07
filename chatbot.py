"""
Created on Fri Mar 27 14:42:40 2020
@author: Masum Ahmed EeSha
"""

import numpy as np
import tensorflow as tf
import re 
import time

# tf.disable_v2_behavior() 

# DATA PRE-PROCESSING

lines = open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

# print(lines)

lineDic = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        lineDic[_line[0]]=_line[4]

conversationId=[]
for conv in conversations[:-1]:
    _con = conv.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversationId.append(_con.split(','))

# print(conversationId[0])
questions = []
answers = []
for convLine in conversationId:
    for i in range(len(convLine)-1):
        questions.append(lineDic[convLine[i]])
        answers.append(lineDic[convLine[i+1]])

def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"didn't", "did not", text)
    text = re.sub(r"doesn't", "does not", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text

clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))
    
wordsCountQuestion = {}
for c_q in clean_questions:
    for word in c_q.split():
        if word not in wordsCountQuestion:
            wordsCountQuestion[word]=1
        else:
            wordsCountQuestion[word]+=1

wordsCountAnswer = {}
for c_a in clean_answers:
    for word in c_a.split():
        #if word not in wordsCountAnswer:
        if word not in wordsCountQuestion:
            #wordsCountAnswer[word]=1
            wordsCountQuestion[word]=1
        else:
            #wordsCountAnswer[word]+=1
            wordsCountQuestion[word]+=1
        
threshold = 20
questionsWordsInteger = {}
wordNumber = 0
for word, count in wordsCountQuestion.items():
    if count >= threshold:
        questionsWordsInteger[word] = wordNumber
        wordNumber+=1

answersWordsInteger = {}
wordNumber = 0
#for word, count in wordsCountAnswer.items():
for word, count in wordsCountQuestion.items():
    if count >= threshold:
        answersWordsInteger[word] = wordNumber
        wordNumber+=1

tokens = ['<PAD>','<EOS>','<OUT>','<SOS>']
for token in tokens:
    questionsWordsInteger[token] = len(questionsWordsInteger) + 1
for token in tokens:
    answersWordsInteger[token] = len(answersWordsInteger) + 1

answerIntInverseMap = {w_i : w for w, w_i in answersWordsInteger.items()}

for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'

questionsConvertToInt = []
for question in clean_questions:
    temp = []
    for word in question.split():
        if word not in questionsWordsInteger:
            temp.append(questionsWordsInteger['<OUT>'])
        else:
            temp.append(questionsWordsInteger[word])
    questionsConvertToInt.append(temp)

answersConvertToInt = []
for answer in clean_answers:
    temp = []
    for word in answer.split():
        if word not in answersWordsInteger:
            temp.append(answersWordsInteger['<OUT>'])
        else:
            temp.append(answersWordsInteger[word])
    answersConvertToInt.append(temp)

sortedQuestions = []
sortedAnswers = []
for length in range(1, 25+1):
    for i in enumerate(questionsConvertToInt):
        if len(i[1]) == length:
            sortedQuestions.append(questionsConvertToInt[i[0]])
            sortedAnswers.append(answersConvertToInt[i[0]])
        


# BUILDING MODEL

def modelInputs():
    inputs = tf.placeholder(tf.int32,[None,None], name='input')
    targets = tf.placeholder(tf.int32,[None,None], name='target')
    learningRate = tf.placeholder(tf.float32, name='learningRate')
    keepProb = tf.placeholder(tf.float32, name='keepProb')
    return inputs,targets,learningRate,keepProb

def preprocessTargets(targets, wordConvertToInt_UniqueIndentifier, batchSize):
    leftSide = tf.fill([batchSize,1], wordConvertToInt_UniqueIndentifier['<SOS>'])
    rightSide = tf.strided_slice(targets, [0,0], [batchSize,-1], [1,1])
    preprocessedTarget = tf.concat([leftSide, rightSide], 1)
    return preprocessedTarget

def encoderRnn(rnnInputs, rnnSize, numberOfLayers, keepProb, sequenceLength):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnnSize)
    lstmDropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keepProb)
    encoderCell = tf.contrib.rnn.MultiRNNCell([lstmDropout] * numberOfLayers)
    encoderOutput, encoderState = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoderCell,
                                                      cell_bw = encoderCell,
                                                      sequence_length = sequenceLength,
                                                      inputs = rnnInputs,
                                                      dtype = tf.float32)
    return encoderState


def decodeTrainingSet(encoderState, decoderCell, decoderEmbeddedInput, sequenceLength, 
                      decodingScope, outputFunction, keepProb, batchSize):
    attentionStates = tf.zeros([batchSize, 1, decoderCell.output_size])
    attentionKeys, attentionValues, attentionScoreFunction, attentionConstructFunction = tf.contrib.seq2seq.prepare_attention(
        attentionStates, 
        attention_option = "bahdanau",
        num_units = decoderCell.output_size)
    trainingDecoderFunction = tf.contrib.seq2seq.attention_decoder_fn_train(encoderState[0],
                                                                            attentionKeys,
                                                                            attentionValues,
                                                                            attentionScoreFunction,
                                                                            attentionConstructFunction,
                                                                            name = "attn_dec_train")
    decoderOutput, decoderFinalState, decoderFinalContextState = tf.contrib.seq2seq.dynamic_rnn_decoder(decoderCell,
                                                                                                        trainingDecoderFunction,
                                                                                                        decoderEmbeddedInput,
                                                                                                        sequenceLength,
                                                                                                        scope = decodingScope)
    decoderOutputDropout = tf.nn.dropout(decoderOutput, keepProb)
    return outputFunction(decoderOutputDropout)


# Test/Validation Set
def decodeTestSet(encoderState, decoderCell, decoderEmbeddingsMatrix, sosId, eosId, maximumLength,
                   numberOfWords, decodingScope, outputFunction, keepProb, batchSize):
    attentionStates = tf.zeros([batchSize, 1, decoderCell.output_size])
    attentionKeys, attentionValues, attentionScoreFunction, attentionConstructFunction = tf.contrib.seq2seq.prepare_attention(
        attentionStates, 
        attention_option = "bahdanau",
        num_units = decoderCell.output_size)
    testDecoderFunction = tf.contrib.seq2seq.attention_decoder_fn_inference(outputFunction,
                                                                            encoderState[0],
                                                                            attentionKeys,
                                                                            attentionValues,
                                                                            attentionScoreFunction,
                                                                            attentionConstructFunction,
                                                                            decoderEmbeddingsMatrix, 
                                                                            sosId, 
                                                                            eosId, 
                                                                            maximumLength, 
                                                                            numberOfWords,
                                                                            name = "attn_dec_inf")
    testPredictions, decoderFinalState, decoderFinalContextState = tf.contrib.seq2seq.dynamic_rnn_decoder(decoderCell,
                                                                                                        testDecoderFunction,
                                                                                                        scope = decodingScope)
    return testPredictions


def decoderRnn(decoderEmbeddedInput, decoderEmbeddingsMatrix, encoderState, numberOfWords, sequenceLength, 
               rnnSize, numberOfLayers, wordConvertToInt_UniqueIndentifier, keepProb, batchSize):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnnSize)
        lstmDropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keepProb)
        decoderCell = tf.contrib.rnn.MultiRNNCell([lstmDropout] * numberOfLayers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        outputFunction = lambda x: tf.contrib.layers.fully_connected(x,
                                                                     numberOfWords,
                                                                     None,
                                                                     scope = decoding_scope,
                                                                     weights_initializer = weights,
                                                                     biases_initializer = biases)
        trainingPredictions = decodeTrainingSet(encoderState, 
                                                decoderCell, 
                                                decoderEmbeddedInput, 
                                                sequenceLength, 
                                                decoding_scope, 
                                                outputFunction, 
                                                keepProb, 
                                                batchSize)
        decoding_scope.reuse_variables()
        testPredictions = decodeTestSet(encoderState, 
                                        decoderCell, 
                                        decoderEmbeddingsMatrix, 
                                        wordConvertToInt_UniqueIndentifier['<SOS>'], 
                                        wordConvertToInt_UniqueIndentifier['<EOS>'], 
                                        sequenceLength - 1, 
                                        numberOfWords, 
                                        decoding_scope, 
                                        outputFunction, 
                                        keepProb, 
                                        batchSize)
        
    return trainingPredictions, testPredictions


def seq2seqModel(inputs, targets, keepProb, batchSize, sequenceLength, answersNumberOfWords, questionsNumberOfWords,
                 encoderEmbeddingSize, decoderEmbeddingSize, rnnSize, numberOfLayers, questionsConvertToInt):
    encoderEmbeddedInput = tf.contrib.layers.embed_sequence(inputs,
                                                            answersNumberOfWords + 1,
                                                            encoderEmbeddingSize,
                                                            initializer = tf.random_uniform_initializer(0,1))
    encoderState = encoderRnn(encoderEmbeddedInput, 
                              rnnSize, 
                              numberOfLayers, 
                              keepProb, 
                              sequenceLength)
    preprocessedTarget = preprocessTargets(targets, 
                                           questionsConvertToInt, 
                                           batchSize)
    decoderEmbeddingsMatrix = tf.Variable(tf.random_uniform([questionsNumberOfWords + 1, decoderEmbeddingSize], 0, 1))
    decoderEmbeddedInput = tf.nn.embedding_lookup(decoderEmbeddingsMatrix, preprocessedTarget)
    trainingPredictions, testPredictions = decoderRnn(decoderEmbeddedInput, 
                                                      decoderEmbeddingsMatrix, 
                                                      encoderState, 
                                                      questionsNumberOfWords, 
                                                      sequenceLength, 
                                                      rnnSize, 
                                                      numberOfLayers, 
                                                      questionsConvertToInt, 
                                                      keepProb, 
                                                      batchSize)
    return trainingPredictions, testPredictions

    
       
    
# Training

# Hyper Parameters
epochs = 100
batchSize = 64
rnnSize = 512
numberOfLayers = 3
encoderEmbeddingSize = 512
decoderEmbeddingSize = 512
learningRate = 0.01
learningRateDecay = 0.9
minLearningRate = 0.0001
keepProbability = 0.5

# Session
tf.reset_default_graph()
session = tf.InteractiveSession()

inputs, targets, lr, keepProb = modelInputs()

sequenceLength = tf.placeholder_with_default(25, None, name='sequenceLength')

inputShape = tf.shape(inputs)

# training and test predictions 
trainingPredictions, testPredictions = seq2seqModel(tf.reverse(inputs,[-1]), 
                                                    targets, 
                                                    keepProb, 
                                                    batchSize, 
                                                    sequenceLength, 
                                                    len(answersWordsInteger), 
                                                    len(questionsWordsInteger), 
                                                    encoderEmbeddingSize, 
                                                    decoderEmbeddingSize, 
                                                    rnnSize, 
                                                    numberOfLayers, 
                                                    questionsWordsInteger)
# Loss Error, Optimizer and Gradient Clipping
with tf.name_scope("optimization"):
    lossError = tf.contrib.seq2seq.sequence_loss(trainingPredictions,
                                                 targets,
                                                 tf.ones([inputShape[0], sequenceLength]))
    optimizer = tf.train.AdamOptimizer(learningRate)
    gradients = optimizer.compute_gradients(lossError)
    clippedGradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizerGradientClipping = optimizer.apply_gradients(clippedGradients)

# Padding with <PAD>
def applyPadding(batchOfSequences, wordConvertToInt):
    maxSequenceLength = max([len(sequence) for sequence in batchOfSequences])
    return [sequence + [wordConvertToInt['<PAD>']] * (maxSequenceLength - len(sequence)) for sequence in batchOfSequences]

# Splitting data into batches 
def splitIntoBatches(questions, answers, batchSize):
    for batchIndex in range(0, len(questions) // batchSize):
        startIndex = batchIndex * batchSize
        questionsInBatch = questions[startIndex : startIndex + batchSize]
        answersInBatch = answers[startIndex : startIndex + batchSize]
        paddedQuestionsInBatch = np.array(applyPadding(questionsInBatch, 
                                                       questionsWordsInteger))
        paddedAnswersInBatch = np.array(applyPadding(answersInBatch,
                                                     answersWordsInteger))
        yield paddedQuestionsInBatch, paddedAnswersInBatch

# Splitting into training and Validation sets
trainingValidationSplit = int(len(sortedQuestions) * 0.15)
trainingQuestions = sortedQuestions[trainingValidationSplit:]
trainingAnswers = sortedAnswers[trainingValidationSplit:]
validationQuestions = sortedQuestions[:trainingValidationSplit]
validationAnswers = sortedAnswers[:trainingValidationSplit]

# Training
batchIndexCheckTrainingLoss = 100
batchIndexCheckValidationLoss = ((len(trainingQuestions)) // batchSize // 2) - 1
totalTrainingLossError = 0
listValidationLossError = []
earlyStoppingCheck = 0
earlyStoppingStop = 1000
checkpoint = "chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs + 1):
    for batchIndex, (paddedQuestionsInBatch, paddedAnswersInBatch) in enumerate(splitIntoBatches(trainingQuestions, trainingAnswers, batchSize)):
        startingTime = time.time()
        _, batchTrainingLossError = session.run([optimizerGradientClipping, lossError], {inputs: paddedQuestionsInBatch,
                                                                                               targets: paddedAnswersInBatch,
                                                                                               lr: learningRate,
                                                                                               sequenceLength: paddedAnswersInBatch.shape[1],
                                                                                               keepProb: keepProbability})
        totalTrainingLossError += batchTrainingLossError
        endingTime = time.time()
        batchTime = endingTime - startingTime
        if batchIndex % batchIndexCheckTrainingLoss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                                                                                       epochs,
                                                                                                                                       batchIndex,
                                                                                                                                       len(trainingQuestions) // batchSize,
                                                                                                                                       totalTrainingLossError / batchIndexCheckTrainingLoss,
                                                                                                                                       int(batchTime * batchIndexCheckTrainingLoss)))
            totalTrainingLossError = 0
        if batchIndex % batchIndexCheckValidationLoss == 0 and batchIndex > 0:
            totalValidationLossError = 0
            startingTime = time.time()
            for batchIndexValidation, (paddedQuestionsInBatch, paddedAnswersInBatch) in enumerate(splitIntoBatches(validationQuestions, validationAnswers, batchSize)):
                batchValidationLossError = session.run(lossError, {inputs: paddedQuestionsInBatch,
                                                                       targets: paddedAnswersInBatch,
                                                                       lr: learningRate,
                                                                       sequenceLength: paddedAnswersInBatch.shape[1],
                                                                       keepProb: 1})
                totalValidationLossError += batchValidationLossError
            endingTime = time.time()
            batchTime = endingTime - startingTime
            averageValidationLossError = totalValidationLossError / (len(validationQuestions) / batchSize)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(averageValidationLossError, int(batchTime)))
            learningRate *= learningRateDecay
            if learningRate < minLearningRate:
                learningRate = minLearningRate
            listValidationLossError.append(averageValidationLossError)
            if averageValidationLossError <= min(listValidationLossError):
                print('I speak better now!!')
                earlyStoppingCheck = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry I do not speak better, I need to practice more.")
                earlyStoppingCheck += 1
                if earlyStoppingCheck == earlyStoppingStop:
                    break
    if earlyStoppingCheck == earlyStoppingStop:
        print("My apologies, I cannot speak better anymore. This is the best I can do.")
        break
print("Game Over")

# Loading weights
checkpoint = "./chatbot_weights.ckpt"
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session, checkpoint)

# Converting Question to integer
def convertQtoInt(question, wordConvertInt):
    question = clean_text(question)
    return [wordConvertInt.get(word, wordConvertInt['<OUT>']) for word in question.split()]

# CHATBOT
while(True):
    question = input("Ask Question: ")
    if question == 'Goodbye':
        break
    question = convertQtoInt(question, questionsWordsInteger)
    question = question + [questionsWordsInteger['<PAD>']] * (25 - len(question))
    fakeBatch = np.zeros((batchSize, 25))
    fakeBatch[0] = question
    
    predictedAnswer = session.run(testPredictions, {inputs: fakeBatch,
                                                    keepProb: 0.5})[0]
    print(predictedAnswer)
    answer = ''
    
    for i in np.argmax(predictedAnswer, 1):
        if answersWordsInteger[i] == 'i':
            token = ' I'
        elif answersWordsInteger[i] == '<EOS>':
            token = '.'
        elif answersWordsInteger[i] == '<OUT>':
            token = 'out'
        else:
            token = ' ' + answersWordsInteger[i]
        answer += token
        if token == '.':
            break
    print('Chatbot: ' + answer)
    




