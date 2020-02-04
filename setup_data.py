from model import BaseSequenceLabelingSplitImpExp
import torch
import pandas as pd
import gensim
import nltk
from nltk import word_tokenize
from nltk.tag import StanfordPOSTagger,StanfordNERTagger


def process_dataset():
    raw_df = pd.read_pickle("data/korbit/std_test27.pkl")

    # Process input samples. Input size: 1 x num_words x word_embedding_dimension (343)
    raw_df['para_embedding'] = raw_df['edus'].map(process_sample)

    # EOS: list containing length of edus for this sample
    raw_df['eos'] = raw_df['edus'].map(lambda edus : [x for x in accumulate([word_tokenize(edu) for edu in edus])])

    # Target size: len(discourse_list) x 4 -- one-hot encoding
    raw_df['target'] = raw_df['relations'].map(process_target)
    raw_df.to_pickle("data/korbit/std_test27_proc.pkl")


def process_sample(edus):
    std_response = " ".join(edus)
    sents = nltk.sent_tokenize(std_response)
    word_embed_list = []
    for sent in sents:
        word_embeds = process_sentence(sent)
        word_embed_list.append(word_embeds)
    para_embedding = torch.cat(word_embed_list)
    para_embedding = para_embedding.view(1, -1, para_embedding.size(-1))
    return para_embedding


def process_target(discourse_list):
    """Process the raw discourse target list into a one-hot encoding matrix."""
    label_map = {"sequence": 0, "comparison": 1, "cause": 2, "elaboration": 3, "attribution": 3}
    target_indices = torch.tensor([label_map[target] for target in discourse_list]).view(len(discourse_list), -1.0)
    y = torch.zeros(len(discourse_list), 4).scatter_(dim=1, index=target_indices, value=1.0)
    return y


def load_pretrained_model():
    batch_size_list = [128]  # fixed 128 > 64 > 256
    hidden_size_list = [300]  # 600>300>100
    dropout_list = [5]
    l2_reg_list = [0]  # fixed 0
    nb_epoch_list = [50]
    encoder_sentence_embedding_type_list = ['max']  # max > mean > last
    sentence_zero_inithidden_list = [False]
    optimizer_type_list = ['adam']  # adam > adagrad > other
    num_layers_list = [1]
    parameters_list = []
    for num_layers in num_layers_list:
        for sentence_embedding_type in encoder_sentence_embedding_type_list:
            for sentence_zero_inithidden in sentence_zero_inithidden_list:
                for batch_size in batch_size_list:
                    for optimizer_type in optimizer_type_list:
                        for hidden_size in hidden_size_list:
                            for nb_epoch in nb_epoch_list:
                                for weight_decay in l2_reg_list:
                                    for dropout in dropout_list:
                                        parameters = {}
                                        parameters['nb_epoch'] = nb_epoch
                                        parameters['sentence_embedding_type'] = sentence_embedding_type
                                        parameters['sentence_zero_inithidden'] = sentence_zero_inithidden
                                        parameters['num_layers'] = num_layers
                                        parameters['batch_size'] = batch_size
                                        parameters['hidden_size'] = hidden_size
                                        parameters['optimizer_type'] = optimizer_type
                                        parameters['dropout'] = dropout * 0.1
                                        parameters['weight_decay'] = weight_decay
                                        parameters_list.append(parameters)

    word_embedding_dimension = 343
    number_class = 4

    # Model
    model = BaseSequenceLabelingSplitImpExp(word_embedding_dimension, number_class,
                                            hidden_size=parameters['hidden_size'],
                                            sentence_embedding_type=parameters['sentence_embedding_type'],
                                            sentence_zero_inithidden=parameters['sentence_zero_inithidden'],
                                            cross_attention=False, attention_function='feedforward', NTN_flag=False,
                                            num_layers=parameters['num_layers'], dropout=parameters['dropout'])

    # Load weights
    pretrained_weights = torch.load('pre-trained_model/result/model/pdtb_implicit_moreexplicit_discourse_withoutAltLex_paragraph_multilabel_addposnerembedding_BiLSTMCRFSplitImpExp_rand_viterbi_eachiterationmodel_hidden300_addoutputdropout_exp2.pt')
    model.load_state_dict(pretrained_weights)

    return model.cuda()


def accumulate(iterator):
    # Needs to take in a list of list of words
    total = 0
    for item in iterator:
        total += len(item)
        yield total

####################################################
### Hacky methods from the original script below ###
####################################################
model = gensim.models.KeyedVectors.load_word2vec_format('../resource/GoogleNews-vectors-negative300.bin', binary=True)
stanford_dir = 'data/resource/stanford-postagger-2018-10-16/'
modelfile = stanford_dir + 'models/english-left3words-distsim.tagger'
jarfile = stanford_dir + 'stanford-postagger.jar'
pos_tager = StanfordPOSTagger(modelfile, jarfile, encoding='utf8')

stanford_dir = 'data/resource/stanford-ner-2018-10-16/'
modelfile = stanford_dir + 'classifiers/english.muc.7class.distsim.crf.ser.gz'
jarfile = stanford_dir + 'stanford-ner.jar'
ner_tager = StanfordNERTagger(modelfile, jarfile, encoding='utf8')
def process_sentence(sentence, posner_flag=True, sentencemarker=False, paramarker=False):
    if posner_flag:
        word_list = nltk.word_tokenize(sentence)
        pos_list = pos_tager.tag(word_list)
        ner_list = ner_tager.tag(word_list)

        if sentencemarker:
            pos_list.insert(0, ('<SOS>', ''))
            ner_list.insert(0, ('<SOS>', ''))

            pos_list.append(('<EOS>', ''))
            ner_list.append(('<EOS>', ''))

        if paramarker:
            pos_list.insert(0, ('<ParaBoundary>', ''))
            ner_list.insert(0, ('<ParaBoundary>', ''))

        return tansfer_word2vec((pos_list, ner_list), posner_flag=True)
    else:
        word_list = nltk.word_tokenize(sentence)
        # word_list = st.tokenize(sentence)

        if sentencemarker:
            word_list.insert(0, '<SOS>')
            word_list.append('<EOS>')

        if paramarker:
            word_list.insert(0, '<ParaBoundary>')

        return tansfer_word2vec(word_list, posner_flag=False)


vocab={}
def unknown_words(word,k=300):
    if word == '' or word in ['<SOS>','<EOS>','<ParaBoundary>']:
        return torch.zeros(k)
    if word not in vocab:
        vocab[word] = torch.rand(k)/2 - 0.25
    return vocab[word]

NER_LIST = ['ORGANIZATION','LOCATION','PERSON','MONEY','PERCENT','DATE','TIME']
PEN_TREEBANK_POS_LIST = ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB']
def tansfer_word2vec(input_list,posner_flag=True,k=300):
    if posner_flag:
        pos_list,ner_list = input_list[0],input_list[1]
        embedding = torch.zeros(len(pos_list),k+len(PEN_TREEBANK_POS_LIST)+len(NER_LIST))

        for i in range(len(pos_list)):
            word,pos,ner = pos_list[i][0],pos_list[i][1],ner_list[i][1]

            if word in model:
                embedding[i,:k] = torch.from_numpy(model[word])
            #elif word.lower() in model:
            #	embedding[i,:k] = torch.from_numpy(model[word.lower()])
            else:
                embedding[i,:k] = unknown_words(word)

            if pos in PEN_TREEBANK_POS_LIST:
                embedding[i,k+PEN_TREEBANK_POS_LIST.index(pos)] = 1
            if ner in NER_LIST:
                embedding[i,k+len(PEN_TREEBANK_POS_LIST)+NER_LIST.index(ner)] = 1

        return embedding
    else:
        word_list = input_list
        embedding = torch.zeros(len(word_list),k)
        for i in range(len(word_list)):
            word = word_list[i]

            if word in model:
                embedding[i,:] = torch.from_numpy(model[word])
            #elif word.lower() in model:
            #	embedding[i,:] = torch.from_numpy(model[word.lower()])
            else:
                embedding[i,:] = unknown_words(word)
        return embedding


if __name__ == "__main__":
    process_dataset()
