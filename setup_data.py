import torch
import pandas as pd
import gensim
import nltk
from nltk import word_tokenize
from nltk.tag import StanfordPOSTagger,StanfordNERTagger
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def process_dataset(test_file):
    logging.info("Loading data.")
    raw_df = pd.read_csv("data/korbit/{}.csv".format(test_file))

    # Process input samples. Input size: 1 x num_words x word_embedding_dimension (343)
    logging.info("Processing input samples.")
    raw_df['para_embedding'] = raw_df['edus'].map(process_sample)

    # EOS: list containing length of edus for this sample
    logging.info("Processing EOS symbols.")
    raw_df['eos'] = raw_df['edus'].map(lambda edus : [x for x in accumulate([word_tokenize(edu) for edu in edus])])

    # Target size: len(discourse_list) x 4 -- one-hot encoding
    logging.info("Processing relation targets.")
    raw_df['target'] = raw_df['relations'].map(process_target)
    raw_df.to_pickle("data/korbit/{}_proc.pkl".format(test_file))


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
    target_indices = torch.tensor([label_map[target] for target in discourse_list]).view(len(discourse_list), -1)
    y = torch.zeros(len(discourse_list), 4).scatter_(dim=1, index=target_indices, value=-1.0)
    return y


def accumulate(iterator):
    # Needs to take in a list of list of words
    total = 0
    for item in iterator:
        total += len(item)
        yield total

####################################################
### Hacky methods from the original script below ###
####################################################
model = gensim.models.KeyedVectors.load_word2vec_format('data/resource/GoogleNews-vectors-negative300.bin', binary=True)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default='relations_test_new')
    args = parser.parse_args()
    process_dataset(args.test_file)
