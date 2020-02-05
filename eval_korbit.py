import torch
import pandas as pd
from tqdm import tqdm
from pre_trained_model.evaluate_stored_model import print_evaluation_result
from model import BaseSequenceLabelingSplitImpExp

def load_pretrained_model():
    parameters = {
        'nb_epoch': 50,
        'sentence_embedding_type': 'max',
        'sentence_zero_inithidden': False,
        'num_layers': 1,
        'batch_size': 128,
        'hidden_size': 300,
        'optimizer_type': 'adam',
        'dropout': 0.5,
        'weight_decay': 0
    }
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
    pretrained_weights = torch.load('pre_trained_model/result/model/pdtb_implicit_moreexplicit_discourse_withoutAltLex_paragraph_multilabel_addposnerembedding_BiLSTMCRFSplitImpExp_rand_viterbi_eachiterationmodel_hidden300_addoutputdropout_exp2.pt')
    model.load_state_dict(pretrained_weights)
    return model.cuda()


if __name__ == "__main__":
    # Load Korbit data
    test = pd.read_pickle("data/korbit/std_test27_proc.pkl")

    # Load pretrained model
    model = load_pretrained_model()

    label_map = {0: "sequence", 1: "comparison", 2: "cause", 3: "elaboration/attribution"}
    predictions = []
    pred_labels = []
    targets = test['target'].map(lambda x : x.argmin()).tolist()
    with torch.no_grad():
        for index, row in tqdm(test.iterrows(), total=len(test.index)):
            input_vecs = row['para_embedding'].cuda()
            target = row['target'].cuda()
            eos = row['eos']

            pred = model(input_vecs, eos, target)
            pred_label = pred.argmax().item()
            predictions.append(pred_label)
            pred_labels.append(label_map[pred_label])

    print_evaluation_result((predictions, targets))
