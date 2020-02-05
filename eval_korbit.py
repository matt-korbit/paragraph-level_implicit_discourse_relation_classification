from setup_data import load_pretrained_model
import pandas as pd
from tqdm import tqdm


def compute_scores(targets, predictions):
    pass

if __name__ == "__main__":
    # Load Korbit data
    test = pd.read_pickle("data/korbit/std_test27_proc.pkl")

    # Load pretrained model
    model = load_pretrained_model()

    label_map = {0: "sequence", 1: "comparison", 2: "cause", 3: "elaboration/attribution"}

    predictions = []
    pred_labels = []
    targets = test['target'].map(lambda x : x.numpy()).tolist()
    for index, row in tqdm(test.iterrows(), total=len(test.index)):
        input_vecs = row['para_embedding'].cuda()
        target = row['target'].cuda()
        eos = row['eos']

        pred = model(input_vecs, eos, target)
        predictions.append(pred.item())
        pred_labels.append(label_map[pred.argmax()])

    compute_scores(targets, predictions)