from setup_data import load_pretrained_model, load_korbit_test_set
import pandas as pd

if __name__ == "__main__":
    # Load Korbit data
    test = pd.read_pickle("data/korbit/std_test27.pkl")

    # Load pretrained model
    model = load_pretrained_model()


