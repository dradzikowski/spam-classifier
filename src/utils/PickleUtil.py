import os
import pickle

PICKLES_DIR = os.path.join('../pickles')


def save_pickle(pipeline, score):
    if score > 0.9935:
        with open(os.path.join(PICKLES_DIR, str(score)) + '.pkl', 'wb') as fid:
            pickle.dump(pipeline, fid)


def read_pickle():
    with open(os.path.join(PICKLES_DIR, '0.993612981184.pkl'), 'rb') as fid:
        loaded_pickle = pickle.load(fid)
    return loaded_pickle
