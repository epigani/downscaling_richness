# FILEPATH: /Users/epigani/Library/CloudStorage/GoogleDrive-emanuele.pigani.1@unipd.it/.shortcut-targets-by-id/17Sn-Ra2REh5B86l96gE_Smchh22WCsgo/PhD-Emanuele Pigani/finished_projects/Pigani HayMele et al 2023/downscaling_richness/code/synthetic_sampling.py
import pickle
import numpy as np
from tqdm import tqdm

# generate random numbers from power law distribution with inverse transform sampling

def main():
    alpha = 0.5
    S = 10000
    species = range(S)

    u = 1-np.random.uniform(size=S)
    x = np.array((1-u/alpha)**(-1./(alpha)), dtype=int)
    N = np.sum(x)

    # subsamples of size n = p*N
    ps = np.logspace(-7, -1, 7)
    samples = [None] * len(ps)

    for p in ps:
            Np = int(N * p)
            subsample = np.zeros(Np, dtype=int)
            probabilities = np.copy(x)
            for i in tqdm(range(Np)):
                species_extracted = np.random.choice(species, size=1, p=probabilities/probabilities.sum())[0]
                subsample[i] = species_extracted
                probabilities[species_extracted] -= 1
                
            _, counts = np.unique(subsample, return_counts=True)
            samples.append(counts)
            
    samples_dict = dict(zip(ps, samples))
    with open('../data/samples_dict.pkl', 'wb') as f:
        pickle.dump(samples_dict, f)
        
if __name__ == '__main__':
    main()