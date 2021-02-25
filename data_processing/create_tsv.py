import random

def hube_to_tsv():
    with open('../../data/processed/neutral_and_biased.tsv','w') as outfile:
        with open('../../data/raw/Hube2019/statements_neutral_featured','r') as neutralfile:
            for line in neutralfile:
                outfile.write('0\t') # Label
                outfile.write(line)
            outfile.write('\n')
        with open('../../data/raw/Hube2019/statements_biased','r') as biasedfile:
            for line in biasedfile:
                outfile.write('1\t') # Label
                outfile.write(line)

# Creates a balanced version of the Hube dataset (1843 positive and 1843 negative samples)
def hube_to_tsv_balanced():
    random.seed(0) # Always select the same samples
    with open('../../data/processed/hube_balanced.tsv','w') as outfile:
        with open('../../data/raw/Hube2019/statements_neutral_featured','r') as neutralfile:
            lines = neutralfile.readlines()
            chosen_lines = random.sample(lines, 1843) # There are 1843 biased samples
            for line in chosen_lines:
                outfile.write('0\t') # Label
                outfile.write(line)
        with open('../../data/raw/Hube2019/statements_biased','r') as biasedfile:
            for line in biasedfile:
                outfile.write('1\t') # Label
                outfile.write(line)

def read_test():
    with open('../../data/processed/neutral_and_biased.tsv','r') as infile:
        counter = 0
        for line in infile:
            print(line)
            counter += 1
            if counter > 5:
                break


def pryzant_to_tsv():
    """
    Output tsv file has 3 columns: Revision-ID (not unique); Label (1=biased, 0=unbiased); Text
    
    Actual inpath (after downloading data): .../raw/Pryzant2019/bias_data/WNC/biased.full
    """
    outpath = '../../data/processed/pryzant2019_full.tsv'
    inpath = '../../data/raw/Pryzant2019/biased.full'
    with open(outpath,'w') as outfile:
        with open(inpath,'r') as infile:
            for line in infile:
                splits = line.split('\t')
                # Each infile-line contains the biased AND the unbiased version 
                # => 2 samples for outfile (one positive, one negative)
                # Construct 1st string (Biased)
                positive_sample = splits[0] + '\t' + '1' + '\t' + splits[3] + '\n'
                # Construct 1st string (Unbiased)
                negative_sample = splits[0] + '\t' + '0' + '\t' + splits[4] + '\n'
                
                outfile.write(positive_sample)
                outfile.write(negative_sample)

def diffcrawl_to_tsv():
    """
    Similar to pryzant_to_tsv() above.
    Uses the combined diffs (output of combine_crawls.py)
    """
    outpath = '../../data/processed/crawled_diffs.tsv'
    inpath = '../../data/processed/combined_diffs'
    with open(outpath,'w') as outfile:
        with open(inpath,'r') as infile:
            for line in infile:
                splits = line.split('\t')
                # Each infile-line contains the biased AND the unbiased version 
                # => 2 samples for outfile (one positive, one negative)
                # Construct 1st string (Biased)
                positive_sample = splits[0] + '\t' + '1' + '\t' + splits[1] + '\n'
                # Construct 1st string (Unbiased)
                negative_sample = splits[0] + '\t' + '0' + '\t' + splits[2]
                
                outfile.write(positive_sample)
                outfile.write(negative_sample)





if __name__ == "__main__":
    #pryzant_to_tsv()
    #hube_to_tsv_balanced()
    diffcrawl_to_tsv()

