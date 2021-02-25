import os
import re

def combine_crawls():
    OUTFILE_PATH = "../../data/processed/combined_diffs"
    with open(OUTFILE_PATH,'w') as outfile:
        CRAWL_DIR = "../../data/processed/crawl/"
        for diff_file in os.listdir(CRAWL_DIR):
            if diff_file.endswith(".diffs"):
                diffs = []
                with open(CRAWL_DIR + diff_file,'r') as infile:
                    next_line_flag = False
                    for line in infile:
                        if re.match(r'\d+',line): # Starts with rev_id
                            diffs.append(line)
                            next_line_flag = True
                        elif re.match(r'\t',line) and next_line_flag: # Starts with tab (belongs to pevious diff)
                            diffs[-1] = diffs[-1] + line
                            next_line_flag = False
                        else:
                            next_line_flag = False
                
                for diff in diffs:
                    diff = re.sub('\n',' ',diff)
                    diff = diff.strip()
                    diff = re.sub('REDIRECT ','',diff)
                    splits = diff.split('\t')
                    if len(splits) != 3:
                        continue
                    if splits[1] == '' or splits[2] == '':
                        continue
                    outfile.write(diff)
                    outfile.write('\n')



if __name__ == "__main__":
    combine_crawls()

