import re

def tokenize(line):
    # Includes all preprocessing
    # because tokenize is run before preprocessing in data.Field
    
    # Split word with hyphens
    line = re.sub(r'-',' ',line)
    # Remove all special characters
    line = re.sub(r'[^ a-zA-Z0-9]','',line)
    # Remove excess whitespace
    line = re.sub(r'\s+',' ',line)
    # lower and remove left and right whitespace
    line = line.lower().strip()
    return line.split()


def preprocess_token_list(token_list):
    return [re.sub(r'[^a-zA-Z0-9]','',token) for token in token_list]

def preprocess_line(line):
    # Leave spaces
    line = re.sub(r'[^ a-zA-Z0-9]','',line)
    # Remove excess whitespace
    line = re.sub(r'\s+',' ',line)
    return line.strip()

def printout(line):
    print(line)
    return line

