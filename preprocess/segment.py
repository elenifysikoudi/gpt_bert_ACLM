import nltk


with open("../data/processed/dev/segmented.txt", "w") as f:
    for line in open("../data/processed/dev/all.txt"):
        line = line.strip()

        if len(line) == 0:
            f.write('\n')
            continue

        sentences = nltk.sent_tokenize(line)
        sentences = '\n'.join(sentences) 
        f.write(f"{sentences}\n")
        #f.write(f"{sentences}[PAR]\n")

with open("../data/processed/segmented.txt", "w") as f:
    for line in open("../data/processed/all.txt"):
        line = line.strip()

        if len(line) == 0:
            f.write('\n')
            continue

        sentences = nltk.sent_tokenize(line)
        sentences = '\n'.join(sentences)
        f.write(f"{sentences}\n")
        #f.write(f"{sentences}[PAR]\n")
