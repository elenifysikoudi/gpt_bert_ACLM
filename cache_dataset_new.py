from smart_open import open
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Cached Dataset Creation')
parser.add_argument(
    '--segments_path',
    type=str,
    default="/mimer/NOBACKUP/groups/naiss2024-6-297/gpt-bert/data/processed/segmented.txt",
    help='Path to the segmented data file.'
)
parser.add_argument(
    '--sequence_length',
    type=int,
    default=128,
    help='Sequence length of each cached input sequence.'
)
args = parser.parse_args()

SEQ_LEN = args.sequence_length

documents = [[]]
for line in tqdm(open(args.segments_path)):
    line = line.strip()
    if len(line) == 0:
        if len(documents[-1]) > 0:
            documents.append([])
        continue
    documents[-1].append(line)

with open(f"./data/processed/cached_{SEQ_LEN}.txt", "w") as f:
    for document in tqdm(documents):
        segment = []
        for sentence in document:
            words = sentence.split()
            segment.extend(words)
            while len(segment) >= SEQ_LEN:
                chunk = " ".join(segment[:SEQ_LEN])
                f.write(chunk + "\n")
                segment = segment[SEQ_LEN:]
        if segment:
            chunk = " ".join(segment[:SEQ_LEN])
            f.write(chunk + "\n")
