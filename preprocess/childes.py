from smart_open import open
from normalize import clean


def preprocess(f):
    prev_line = None
    for line in f:
        line = line.strip()

        #line = line[0].upper() + line[1:]
        line = clean(line)
        line = f'"{line}"'

        if prev_line is not None and prev_line == line:
            continue

        yield line
        prev_line = line


with open("/mimer/NOBACKUP/groups/naiss2024-6-297/BabyLM2025/text_data/train_10M/childes.train") as f:
    with open("../data/processed/childes.txt", 'w') as g:
        for line in preprocess(f):
            g.write(f"{line}\n")

with open("/mimer/NOBACKUP/groups/naiss2024-6-297/BabyLM2025/text_data/dev/childes.dev") as f:
    with open("../data/processed/dev/childes.txt", 'w') as g:
        for line in preprocess(f):
            g.write(f"{line}\n")
