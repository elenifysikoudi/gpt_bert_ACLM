from smart_open import open
from normalize import clean



def preprocess(f):
    last_num_non_blank_lines = 0
    num_blank_lines = 0
    accumulated_line = []
    for line in f:
        line = ' '.join(line.strip().split())
        line = clean(line, minimal=True)

        if len(line) == 0:
            if len(accumulated_line) > 0:
                yield ' '.join(accumulated_line)
                last_num_non_blank_lines = len(accumulated_line)

            if num_blank_lines == 1 and last_num_non_blank_lines > 1:
                yield ""

            accumulated_line = []
            num_blank_lines += 1
            continue

        num_blank_lines = 0
        accumulated_line.append(line)


with open("/mimer/NOBACKUP/groups/naiss2024-6-297/BabyLM2025/text_data/train_10M/gutenberg.train") as f:
    with open("../data/processed/gutenberg.txt", 'w') as g:
        for line in preprocess(f):
            g.write(f"{line}\n")

with open("/mimer/NOBACKUP/groups/naiss2024-6-297/BabyLM2025/text_data/dev/gutenberg.dev") as f:
    with open("../data/processed/dev/gutenberg.txt", 'w') as g:
        for line in preprocess(f):
            g.write(f"{line}\n")
