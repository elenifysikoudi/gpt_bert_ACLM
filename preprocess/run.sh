mkdir -p ../data/processed

python3 childes.py
python3 bnc_spoken.py
python3 gutenberg.py
python3 open_subtitles.py
python3 simple_wiki.py
python3 switchboard.py

cat ../data/processed/childes.txt ../data/processed/bnc_spoken.txt ../data/processed/gutenberg.txt ../data/processed/open_subtitles.txt ../data/processed/simple_wiki.txt ../data/processed/switchboard.txt  > ../data/processed/all.txt

cat ../data/processed/dev/childes.txt ../data/processed/dev/bnc_spoken.txt ../data/processed/dev/gutenberg.txt ../data/processed/dev/open_subtitles.txt ../data/processed/dev/simple_wiki.txt ../data/processed/dev/switchboard.txt  > ../data/processed/dev/all.txt

python3 segment.py
