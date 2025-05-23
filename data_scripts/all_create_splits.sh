dataset=nist20  # nist20

python data_scripts/make_splits.py --data-dir data/spec_datasets/$dataset --label-file data/spec_datasets/$dataset/labels.tsv  --seed 1 --split-type fingerprint --split-name fingerprint_1.tsv

python data_scripts/make_splits.py --data-dir data/spec_datasets/$dataset/ --label-file data/spec_datasets/$dataset/labels.tsv --seed 1 --split-type scaffold --split-name scaffold_1.tsv --greedy


python data_scripts/make_splits.py --data-dir data/spec_datasets/$dataset/ --label-file data/spec_datasets/$dataset/labels.tsv --seed 1


python data_scripts/make_splits.py --data-dir data/spec_datasets/$dataset/ --label-file data/spec_datasets/$dataset/labels.tsv --seed 2


python data_scripts/make_splits.py --data-dir data/spec_datasets/$dataset/ --label-file data/spec_datasets/$dataset/labels.tsv --seed 3

# hyperopt
python data_scripts/make_splits.py --data-dir data/spec_datasets/$dataset/ --label-file data/spec_datasets/$dataset/labels.tsv  --seed 1 --split-name hyperopt.tsv --test-frac 0.5

