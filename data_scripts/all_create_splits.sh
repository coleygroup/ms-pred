dataset=nist20

python data_scripts/make_splits.py --data-dir data/spec_datasets/$dataset/ --label-file data/spec_datasets/$dataset/labels.tsv --seed 1 --split-type scaffold --split-name scaffold_1.tsv --greedy

python data_scripts/make_splits.py --data-dir data/spec_datasets/$dataset/ --label-file data/spec_datasets/$dataset/labels.tsv --seed 1

# hyperopt
python data_scripts/make_splits.py --data-dir data/spec_datasets/$dataset/ --label-file data/spec_datasets/$dataset/labels.tsv  --seed 1 --split-name hyperopt.tsv --test-frac 0.5

dataset=nist23

python data_scripts/make_splits.py --data-dir data/spec_datasets/$dataset/ --label-file data/spec_datasets/$dataset/labels.tsv --seed 1 --split-type scaffold --split-name scaffold_1.tsv --greedy

# reflect existing nist20 split to make numbers comparable
python data_scripts/make_splits.py --data-dir data/spec_datasets/$dataset/ --label-file data/spec_datasets/$dataset/labels.tsv --seed 1 --existing-split-file data/spec_datasets/nist20/splits/split_1.tsv --existing-label-file data/spec_datasets/nist20/labels.tsv
