dataset=nist20  # nist20
ppm_diff=20
workers=64

python data_scripts/forms/01_assign_subformulae.py \
    --data-dir data/spec_datasets/$dataset/ \
    --labels-file data/spec_datasets/$dataset/labels.tsv \
    --use-all \
    --output-dir no_subform.hdf5 \
    --num-workers $workers

python data_scripts/forms/01_assign_subformulae.py \
    --data-dir data/spec_datasets/$dataset/ \
    --labels-file data/spec_datasets/$dataset/labels.tsv \
    --use-magma \
    --mass-diff-thresh $ppm_diff \
    --output-dir magma_subform_50.hdf5 \
    --num-workers $workers

python data_scripts/forms/03_add_form_intens.py \
    --num-workers $workers \
    --pred-form-folder data/spec_datasets/$dataset/subformulae/magma_subform_50.hdf5 \
    --true-form-folder data/spec_datasets/$dataset/subformulae/no_subform.hdf5 \
    --add-raw \
    --binned-add \
    --out-form-folder data/spec_datasets/$dataset/subformulae/magma_subform_50_with_raw.hdf5
