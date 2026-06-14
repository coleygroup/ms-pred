dataset=nist20 # nist20, nist23, msg
max_peaks=50
ppm_diff=20
workers=32

python3 src/ms_pred/magma/run_magma.py  \
--spectra-dir data/spec_datasets/$dataset/spec_files.hdf5  \
--output-dir data/spec_datasets/$dataset/magma_outputs  \
--spec-labels data/spec_datasets/$dataset/labels.tsv \
--max-peaks $max_peaks \
--ppm-diff $ppm_diff \
--workers $workers \

# for msg: may need to override the --spectra-dir flag to point to a different spec_files folder as needed. 

if [ -f "data/spec_datasets/$dataset/subformulae/no_subform.hdf5" ]; then
  echo "no_subform.hdf5 exists for $dataset, skipping"
else
  python data_scripts/forms/01_assign_subformulae.py \
  --data-dir data/spec_datasets/$dataset/ \
  --spectra-dir data/spec_datasets/$dataset/spec_files_w_imputed_eV \
  --labels-file data/spec_datasets/$dataset/labels.tsv \
  --use-all \
  --output-dir no_subform.hdf5 \
  --num-workers $workers \

fi
--ppm-diff $ppm_diff \
--workers $workers

if [ -f "data/spec_datasets/$dataset/subformulae/no_subform.hdf5" ]; then
  echo "no_subform.hdf5 exists for $dataset, skipping"
else
  python data_scripts/forms/01_assign_subformulae.py \
  --data-dir data/spec_datasets/$dataset/ \
  --labels-file data/spec_datasets/$dataset/labels.tsv \
  --use-all \
  --output-dir no_subform.hdf5 \
  --num-workers $workers
fi