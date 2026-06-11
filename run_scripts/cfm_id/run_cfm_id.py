import pandas as pd
import subprocess
from pathlib import Path
from ms_pred import common


datasets = ["nist20"]
# datasets = ["casmi22"]
num_threads = 96

for dataset  in datasets:

    labels = f"data/spec_datasets/{dataset}/labels.tsv"
    res_folder = Path(f"results/cfm_id_{dataset}/")
    res_folder.mkdir(exist_ok=True)

    cfm_inputs = []
    cfm_output_specs = res_folder / "cfm_out"
    cfm_batch_scripts = res_folder / "batches"
    cfm_batch_scripts.mkdir(exist_ok=True)

    df = pd.read_csv(labels, sep="\t")
    input_items = [f"{i} {j}" for i, j in df[["spec", "smiles"]].values]

    batches = common.batches_num_chunks(input_items, num_threads)
    batches = list(batches)
    for batch_ind, batch in enumerate(batches):
        input_file = cfm_batch_scripts / f"cfm_input_{batch_ind}.txt"
        input_str = "\n".join(batch)
        with open(input_file, "w") as fp:
            fp.write(input_str)
        cfm_inputs.append(input_file)


    def make_cfm_command(cfm_input):
        return f"""cfm-predict '/cfmid/public/{cfm_input}' 0.001 \\
        /trained_models_cfmid4.0/[M+H]+/param_output.log \\
        /trained_models_cfmid4.0/[M+H]+/param_config.txt 1 \\
        /cfmid/public/{cfm_output_specs}"""


    cfm_commands = [make_cfm_command(i) for i in cfm_inputs]

    # Run in background
    full_cmd = "\n".join([f"{i} &" for i in cfm_commands])
    cmd_file = res_folder / "cfm_full_cmd.sh"

    wait_forever_cmd = "\nwhile true; do\n\tsleep 100\ndone"
    with open(cmd_file, "w") as fp:
        fp.write(full_cmd)
        fp.write(wait_forever_cmd)


    docker_str = f""" sudo docker run --rm=true -v $(pwd):/cfmid/public/ \\
    -i wishartlab/cfmid:latest  \\
    sh -c  ". /cfmid/public/{cmd_file}"
    """

    print(docker_str)
    subprocess.run(docker_str, shell=True)
