from scipy.stats import bootstrap
import numpy as np
rank = [1, 5, 20]
is_hit = ["true", "false"]
np.random.seed(1)
with open("results/marason_inten_msg/split_rnd1_entropy/retrieval_msg_test_formula_256/rerank_eval_entropy.yaml", "r") as file:
    contents = file.read()
for r in rank:
    results = []
    for hit in is_hit:
        target_string = f"top_{r}: {hit}"
        count = contents.count(target_string)
        print(f"The string '{target_string}' occurs {count} times.")
        if hit == "true":
            results += [True] * count
        else:
            results += [False] * count
    res = bootstrap(
        (results,),  # Must be a tuple
        np.mean,  # Statistic function
        confidence_level=0.999,
        n_resamples=20000,
    )

    print(f"Top {r} Estimated proportion: {np.mean(results):.4f}")
    print(f"Top {r} 99.9% Confidence interval: ({res.confidence_interval.low:.4f}, {res.confidence_interval.high:.4f})")
    



