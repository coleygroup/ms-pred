from scipy.stats import bootstrap
import numpy as np
import yaml
np.random.seed(1)

path = "dag_inten_msg_spec/split_1_rnd1/retrieval_msg_test_formula_256/rerank_eval_entropy_torchmetrics.yaml"
with open(path, "r") as f:
    data = yaml.load(f, Loader=yaml.UnsafeLoader)
#print(data)

# dists:
entropy_sim = []
cosine_sim = []

for entry in data['individuals']:
    if "true_dist" in entry:
        esim = 1 - entry['true_dist']
        entropy_sim.append(esim)
    if 'cosine_dist' in entry:
        c_sim = 1 - entry['cosine_dist']
        cosine_sim.append(c_sim)

res = bootstrap(
    (entropy_sim,),  # Must be a tuple
    np.mean,  # Statistic function
    confidence_level=0.999,
    n_resamples=20000,
)
print(f"Entropy distribution Estimated proportion: {np.mean(entropy_sim):.4f}")
print(f"Entropy distribution 99.9% Confidence interval: ({res.confidence_interval.low:.4f}, {res.confidence_interval.high:.4f})")
    
res = bootstrap(
    (cosine_sim,),  # Must be a tuple
    np.mean,  # Statistic function
    confidence_level=0.999,
    n_resamples=20000,
)
print(f"Cosine distribution Estimated proportion: {np.mean(cosine_sim):.4f}")
print(f"Cosine distribution 99.9% Confidence interval: ({res.confidence_interval.low:.4f}, {res.confidence_interval.high:.4f})")

metrics = data['metrics']
print(metrics)
# Metrics are means of a Bernoulli distribution, so we can just reconstruct the original array for the num of examples. 
top1 = int(metrics['hit_rate_at_1'] * len(data['individuals'])) * [1] + int((1 - metrics['hit_rate_at_1']) * len(data['individuals'])) * [0]
top5 = int(metrics['hit_rate_at_5'] * len(data['individuals'])) * [1] + int((1 - metrics['hit_rate_at_5']) * len(data['individuals'])) * [0]
top20 = int(metrics['hit_rate_at_20'] * len(data['individuals'])) * [1] + int((1 - metrics['hit_rate_at_20']) * len(data['individuals'])) * [0]


res = bootstrap(
    (top1,),  # Must be a tuple
    np.mean,  # Statistic function
    confidence_level=0.999,
    n_resamples=20000,
)
print(f"Top 1 distribution Estimated proportion: {np.mean(top1):.4f}")
print(f"Top 1 distribution 99.9% Confidence interval: ({res.confidence_interval.low:.4f}, {res.confidence_interval.high:.4f})")


res = bootstrap(
    (top5,),  # Must be a tuple
    np.mean,  # Statistic function
    confidence_level=0.999,
    n_resamples=20000,
)
print(f"Top 5 distribution Estimated proportion: {np.mean(top5):.4f}")
print(f"Top 5 distribution 99.9% Confidence interval: ({res.confidence_interval.low:.4f}, {res.confidence_interval.high:.4f})")


res = bootstrap(
    (top20,),  # Must be a tuple
    np.mean,  # Statistic function
    confidence_level=0.999,
    n_resamples=20000,
)
print(f"Top 20 distribution Estimated proportion: {np.mean(top20):.4f}")
print(f"Top 20 distribution 99.9% Confidence interval: ({res.confidence_interval.low:.4f}, {res.confidence_interval.high:.4f})")


