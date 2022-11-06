"""
This Project aims to implement two methods for parameter estimation:
    1. Ordinary Least Squares (OLS)
    2. Maximum Likelihood Estimation
    3. Bayesian Inference

ESTIMATION PROBLEM: Understanding the height of Python programmers.
A. We assume that the model which best describes the data generation process is GAUSSIAN (Normal) => We estimate the mean and std

OBS:
* The Normal distribution when modelling natural phenomena like human heights.
* The Beta distribution when modelling probability distributions.
* The Poisson distribution when modelling the frequency of events occurring.

Each of these distributions have parameters: α and β for the beta distribution, λ for the Poisson,
lower and upper bounds for the uniform distribution, or μ and σ for the normal distribution of our example.
"""


import math
import numpy as np
import scipy.stats
from matplotlib import pyplot

# LIKELIHOOD - Compute the Maximum Likelihood Estimation
sample_of_python_programmers_height = [183, 168, 177, 170, 175, 177, 178, 166, 174, 178]
mu = 175.
sigma = 5.
likelihood = scipy.stats.norm.pdf(sample_of_python_programmers_height, mu, sigma)
likelihood_log = scipy.stats.norm.logpdf(sample_of_python_programmers_height, mu, sigma)
print(likelihood)
print(likelihood_log)

# PRIOR - choosing a prior is quite subjective
# Example:
# mean_height = numpy.linspace(0, 272, 273) # min (0) to max (272) height in 273 steps
# probability = scipy.stats.uniform.pdf(mean_height, 0, 272) # uniform distribution
# MANUAL METHOD:
world_height_mean = 165
world_height_standard_deviation = 7
# mean_height = numpy.linspace(0, 272, 273)
# probability = scipy.stats.norm.pdf(mean_height, world_height_mean, world_height_standard_deviation * 2)
# pyplot.plot(mean_height, probability)
# pyplot.xlabel('Mean height')
# pyplot.ylabel('Probability')
# pyplot.title('Informed prior for Python developers height')
# pyplot.show()
# EFFICIENT METHOD:
prior = scipy.stats.norm.pdf(sample_of_python_programmers_height, world_height_mean, world_height_standard_deviation * 2)
prior_log = scipy.stats.norm.logpdf(sample_of_python_programmers_height, world_height_mean, world_height_standard_deviation * 2)
print(prior)
print(prior_log)

# EVIDENCE:
"""
to be discussed
"""
# POSTERIOR:
posterior = prior * likelihood.prod()
posterior_log = np.exp(prior_log + likelihood_log.sum())
print("*")
print(posterior)
print("*")
print(posterior_log)



