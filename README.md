# Rejection-Sampling-Python
This is an implementation of rejection sampling theory in python, where one can sample from any kind of distribution and visualize it. 
This code samples from Kumaraswamy distribution, a bivariate distribution.




To sample from the bivariate distribution, two 1-D uniform distribution generated. Then, using copula, their joint PDF is generated. Each pair of samples with their corresponding probability is compared with their correspoing distribution probabilty, then they are either rejected or accepted based on a threshold