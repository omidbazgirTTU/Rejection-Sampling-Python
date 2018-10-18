# Rejection-Sampling-Python
This is an implementation of rejection sampling theory in python, where one can sample from any kind of distribution and visualize it. 
This code samples from Kumaraswamy distribution, a bivariate distribution. The bivariate Kumaraswamy distribution equation is 
shown in the image of the provided link:

(https://github.com/omidbazgirTTU/Rejection-Sampling-Python/blob/master/Kumaraswamy.png)

To sample from the bivariate distribution, two 1-D uniform distribution generated. Then, using copula, their joint pdf is generated. Each pair of samples with their corresponding probability is compared with their correspoing distribution probabilty, then they are either rejected or accepted based on a threshold
