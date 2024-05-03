# Similar Env (Push, Buckets, Seesaw) : Shared - 23045824, line 25036
# model1_means = list(0.72350, 0.60250, 0.04300)
# model1_stds = list(2.2676, 2.3350, 0.0276)
# Similar Env: SM Shallow - 23066522, line 12394
# model2_means = list(5.1295, 2.9870, 1.2355)
# model2_stds = list(4.7826, 4.5379, 0.5365)  

# Similar Env: SM Deep - 23044551, line 20822
# model2_means = list(1.837, 1.6905, 0.0525)
# model2_stds = list(3.7069, 3.8902, 0.0274)

# Different Env(Push, Ladder, Cannon) : Shared - 23065450, line 18112
# model1_means = list(0.8275, 3.1775, 0.2935)
# model1_stds = list(2.2943, 4.2670, 0.4391)
# Different Env(Push, Ladder, Cannon) : SM Shallow - 23067510, line 18113
# model2_means = list(0.2540, 5.1814, 1.246)
# model2_std = list(3.0850, 4.3608, 3.1817)

# Different Env: SM Deep - 
# model2_means = list(0.0380, 1.2195, 0.5745)
# model2_stds = list(4.1322, 2.9724, 0.5114)

# 5 Similar Env: Shared (Push, Bucket, Seesaw, PushMN, BucketMN) - 23070384, line = 29356
model1_means = list(0.0625, 0.0615, 0.0355, 1.0445, 0.7345)
model1_stds = list(0.0166, 0.0292, 0.0308, 0.0235, 2.4335)

# 5 Similar Env: SM Shallow - 23070383, line 12041
# model2_means = list(0.1225, 0.2245, 0.1880, 1.041, 1.6125)
# model2_stds = list(0.2349, 0.4733, 0.4054, 0.0113, 2.5472)

# 5 Similar Env: SM Deep - 23070382, line 9044
model2_means = list(0.1022, 1.218, 0.0335, 1.033, 0.1115)
model2_stds = list(0.3632, 3.3007, 0.0173, 0.0192, 0.3349)

num_envs = 5
sample_size = 20
df = 2 * 20 - 2 # 38
alpha <- 0.05

# Perform the t-test for each environment and apply Bonferroni correction
for (i in 1:num_envs) {
  # Extract the mean rewards and standard deviations for the current environment
  model1_mean <- unlist(model1_means[i])
  model1_std <- unlist(model1_stds[i])
  model2_mean <- unlist(model2_means[i])
  model2_std <- unlist(model2_stds[i])
  
  # print(model1_mean)
  # print(model2_mean)
  
  # Perform the two-sample t-test
  lower_half = sqrt((model1_std)^2/sample_size + (model2_std)^2/sample_size)
  t_value = (model1_mean - model2_mean) / lower_half
  
  # print(t_value)
  
  # Interested if mu_2 > mu_1
  p_value = pt(abs(t_value), df, lower.tail = FALSE)
  
  
  # Print the results
  cat("Environment", i, ":\n")
  cat("  p-value =", p_value, "\n")
  cat("  Significance level =", alpha, "\n")
  
  if (p_value < alpha) {
    cat("  The performance of the two models is significantly different.\n")
  } else {
    cat("  The performance of the two models is not significantly different.\n")
  }
  cat("\n")
}
