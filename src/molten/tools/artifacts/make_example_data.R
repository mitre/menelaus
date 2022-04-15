# Leigh Nicholl
# Created 2021-10-21
# 2022-01-27 updates:
# This script was used to generate example_data.csv, already included in the repo.
# 1) add a persistent drift (never reverting back to ref for one var)
# 2) add an extra year between true drift
# 3) make change in variance more dramatic
# 4) add categorical variable (1-7) 

# 2022-02-14 updates: add correlation change where marginals stay the same

# 2022-03-18 updates: add an extra column representing classifier confidence scores and add drift indicator variable

# Purpose: create fake data with synthetic drift 

############################# Eligibility data #################################

set.seed(123)
df <- data.frame(id = rep(1:20000, 15))
df$year  <- sort(rep(2007:2021, 20000))

df$a <- rgamma(300000, shape = 8)*1000 
df$b <- rnorm(300000, mean = 200, sd = 10)
df$c <- rgamma(300000, shape = 7)*1000
df$d <- rgamma(300000, shape = 10)*10000
df$e <- NA
df$f <- NA
df[, c("e", "f")] <- MASS::mvrnorm(n = 300000, mu = c(0, 0), 
                                   Sigma = matrix(c(2, 0, 0, 2), 2, 2, byrow = TRUE))
df$g <- rgamma(300000, shape = 11)*10000
df$h <- rgamma(300000, shape = 12)*1000
df$i <- rgamma(300000, shape = 9)*1000
df$j <- rgamma(300000, shape = 10)*100
df$cat <- sample(1:7, 300000, replace = T, prob = c(0.3, 0.3, 0.2, 0.1, 0.05, 0.04, 0.01))
df$confidence <- runif(300000, min = 0, max = 0.6)

# Drift 1: change the mean of B in 2009, means will revert for 2010 on
df[df$year == 2009, ]$b <- rnorm(20000, mean = 500, sd = 10) # bigger change

# Drift 2: change the variance of c and d in 2012 by replacing some with the mean
# keep same mean as other years, revert by 2013
mu_c <- mean(df$c)
mu_d <- mean(df$d)
df[df$year == 2012 & df$id %% 10 == 0, ]$c <- mu_c + rnorm(1000, sd = 10) # subtle change, every 10 obs
df[df$year == 2012 & df$id %% 2 == 0, ]$d <- mu_d + rnorm(10000, sd = 10) # bigger change, every other obs

# Drift 3: change the correlation of e and f in 2015 (go from correlation of 0 to correlation of 0.5)
df[df$year == 2015, c("e", "f")] <- MASS::mvrnorm(n = 20000, mu = c(0, 0), 
                                                    Sigma = matrix(c(2, 1, 1, 2), 2, 2, byrow = TRUE))

# Drift 4: change mean and var of H and persist it from 2018 on, change range of confidence scores
df[df$year > 2018, ]$h <- rgamma(60000, shape = 1, scale = 1)*1000
df[df$year > 2018, ]$confidence <- runif(60000, min = 0.4, max = 1)

# Drift 5: change mean and var just for a year of J in 2021
df[df$year == 2021, ]$j <- rgamma(20000, shape = 10)*10

df$drift <- df$year %in% c(2009, 2012, 2015, 2018, 2021)

# write out to shared drive
write.csv(df, 'example_data.csv',
          row.names = F)
