2022-04-15

README for example_data.csv.
Generated from make_example_data.R.

Covers years 2007 through 2021, each year has 20,000 observations
Features B, E, and F are normally distributed; A, C, D, G, H, I, J are gamma.
Feature cat contains categorical variables ranging from 1-7 sampled with varying probability
Feature confidence contains integers ranging from (0,0.6)

Drift 1 (2009): B changes in mean but not variance
Drift 2 (2012): D changes in just variance (mean stays same)
Drift 3 (2015): E and F become more strongly correlated. Their mean and variance stay the same
Drift 4 (2018 - 2021): H changes in mean and variance, range of Confidence Scores increases to (0.4,1)
Drift 5 (2021): J change mean and variance 

With the except of drift 4, all instances of drift are only for that single year of data. 

