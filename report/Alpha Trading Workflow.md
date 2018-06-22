# Alpha Trading Workflow

Analyst: Yuxuan Xia

Date: 2018/06/01

## TODO

* Input more effective factors: take advice from people and industry reports
* Quaterly Data and Annually data, how to use them? Decrease the system frequency to quaterly?
* Improve perfomance through deep learning or statistical models?
* Find well-known metrics to express results

## Workflow
\checkmark stands for finished and \vartriangle stands for TODO

* Universe definition
* Factors collection and preprocessing
	* $\vartriangle$ Factors collection
		- Sources
			- balance sheet
			- cash flow statement
			- income statement
			- earning report
		- Econometric Classifications
			- value
			- growth
			- profitability
			- market size
			- liquidity
			- volatility
			- Momentom
			- Financial leverage (debt-to-equity ratio)
	* Factors preprocessing
		- $\vartriangle$daily, quaterly, annually
		- continuous: rescale, outliers
		- $\checkmark$discrete: rank
* Factors screening and combination
	* Factors screening
		- $\checkmark$Factors' correlation
		- $\checkmark$Factors' foreseeablity
		- Fama-Macbeth regression
	* $\vartriangle$Factors combination
		- PCA, FA
		- Financial Modeling
		- Linear combination to maximize Sharpe ratio
		- Non-linear learning algorithms
			- $\checkmark$AdaBoost
			- Reinforcement learning

* Portfolio allocation


## Factors' Correlations
Here, I use correlation matrix as the measure. The difference from the second result is that the correlation matrix is calculated by the rank data rather than the raw data
### Two ICs comparison
* Pearson's IC: If the sample size is moderate or large and the population is normal, then, in the case of the bivariate normal distribution, the sample correlation coefficient is the maximum likelihood estimate of the population correlation coefficient, and is asymptotically unbiased and efficient, which roughly means that it is impossible to construct a more accurate estimate than the sample correlation coefficient. The number itself has no sense if you don't find a proper way or "common sense" to interpret it. Multi-variate Gaussian distribution give us such a common sense of how it should looks like.

* Spearman's IC: while Pearson's correlation assesses linear relationships, Spearman's correlation assesses monotonic relationships (whether linear or not). Since We only care about the monotonic relationships. Spearman's IC wins.


### Regular IC(Pearson's correlation coefficient) for each factors
![](Corr matrix for raw factors.png)
### Spearman's Rank correlation coefficient for each factors
![](Corr matrix for factor ranks.png)

### How to rule out redundant factors and why Spearman's rank correlation coefficients?
From the correlation coefficients below, we can again conclude that Spearman's rank IC is far more robust. Take ps_ratio and sales_yield as a example.
$$ps\_ratio = \frac{\mbox{adjusted close price}}{\mbox{sales per share}}$$
whereas
$$sales\_yield = \frac{\mbox{sales per share}}{\mbox{price}}$$
Ahthogh the price in sales_yield formula is vague in our data source we can see roughly speaking, these two variable should be inverse of each other. The Spearman's rank correlation coefficient is -0.98 which verifies this statement, and we should avoid using both of these factors, which would exeggarate the impact of this peticular factor. However, we can not see such identity in the Pearson's regular correlation coefficients. It's quite misleading actually and that's why we choose Spearman's rank IC.

## Factors' Foreseeability

### Mehods
* Spearman's rank correlation coefficients
* Fama-Macbeth regression: Not only consider the foreseeability of factors itself but also consider the co-vary of different factors, which means rule out factors if the returns can be explained by the recent factors.


### Spearman's rank IC for factors vs. forward returns

![](mean spearmans rank IC.png)

### Spearman's rank IC (absolute value) for factors vs. forward returns
![](mean spearmans rank IC (absolute value).png)

### Rank of the Spearman's rank IC (absolute value) for factors vs. forward returns
![](rank of mean spearmans rank IC (absolute value).png)

## Alpha Factor Combination
construct an aggregate alpha factor which has its return distribution profitable. The term "profitable" here means condense, little turnover, significant in the positive return.
### Methods
#### linear methods
* normalize factors and try a linear combination 
* rank each factor and then sum up
* Financial modeling
* linear combination to maximize Sharpe ratio

#### Non-linear methods
* AdaBoost
* Reinforement Learning

### AdaBoost
#### Description
The algorithm sequentially applies a weak classification to modified versions of the data. By increasing the weights of the missclassified observations, each weak learner focuses on the error of the previous one. The predictions are aggregated through a weighted majority vote.

#### Algorithm

![](adaboost_algorithm.png)

#### Train set
![](train_score_dist.png)
![](train_accuracy_bar.png)

#### Test set
![](test_score_dist.png)
![](test_accuracy_bar.png)

## References
* Jonathan Larkin, *A Professional Quant Equity Workflow*. August 31, 2016
* *A Practitioner‘s Guide to Factor Models*. The Research Foundation of The Institute of Chartered Financial Analysts
* Thomas Wiecki, Machine Learning on Quantopian
* Inigo Fraser Jenkins, *Using factors with different alpha decay times: The case for non-linear combination* 
* PNC, *Factor Analysis: What Drives Performance?*
* O’Shaughnessy, *Alpha or Assets? — Factor Alpha vs. Smart Beta*. April 2016
* *O’Shaughnessy Quarterly Investor Letter Q1 2018* 
* Jiantao Zhu, Orient Securities, *Alpha Forecasting - Factor-Based Strategy Research Series 13*
* Yang Song, Bohai Securities, *Multi-Factor Models Research: Single Factor Testing*, 2017/10/11