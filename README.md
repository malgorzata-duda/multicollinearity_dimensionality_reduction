# Properties of dimensionality reduction methods in the estimation of linear regression models in the presence of multicollinearity

## Project Overview
The purpose of the project is the analysis of properties of dimensionality reduction methods, such as **Principal Component Regression (PCR) and Partial Least Squares Regression (PLSR)**, in the estimation of linear regression models under different levels of approximate multicollinearity among the explanatory variables. 
The objective of the analysis is:
* the examination the properties of the structural parameters estimators obtained using the above-mentioned methods in various conditions, 
* the comparison of the models' goodness of fit and generalization abilities,
* the comparison of the obtained results with the properties of an OLS model. 

Estimation of the linear regression models was conducted using 3 methods: PCR, PLSR and OLS. Then the properties of the obtained models were studied. 

## Data

In order to analyze various conditions under which a linear regression model could be estimated, analyses were conducted using **simulated data**. Generated datasets vary in:
* sizes, 
* sets of used explanatory variables, 
* strength of correlation between them
* sets of structural parameters' true values.


First, sets of X variables correlated by $\rho$ were generated in the following way: 

$X_{i} = U\sqrt{\rho}\,  + Z_i \sqrt{1-\rho}\$, where:

$Z_i \sim N(0, 1)$,  
$U \sim N(0, 1).$

Analyzed correlation strengths $\rho$ included:
* $\rho$ = 0.2 - almost no multicollinearity, 
* $\rho$ = 0.6 - moderate multicollinearity, 
* $\rho$ = 0.8 - strong multicollinearity,
* $\rho$ = 0.95 - almost perfect multicollinearity.

The target Y variable was constructed in the following way: 

$Y = \beta_1 X_{1} + \beta_2 X_{2} + ... + \beta_p X_{p} + \epsilon$ , where:  
$\epsilon_i \sim i.i.d.  N(0, \sigma^2), \sigma = 3$.

Different model scenarios and sets of structural $\beta$ parameters were analyzed:
* Scenario 1: baseline - all X variables are used in generating y and estimating $\hat{y}$:
* Scenario 2: extra variables - in estimating $\hat{y}$ additional variables were used, 
* Scenario 3: omitted variables - in estimating $\hat{y}$ some variables were omitted.
   
Datasets were generated for different sample sizes: $n$ = 50 and $n$ = 500. In each case 1000 simulations were performed. 

## Methodology
The project compares 3 methods of estimating linear regression models: OLS, PCR and PLSR. Detection of multicollinearity was conducted through Variance Inflation Factor (VIF) and Condition Index (CI) measures. After model estimation, the following properties of the estimators were analyzed: absolute bias, standard deviation (variance) and proportion of accurately estimated parameter signs. Predictive and generalization abilities of the model were analyzed by $R^2$ determination coefficient and MAE obtained through 5-fold cross-validation.  
In PCR, 5 levels of variance explained by the components were analyzed: 0.5, 0.6, 0.7, 0.8 and 0.9. Usage of the number of components equal to the number of explaining variables would be identical to an OLS estimation. In PLSR the number of latent variables used was equal to the number of PCR components to ensure the comparability of the methods.  


## Results

### Bias
<img width="1176" height="446" alt="bias2" src="https://github.com/user-attachments/assets/44e107e8-5ef2-4e86-96b6-ad8835d424b2" />


* OLS estimators are unbiased, regardless of the multicollinearity degree. On the contrary, the PCR and PLSR methods generally introduce substantial estimation bias, at times ranging to even 100-400% of the true parameter value. 
* In general, PLSR allows for less biased estimations than PCR, especially in case of lower correlations. 

### Standard deviation

<img width="1189" height="446" alt="sd" src="https://github.com/user-attachments/assets/6897afd4-461c-4cdc-8fbc-20d9806393cc" />


* Application of PCR and PLSR methods allows to reduce the standard deviations of estimators especially in 2 cases:
    * very strong correlation,
    * strong correlation and small sample.  
* PLSR method is more universal than the PCR method, as it allows for precise estimation both in case of weak and strong multicollinearity. 
* In the baseline model, it is recommended to use the number of components covering no more than 0.6 of the variance. Using too many components/latent variables might result in a sudden drop of the precision. 

### Sign accuracy

<img width="1189" height="446" alt="sign_accuracy_1" src="https://github.com/user-attachments/assets/c3c3f7ad-6299-4fbe-9956-ecb2912032a2" />
<img width="1189" height="446" alt="sign_accuracy_2" src="https://github.com/user-attachments/assets/c2d561ac-8af4-434d-a52b-d41109845bbf" />

* When all real signs of the coefficients are consistent (all positive or all negative), PLSR method allows to almost perfectly assess the signs of the coefficients. 
* PLSR method is more universal than PCR method - it allows for accurate estimation in more versatile conditions, such as weaker multicollinearity.
* Dimensionality reduction methods are especially effective in case of small sample and high or very high correlation. 
* However, in case of varying coefficient signs, the accuracy of PCR and PLSR methods is lower than the OLS method. 

### Goodness of fit and generalization abilities

<img width="1172" height="446" alt="metrics" src="https://github.com/user-attachments/assets/722317fe-4ebb-47c5-ba60-de6cdefcd0d8" />

* The dimensionality reduction methods do not improve the predictive and generalization abilities of the regression model. 
* However, if the dimensionality reduction methods are used, PLSR performs better than PCR. 


### Result summary
The analysis allowed to outline the properties of the PCR and PLSR methods in terms of the bias, variance, sign accuracy and model's goodness of fit, depending on the size of the sample, the degree of multicollinearity and number of variables used. Based on these properties, the following recommendations can be formed:
* The dimensionality reduction methods allow to reduce the variation of the estimators at the cost of introducing bias (bias-variance tradeoff). 
* The precision of the estimation can be improved by these methods especially in 2 particular cases: 
    * very high correlation between variables,
    * high correlation between variables and small sample size.
* PCR and PLSR methods perform very similarly in case of strong mutlicollinearity. However, in case of weaker correlations, PLSR method is generally more effective than the PCR method, which sometimes performs even worse than the OLS method. Therefore, the PLSR method is more universal and effective. 
* Benefits from implementing the dimensionality reduction methods occur when the number of explaining variables is appropriately reduced. It is recommended to use the number of variables explaining no more than 60-70% of variance. Using too many components or latent variables might even result in degredation of model performance in comparison to the OLS method. 
* PLSR method allows to accurately estimate the directions of the relationships between the variables, only when real parameter signs are constistent. Therefore, this method is recommended when expertise or prior knowledge indicates consistent parameter signs. 
* The dimensionality reduction methods do not improve the predictive abilities of the model. However, if based on other factors these methods are to be implemented, it is recommended to use the PLSR method instead of PCR.

# Running the analysis
The analysis was conducted in 2 provided notebooks. 

1. `01_data_simulation_and_model_estimation.ipynb`
2. `02_visualization_and_analysis.ipynb` 

Due to the fact that simulations in `01_data_simulation_and_model_estimation.ipynb` might be computationally intensive, the already obtained results are available in `data/simulated_data.pkl`. Therefore running the first notebook is not necessary as `02_visualization_and_analysis.ipynb` reads in already computed dataset. Due to a high number of scenarios and sets of structural parameters analyzed, the visualizations in this notebook are interactive - a dropdown menu allows the user to select which scenarios and parameters ought to be visualized.


## Technologies
Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, statsmodels, Jupyter Notebook, IPyWidgets

## Use of AI
AI tools were used to support the development of the code, calculations, and visualization of the results.
