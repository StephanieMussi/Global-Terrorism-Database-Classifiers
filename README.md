# Global_Terrorism_Database_Classifiers
This projects aims to perform data mining tasks on the [Global Terrorism Database](https://www.kaggle.com/START-UMD/gtd). Particularly, two classifiers are implemented to explore the correlation among different variables of the terrorist attacks.

## Data pre-processing
The [Global Terrorism Database](https://www.kaggle.com/START-UMD/gtd) contains information of 181692 terrorist attacks from 1970 to 2017 (except for 2013). Moreover, there are 135 variables for each case. Data pre-processing is required to remove missing attributes and outliers, and reduce the size of the dataset.
The cleaned dataset "gtdb(clean).csv" contains 159670 cases with 13 variables, among which only the following variables of interest are used for classification: 
*year*, *month*, *country*, *region*, *city*, *success*, *suicide*, *attack type* and *target type*. 

## Month Classifier
## Region Classifier
