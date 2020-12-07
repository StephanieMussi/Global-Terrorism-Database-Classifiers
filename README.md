# Global_Terrorism_Database_Classifiers
This projects aims to perform data mining tasks on the [Global Terrorism Database](https://www.kaggle.com/START-UMD/gtd). Particularly, two classifiers are implemented to explore the correlation among different variables of the terrorist attacks.

## Data pre-processing
The [Global Terrorism Database](https://www.kaggle.com/START-UMD/gtd) contains information of 181692 terrorist attacks from 1970 to 2017 (except for 2013). Moreover, there are 135 variables for each case. Data pre-processing is required to remove missing attributes and outliers, and reduce the size of the dataset.  
The cleaned dataset ["gtdb(clean).csv"](https://github.com/StephanieMussi/Global_Terrorism_Database_Classifiers/blob/main/gtdb(clean).zip) contains 159670 cases with 13 variables, among which only the following variables of interest are used for classification:   
__year__, __month__, __country__, __region__, __city__, __success__, __suicide__, __attack type__ and __target type__. 

## Month Classifier
In this classifier, there are 12 output classes:  
1(_January_), 2(_February_), 3(_March_), 4(_April_), 5(_May_), 6(_June_), 7(_July_), 8(_August_), 9(_September_), 10(_October_), 11(_November_), 12(December_).  
A deep neural network with 5 hidden layers is used:  

```python
model = keras.Sequential([  
    keras.layers.Flatten(input_shape=(1, 8)),    
    keras.layers.Dense(640, activation='relu'),  
    keras.layers.Dense(480, activation='relu'),  
    keras.layers.Dense(320, activation='relu'),  
    keras.layers.Dense(160, activation='relu'),  
    keras.layers.Dense(80, activation='relu'),  
    keras.layers.Dense(12, activation='softmax')   
])
```
There are two ways to implement the output Softmax layer:  
  1. No. of classes = 12, and all values of month are decreased by 1 to fit the 0~11 output range of the Softmax layer.
  1. No. of classes = 13, and no modification is needed.

## Region Classifier
In this classifier, there are 12 output classes:  
1(_North America_), 2(_Central America & Caribbean_), 3(_South America_), 4(_East Asia_), 5(_Southeast Asia_), 6(_South Asia_), 7(_Central Asia_), 8(_Western Europe_), 9(_Eastern Europe_), 10(_Middle East & North Africa_), 11(_Sub-Saharan Africa_), 12(_Australasia & Oceania_).
