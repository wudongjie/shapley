# Using Shapley Value Regression with Gini coefficients

This is code for running Shapley Value Regression with Gini coefficients. After training a model, we want to learn how each predictor contributes to the model prediction. One way to tackle this problem is to use Shapley Value Regression (SVR). A conventional SVR approach applies on a linear regression model and decompose the R-square using Shapley values. This code uses Gini coefficients instead of R-square so that it can be applied to not only linear regression models but also a variety of different models such as decision tree, neural network, etc.

## Current Supported Models

Currently, this code only supports a linear regression model and a log-linear regression model. The log-linear regression model is the model in which the dependent variable is transformed using the logarithm function. In the future, more models will be added.

## Usage
usage: 
```
train.py [-h] [-v] [-i INPUT] [-o OUTPUT] [-m MODEL] [-s STEP]

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         verbose computation
  -i INPUT, --input INPUT
                        add the input csv file
  -o OUTPUT, --output OUTPUT
                        the csv file you want to output (default result.csv)
  -m MODEL, --model MODEL
                        specify the model, currently support 'linear' or 'log-
                        linear', default: linear
  -s STEP, --step STEP  steps for computing each Shapley value (default
                        10,000)
```

**Training**
1. Prepare your training data and record the path of your data

You can create a directory named "data" in the root directly. I created a "test_data" directory for you to try and test this code. 

The data should be saved in a **csv** file. The **first row** should be the name of each variable and the **first column** should be your dependent variable.

2. run the "train.py" using python3.

Please specify the input and output files using "-i" for input and "-o" for output. The input path is the path of your csv file. The default output is "result.csv" in the root directory.

3. specify the model you use by "-m".
Currently, only "log-linear" and "linear" are supported. The default value is "linear".

**Using The Code of the Shapley Regression Directly**

You can also using Shapley regression directly without running in the command line. You can import the shapley model in your code:

``` python
import shapley  # make sure your file is at the root directory
```

**Requirements**

Numpy

Sklearn

## TODO
* Neural Network Model

