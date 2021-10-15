# Multidimensional-Personality-Test-for-Children-MPT-C-
Development and Deep Learning-based Validation of a New Projective Test for Children. 

Author : Aaron Hameiri

Email : aaron.hameiri@mpt-c.org

This project contains the implementation of training Physcological variables on deep neural networks.
There are main two types of neural networks.

1. Normal Neural Networks with 162 input variables.

2. SEL and SEV Neural Networks with 25 input variables.

Note : These two special variables donot take any predefined input. The input to these variables is the predicted value from trained neural networks on 25 variables. The 25 variables are as follows.

1.ANHEDONIA 

2.ANXIOUSNESS 

3.ATTENTION_SEEKING 

4.CALLOUSNESS 

5.DECEITFULNESS 

6.DEPRESSIVITY 

7.DISTRACTIBILITY 

8.ECCENTRICITY 

9.EMOTIONAL_LABILITY 

10.GRANDIOSITY 

11.HOSTILITY 

12.IMPULSIVITY 

13.INTIMACY_AVOIDANCE 

14.IRRESPONSIBILITY 

15.MANIPULATIVENESS 

16.PERCEPTUAL_DYSREGULATION 

17.PERSEVERATION 

18.RESTRICTED_AFFECTIVITY 

19.RIGID_PERFECTIONISM 

20.RISK_TAKING 

21.SEPARATION_INSECURITY 

22.SUBMISSIVENESS 

23.SUSPICIOUSNESS 

24.UNUSUAL_BELIEFS 

25.WITHDRAWAL

The procdure is to train these Neural Networks of 25 variables on normal input. Than use those 25 neural networks to get a input of 25 variables for the training of the special variables i.e. SEL and SEV 



## Gradient Explosion

Due to very difficult dataset. Gradient explosion is very common. While training the neural Network the user may notice NAN in the loss. This can only be avoided by certain conditions:

1. Trying different learning rates.

2. Multiple random intialization of the training.

The Neural Networks on this dataset were trained with 100's of different random initialization and combination of learning rate to achieve the desired result.
If required results are not obtained in terms of correlation/loss. Rerun the algorithm as the starting is randomly initialized. Different runs yeild different results.

## Example

##### This example will be about to train a neural network on normal 162 varaibles:

The data file in ./data/INPUT_DATA.csv is the file given by author that contains the normal 162 varaibles that shall be given at input of the whole training of the neural network. 


```bat
git clone https://github.com/ALI7861111/Multidimensional-Personality-Test-for-Children-MPT-C-.git

cd Multidimensional-Personality-Test-for-Children-MPT-C-

python main.py --input_csv_path './data/INPUT_DATA.csv' --output_csv_path './data/PAG.csv' --learning_rate 0.00001 --varaible_name 'PAG' --OUTPUT_DIR './output/' --MAE_WEIGHTAGE 1 --PAE_WEIGHTAGE 5 --epochs 50 --first_layer_neuron 162 --second_layer_neuron 2048 --Special_variable False --Regularization False 

```

The details for each input variable are as follows :

```
--input_csv_path : Path to the input csv 

--output_csv_path : The output ground truth of the variable 

--learning_rate : The learning_rate of the optimizer

--varaible_name : The name of the variable you want to train

--OUTPUT_DIR : The directory where output models and csv and formula txt are gonna be saved 

--MAE_WEIGHTAGE : The weightage given to the Mean Absolute Error. This can be anything between 0- 10. A hyperparameter

--PAE_WEIGHTAGE : The weightage given to the Pearson Correlation. This can be anything between 0- 10. A hyperparameter

--epochs : The number of epochs to train the model

--first_layer_neuron : The number of nodes/neurons in first layer of neural network. A hyperparameter

--second_layer_neuron : The number of nodes/neurons in second layer of neural network. A hyperparameter

--Special_variable : If special Variable == True than SEL and SEV are gonna be trained. One must ensure that input csv is not same as 162 variable csv

--Regularization ; If true turns on regularization in the neural Network 

```












