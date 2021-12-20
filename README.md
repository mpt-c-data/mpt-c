# MPT-C // Multidimensional Personality Test for Children
Development and Deep Learning-based Validation of a New Projective Test for Children. 



This project contains the implementation of training physcological variables on deep neural networks.
There are two main types of neural networks.

1. Normal Neural Networks with 162 input variables.

2. SEL and SEV variables' Neural Networks with 25 input variables*.

*Note : These two special variables do not take any predefined input. The input to these variables is the predicted value from trained neural networks on 25 variables. The 25 variables are as follows. ANHEDONIA, ANXIOUSNESS, ATTENTION_SEEKING, CALLOUSNESS, DECEITFULNESS, DEPRESSIVITY, DISTRACTIBILITY, ECCENTRICITY, EMOTIONAL_LABILITY, GRANDIOSITY, HOSTILITY, IMPULSIVITY, INTIMACY_AVOIDANCE, IRRESPONSIBILITY, MANIPULATIVENESS, PERCEPTUAL_DYSREGULATION, PERSEVERATION, RESTRICTED_AFFECTIVITY, RIGID_PERFECTIONISM, RISK_TAKING, SEPARATION_INSECURITY, SUBMISSIVENESS, SUSPICIOUSNESS, UNUSUAL_BELIEFS, WITHDRAWAL. The procedure is to train these Neural Networks of 25 variables on normal input, then, to use these 25 neural networks to get an input of 25 variables for the training of the special variables, i.e., SEL and SEV.


## Gradient Explosion

Due to very difficult dataset, gradient explosion is very common. While training the neural Network, the user may notice NAN in the loss. This can be avoided only by certain conditions:

1. Trying different learning rates.

2. Multiple random intialization of the training.

The Neural Networks on this dataset were trained with 100's of different random initialization and combination of learning rate to achieve the desired result.
If required results are not obtained in terms of correlation/loss. Rerun the algorithm as the starting is randomly initialized. Different runs yeild different results.

## Example

##### This example will be about to train a neural network on normal 162 varaibles:

The data file in ./data/INPUT_DATA.csv is the file that contains the normal 162 varaibles/inputs that shall be given at input of the whole training of the neural network. 


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




## Working sample 

The default configuration in the repository is to train ANT variable. After clonning the repository. Just run

```bat

python main.py

```

The output shall be generated inside the output folder :

1. Formula text file 
2. Model weights
3. Predicted csv
4. The Pearson correlation coefficient shall be printed in the terminal screen.








