from src.train_NN import *
from src.train_NN_SEL_SEV import *
import argparse
from src.Formula_creator import file_creator


parser = argparse.ArgumentParser(description="Machine Learning")


parser.add_argument("--input_csv_path", type=str, default='./data/INPUT_DATA.csv',
                    help='Path to the input csv ')
parser.add_argument('--output_csv_path', type= str, default='./data/PAG.csv', 
                    help='The output ground truth of the variable')
parser.add_argument('--learning_rate', type=float, default=0.01,
                    help='The learning_rate of the optimizer')
parser.add_argument('--varaible_name', type=str, default='PAG',
                    help='The name of the variable you want to train')
parser.add_argument('--OUTPUT_DIR', type=str, default='./output/',
                    help='The directory where output models and csv are gonna be saved ')
parser.add_argument('--MAE_WEIGHTAGE', type=float, default=0,
                    help='Mae weightage between 0 - 10 ')
parser.add_argument('--PAE_WEIGHTAGE', type=float, default=10,
                    help='Mae weightage between 0 - 10 ')
parser.add_argument('--epochs', type=int, default=200,
                    help='The number of epochs ')
parser.add_argument('--first_layer_neuron', type=int, default=162,
                    help='The number of neurons in the first layer of Neural Network ')               
parser.add_argument('--second_layer_neuron', type=int, default=2048,
                    help='The number of neurons in the second layer of Neural Network ')
parser.add_argument('--Special_variable', type=bool, default=False,
                    help='This specifies if the user want to train SEL and SEV variables ')
parser.add_argument('--Regularization', type=bool, default=False,
                    help='This specifies if the user wants regularization implemented in the Neural Network')



args = parser.parse_args()
Input_path = args.input_csv_path
Output_path = args.output_csv_path
lr = args.learning_rate
Variable_name = args.varaible_name
MAE_WT = args.MAE_WEIGHTAGE
PAE_WT = args.PAE_WEIGHTAGE
Special_variable = args.Special_variable
epochs =  args.epochs
output_dir = args.OUTPUT_DIR
first_layer_neurons = args.first_layer_neuron
second_layer_neurons= args.second_layer_neuron
regularization_variable = args.Regularization

if Special_variable == False:


    object = train_NN( input_csv_path = Input_path ,output_varaible_csv_path = Output_path,
                        learning_rate = lr ,decay = 0.99 ,epochs = epochs ,
                        variable_name = Variable_name , MAE_WEIGHTAGE = MAE_WT , PAE_WEIGHTAGE = PAE_WT, 
                        output_dir=output_dir ,neurons_second_layer = second_layer_neurons,
                        neuron_first_layer =first_layer_neurons, regularization_varaible = regularization_variable)
else:

    object = train_NN_sel_sev( input_csv_path = Input_path ,output_varaible_csv_path = Output_path,
                               learning_rate = lr ,decay = 0.99 ,epochs = epochs ,
                               variable_name = Variable_name , MAE_WEIGHTAGE = MAE_WT , PAE_WEIGHTAGE = PAE_WT, 
                               output_dir=output_dir,neurons_second_layer = second_layer_neurons,
                               neuron_first_layer =first_layer_neurons, regularization_varaible = regularization_variable)


object.train_model()
object.get_test_data_results()
object.create_prediction_csv()
object2 = file_creator(model_path = str(output_dir)+str(Variable_name)+'.h5', path_to_forumla_file = str(output_dir)+str(Variable_name)+'.txt')


object2.formula_creator()
