import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers


class train_NN_sel_sev:
    def __init__(self,input_csv_path,output_varaible_csv_path,
                 learning_rate,decay,epochs,variable_name, 
                 output_dir, MAE_WEIGHTAGE, PAE_WEIGHTAGE, 
                 neurons_second_layer,neuron_first_layer,
                 regularization_varaible):
    
        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.variable_name = variable_name
        # The varaible name should be SEL or SEV when importing this class or using objects from this class 
        self.learning_rate  = learning_rate
        self.epochs  = epochs
        self.decay = decay
        self.pae_weightage = PAE_WEIGHTAGE
        self.mae_weightage = MAE_WEIGHTAGE
        self.input_csv_path = input_csv_path
        self.output_variable_csv_path = output_varaible_csv_path
        self.number_nodes_second_layer = neurons_second_layer
        self.number_nodes_first_layer = neuron_first_layer
        self.output_dir = output_dir
        self.regularization = regularization_varaible

        self.x = pd.read_csv(str(self.input_csv_path),index_col=None)
        self.y = pd.read_csv(str(self.output_variable_csv_path),index_col=None)
        self.horizontal_stack = pd.concat([self.x, self.y], axis=1)
        # chnage sev to sel if needed
        self.horizontal_stack[str(self.variable_name)].replace("", np.nan, inplace=True )
        self.horizontal_stack.dropna(subset=[str(self.variable_name)], inplace=True )


        self.horizontal_stack = self.horizontal_stack.sample(frac=1.0, replace=True, random_state=42 )
        self.x = self.horizontal_stack.iloc[:,0:25]
        self.y = self.horizontal_stack.iloc[:,25:26]
        self.x_train, self.x_validate, self.y_train, self.y_validate = train_test_split(self.x, self.y, test_size=0.2)
        self.x_test, self.x_validate, self.y_test, self.y_validate = train_test_split(self.x_validate, self.y_validate, test_size=0.5)

        # You can view the processed data by uncommneting the lines below 
        # x_train.to_excel('C:/Users/HOME/Desktop/data/'+str(B)+'/'+str(B)+'_training_input.xlsx')
        # y_train.to_excel('C:/Users/HOME/Desktop/data/'+str(B)+'/'+str(B)+'_training_output.xlsx')
        # x_validate.to_excel('C:/Users/HOME/Desktop/data/'+str(B)+'/'+str(B)+'_validating_input.xlsx')
        # y_validate.to_excel('C:/Users/HOME/Desktop/data/'+str(B)+'/'+str(B)+'_validating_output.xlsx')
        # x_test.to_excel('C:/Users/HOME/Desktop/data/'+str(B)+'/'+str(B)+'_testing_input.xlsx')
        # y_test.to_excel('C:/Users/HOME/Desktop/data/'+str(B)+'/'+str(B)+'_testing_output.xlsx')

 
        # COMMENT BELOW LOSS TO STOP COMBINED LOSS OF PEARSON AND MAE
        #return  abs(loss)


    def train_model(self):

        mcp_save = tf.keras.callbacks.ModelCheckpoint(str(self.output_dir)+str(self.variable_name)+'.h5',save_best_only=True, monitor='val_loss', mode='min')
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                        self.learning_rate,
                        decay_steps=100,
                        decay_rate=self.decay,
                        staircase=True)

        if self.regularization == False:
            self.model = keras.Sequential([keras.layers.Dense(self.number_nodes_first_layer, activation=None, input_shape=[25]),
                                keras.layers.Dense(self.number_nodes_second_layer),
                                keras.layers.Dense(1, activation=tf.nn.relu)])
        if self.regularization == True:
            self.model = keras.Sequential([keras.layers.Dense(self.number_nodes_first_layer, activation=None, input_shape=[25]),
                                keras.layers.Dense(self.number_nodes_second_layer, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                bias_regularizer=regularizers.l2(1e-4),
                                activity_regularizer=regularizers.l2(1e-5) ,activation=tf.nn.relu),
                                keras.layers.Dense(1, activation=tf.nn.relu)])            
        # above is the model for deep learning
        # input first layer 162 nodes
        # first hidden layer with 2048 nodes
        # regularization technique L1 and L2 implemented to avoid overfitting
        # last layer with 1 node for the output feature


        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.model.compile(loss=self.correlation,optimizer=optimizer)
        #history = model.fit(x_train,y_train,epochs=e,callbacks=[mcp_save],validation_split=.1)#,validation_data=(x_validate,y_validate))
        self.history = self.model.fit(self.x_train,self.y_train,epochs=self.epochs,callbacks=[mcp_save],validation_data=(self.x_validate,self.y_validate))


    def get_test_data_results(self):

        self.model = keras.models.load_model(str(self.output_dir)+str(self.variable_name)+'.h5', compile=False)
        output = self.model.predict(self.x_test)
        ground_truth =  self.y_test.reset_index(drop=True)
        df = pd.DataFrame(data=output)
        df = df.reset_index(drop=True)
        LOSS = abs(df.iloc[:,0]-ground_truth.iloc[:,0])
        LOSS = LOSS.values.sum()
        LOSS = LOSS / len(df)
        df = pd.concat([df, ground_truth], axis = 1,sort=False)
        print('The loss on testing is as follows')
        print(LOSS)
        print('The correlation factor Test is as follows')
        print(df.corr())
        self.test_data_pearson_corr = df.corr()


    def create_prediction_csv(self):
        self.model = keras.models.load_model(str(self.output_dir)+str(self.variable_name)+'.h5' , compile=False)
        output = self.model.predict(self.x)
        df = pd.DataFrame(data=output)
        df.to_csv(str(self.output_dir)+str(self.variable_name)+'.csv')