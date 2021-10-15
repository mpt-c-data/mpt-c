import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from tensorflow.keras import regularizers


class train_NN :
    def __init__(self,input_csv_path,output_varaible_csv_path,
                 learning_rate,decay,epochs,variable_name, 
                 output_dir,MAE_WEIGHTAGE, PAE_WEIGHTAGE,
                 neurons_second_layer, neuron_first_layer,
                 regularization_varaible):
                         
        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.variable_name = variable_name#'INTIMACY'
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.decay_rate = decay
        self.mae_weightage = MAE_WEIGHTAGE
        self.pae_weightage = PAE_WEIGHTAGE
        self.number_nodes_second_layer = neurons_second_layer
        self.number_nodes_first_layer = neuron_first_layer
        self.output_dir = output_dir
        self.regularization = regularization_varaible
        # x is the input features
        # my local path is C:/Users/HOME/Desktop/data for the data folder give your as required
        # INPUT.csv has features A1-J16
        # There are 2 feature files A1-J16 and other 25 for SEL ans SEV
        # THE input features for sel and sev are saved in csv named INPUT2.csv
        self.x = pd.read_csv(input_csv_path,index_col=None)
        # y is the out put feature
        self.y = pd.read_csv(output_varaible_csv_path,index_col=None)
        self.horizontal_stack = pd.concat([self.x, self.y], axis=1)
        #A = list(horizontal_stack.columns.values.tolist()) 
        # if using input 2 change the 162 to 25
        #A= A[162]
        # This code of line removes observations with no values recorded of output feature
        # horizontal_stack[A].replace("", np.nan, inplace=True )
        # horizontal_stack.dropna(subset=[A], inplace=True )
        # horizontal_stack.iloc[:,162:163].plot.hist(bins=20, alpha=0.5)
        # plt.title('distribution before resampling')
        # The line below takes sampling with replacement = bootstraping

        self.horizontal_stack = self.horizontal_stack.sample(frac=1.0, replace=True, random_state=42 )
        # horizontal_stack.iloc[:,162:163].plot.hist(bins=20, alpha=0.5)
        # plt.title('distribution after resampling')
        # this line below takes random samples for train , validate and testing 
        self.train,self.validate,self.test = self.train_validate_test_split(self.horizontal_stack,0.8, 0.1,0.1,seed=42)

        # saving the testing validation and training files with excel format
        #self.train.to_excel('C:/Users/HOME/Desktop/data/'+str(B)+'/'+str(A)+'_training.xlsx')
        #self.validate.to_excel('C:/Users/HOME/Desktop/data/'+str(B)+'/'+str(A)+'_validating.xlsx')
        #self.test.to_excel('C:/Users/HOME/Desktop/data/'+str(B)+'/'+str(A)+'_testing.xlsx')
        # if usng INPUT2.csv change 0:162 to 0:25
        # you may have to run this loop multiple times for better results 
        # sometimes may get NaN then run loop for better initialization of weights
        # SOMETIME NEED TO RESTART THE KERNEL
        self.x_train = self.train.iloc[:,0:162]
        # if usng INPUT2.csv change 162:163 to 25:26
        self.y_train = self.train.iloc[:,162:163]

        self.x_validate = self.validate.iloc[:,0:162]
        # if usng INPUT2.csv change 162:163 to 25:26
        self.y_validate = self.validate.iloc[:,162:163]
                                                    
        self.x_test     = self.test.iloc[:,0:162]
        # if usng INPUT2.csv change 162:163 to 25:26
        self.y_test     = self.test.iloc[:,162:163]
        # checkpoint makes sure that model with best accuracy on test data is saved

    def correlation(self,x, y):
        MAE = self.mae(x, y)
        mx = tf.math.reduce_mean(x)
        my = tf.math.reduce_mean(y)
        xm, ym = x-mx, y-my
        r_num = tf.math.reduce_mean(tf.multiply(xm,ym))        
        r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
        a = 1-(r_num/r_den)
        loss = abs(a)*self.pae_weightage + self.mae_weightage*MAE
        #loss = a + 10*MAE
        #return a
        return  loss

    def train_validate_test_split(self,df, train_percent, validate_percent,test_percent,seed):
        np.random.seed(seed)
        perm = np.random.permutation(df.index)
        m = len(df.index)
        train_end    = int(train_percent*m)
        validate_end = int(validate_percent*m) + train_end
        test_end     = int(test_percent*m) + validate_end
        train = df.iloc[perm[:train_end]]
        validate = df.iloc[perm[train_end:validate_end]]
        test = df.iloc[perm[validate_end:test_end]]
        return train, validate, test



    def train_model(self):
        mcp_save = tf.keras.callbacks.ModelCheckpoint(str(self.output_dir)+str(self.variable_name)+'.h5',save_best_only=True, monitor='val_loss', mode='min')
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(   self.learning_rate,
                                                                        decay_steps=100,
                                                                        decay_rate=self.decay_rate,
                                                                        staircase=True)
# made a initial learning rate with a decay to get the best performance
        if self.regularization == False:
            self.model = keras.Sequential([keras.layers.Dense(self.number_nodes_first_layer, activation=None, input_shape=[162]),
                                        keras.layers.Dense(self.number_nodes_second_layer ,activation=tf.nn.relu),
                                        keras.layers.Dense(1, activation=tf.nn.relu)])
        if self.regularization == True:
            self.model = keras.Sequential([ keras.layers.Dense(self.number_nodes_first_layer, activation=None, input_shape=[162]),
                                            keras.layers.Dense(self.number_nodes_second_layer, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                            bias_regularizer=regularizers.l2(1e-4),activity_regularizer=regularizers.l2(1e-5) ,activation=tf.nn.relu),
                                            keras.layers.Dense(1, activation=tf.nn.relu)])

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
# above is the model for deep learning
# input first layer 162 nodes
# first hidden layer with 2048 nodes
# last layer with 1 node for the output feature
        self.model.compile(loss=self.correlation,optimizer=optimizer)
        self.history = self.model.fit(self.x_train,self.y_train,epochs=self.epochs,callbacks=[mcp_save],validation_data=(self.x_validate,self.y_validate))


    def get_test_data_results(self):
        self.model = keras.models.load_model(str(self.output_dir)+str(self.variable_name)+'.h5')
        output = self.model.predict(self.x_test)
        ground_truth = self.y_test.reset_index(drop=True)
        df = pd.DataFrame(data=output)
        df = df.reset_index(drop=True)
        LOSS = abs(df.iloc[:,0]-ground_truth.iloc[:,0])
        LOSS = LOSS.values.sum()
        LOSS = LOSS / len(df)
        df = pd.concat([df, ground_truth], axis = 1,sort=False)
        print('The MAE loss on testing is as follows')
        print(LOSS)
        print('The correlation factor Test is as follows')
        print(df.corr())
        self.test_data_pearson_corr = df.corr()

    def create_prediction_csv(self):
        self.model = keras.models.load_model(str(self.output_dir)+str(self.variable_name)+'.h5')
        output = self.model.predict(self.x)
        df = pd.DataFrame(data=output)
        df.to_csv(str(self.output_dir)+str(self.variable_name)+'.csv')


