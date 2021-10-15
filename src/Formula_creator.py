import tensorflow as tf
from tensorflow import keras


class file_creator:
    def __init__(self,model_path,path_to_forumla_file):
        self.model_path = model_path
        self.filename = path_to_forumla_file


    def formula_creator(self):
    # os.mknod('C:/Users/HOME/Desktop/data/FORMULAS/'+str(A)+'.txt')
        self.model = keras.models.load_model(str(self.model_path))

        f = open(str(self.filename),"w+")
        f.write(f'the formula for the varaible is written sequentially below\n')
        f.write(f'Example : k = a+b+c\n')
        f.write(f'it will be written as\n')
        f.write(f'k = a\n')
        f.write(f'k = k + a\n')
        f.write(f'k = k + c\n')    
        f.write(f'k = the relu function used in varaibles i.e u = relu(variable) \n')   
        f.write(f'it means if value of f is , 0 relu(f) = 0 otherwise it is f the same value \n')       
        for layerNum, layer in enumerate(self.model.layers):
            weights = layer.get_weights()[0]
            biases = layer.get_weights()[1]
        
            for toNeuronNum, bias in enumerate(biases):
                f.write(f'L{layerNum+1}N{toNeuronNum} = {bias}\n')
        
            for fromNeuronNum, wgt in enumerate(weights):
                for toNeuronNum, wgt2 in enumerate(wgt):
                    if layerNum == 0:
                        f.write(f'L{layerNum+1}N{toNeuronNum} = L{layerNum+1}N{toNeuronNum} + Input{layerNum}N[{fromNeuronNum}]*{wgt2}\n')
                    elif layerNum==1:
                        f.write(f'L{layerNum+1}N{toNeuronNum} = L{layerNum+1}N{toNeuronNum} + L{layerNum}N{fromNeuronNum}*{wgt2}\n')
                    else:
                        f.write(f'L{layerNum+1}N{toNeuronNum} = L{layerNum+1}N{toNeuronNum} + relu(L{layerNum}N{fromNeuronNum})*{wgt2}\n')
        f.close()
