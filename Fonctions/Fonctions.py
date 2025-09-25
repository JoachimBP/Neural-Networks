import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D
import time
import numpy as np
import torch
from torch.distributions import Categorical
from scipy.linalg import lu
from scipy.sparse.linalg import svds
from scipy.linalg import eigh as largest_eigh


def extract_layers(main_model, starting_layer_ix, ending_layer_ix):
    # create an empty model
    new_model = Sequential()
    for ix in range(starting_layer_ix, ending_layer_ix + 1):
        curr_layer = main_model.get_layer(index=ix)
        # copy this layer over to the new model
        new_model.add(curr_layer)
    return new_model 

class functional_dimension():

    def __init__(self, model, use_logits = 0): 

        self.model = model
        self.use_logits = use_logits

        layers = model.layers
        l = len(layers)
        if use_logits == 1:
            model = extract_layers(model,0,l-2)
            self.model = model

        
        # Find the architecture of the model
        self.layers = model.layers
        input_shape = model.layers[0].input_shape
        input_shape = list(input_shape[1:len(input_shape)])
        L = 0
        Arch = [input_shape]
        param_num = []

        for layer in (x for x in self.layers if isinstance(x, Dense) or isinstance(x, Conv2D)):
            L += 1
            output_shape = len(layer.output.shape)
            Arch.append(layer.output.shape[1 : output_shape].as_list())

            weights = layer.get_weights()
            weights_num = weights[0].size
            biases_num = weights[1].size
            param_num.append((weights_num, biases_num))
        
        self.arch = Arch
        self.number_layers = L
        self.param_number = param_num


        # Number of parameters of the model
        self.Npar = model.count_params()

        # Function computing the gradients
        @tf.function
        def jacob(x):
            with tf.GradientTape() as tape:
                y = model(x)
            model_gradients = tape.jacobian(y, model.trainable_variables)
            return model_gradients
        
        self.jacob = jacob


        
        
    def get_differential(self, X):
        
        self.stored_svd = 0
        self.stored_norm = 0
        self.stored_frob = 0
        self.stored_nuc_norm = 0
        Nsample = X.shape[0]


        jacob = self.jacob

        arch = self.arch
        L = self.number_layers

        # Creation of the matrix that will be the jacobian
        Gamma = np.zeros((Nsample*arch[L][0],self.Npar))
    
        batch_size = min(1000,Nsample,self.Npar)    # We are going to complete Gamma one batch of inputs at a time

        i = 0   # input index
        while i <= Nsample-batch_size:
            line = i * arch[L][0]    # line index for Gamma

            model_gradients = jacob(X[i:i+batch_size])

            column = 0     # column index for Gamma

            # Gradients with respect to the weights
            for cpt in range(L):
                weights_num = self.param_number[cpt][0]
                Gamma[line:line+batch_size*arch[L][0],column:column+weights_num] = model_gradients[2*cpt].numpy().reshape(batch_size*arch[L][0],weights_num)
                column += weights_num
            
            # Gradients with respect to the biases
            for cpt in range(L):
                biases_num = self.param_number[cpt][1]
                Gamma[line:line+batch_size*arch[L][0],column:(column+biases_num)] = model_gradients[2*cpt+1].numpy().reshape(batch_size*arch[L][0],biases_num)
                column += biases_num
            i += batch_size
        


        self.Gamma = Gamma

        return Gamma
    


    def compute_svd(self, use_torch = 1):

        Gamma = self.Gamma

        if self.stored_svd == 0:    # Check if we ave not computed the SVD already
            if use_torch == 1:
                Gamma_torch = torch.from_numpy(Gamma)
                svd = torch.linalg.svdvals(Gamma_torch)
                svd = svd.numpy()
            else:
                svd = np.linalg.svd(Gamma, compute_uv=False)  

            self.svd = svd
            self.stored_svd = 1
        
        svd = self.svd
        
        return svd
    
    
  
    def compute_rank(self, use_torch = 1, rtol = 0):

        # The rank is here computed using the SVD
        
        Gamma = self.Gamma


        if self.stored_svd == 0:    # Check if we ave not computed the SVD already
            self.compute_svd(use_torch = use_torch)
            self.stored_svd = 1
        svd = self.svd

        # Computation of the threshold, similarly as numpy.linalg.matrix_rank:
        (m,n) = Gamma.shape
        if rtol == 0:
            eps = np.finfo(svd.dtype).eps   # Value used by default by numpy.linalg.matrix_rank
            tol =  svd.max() * max(m,n) * eps
        else:
            tol = svd.max() * rtol

        # Computation of the rank:
        r = np.sum(svd >= tol)

        self.rank = r
        
        return r
    
    
    def compute_frob(self):

        Gamma = self.Gamma

        if self.stored_frob == 0:   # Avoid computing the Frobenius norm twice
            if self.stored_svd == 1:    # If we have previously computed the SVD, it is faster to use it
                frob = np.linalg.norm(self.svd, 2)
            else:                         # Otherwise, it is faster to compute the Frobenius directly
                Gsquare = Gamma.T@Gamma
                frob2 = np.trace(Gsquare)
                frob = np.sqrt(frob2)
            self.frob = frob
            self.stored_frob = 1

        frob = self.frob


        return(frob)


    def compute_norm(self):

        Gamma = self.Gamma

        if self.stored_norm == 0:   # First check if we have not computed the norm already
            if self.stored_svd == 1:    # If we already have the svd, then we have the norm
                norm = np.abs(self.svd[0])
            else:                       # Otherwise, we can compute it directly
                [norm] = svds(Gamma, k=1,return_singular_vectors = False)
            self.norm = norm
            self.stored_norm = 1

        norm = self.norm

        return norm


    def compute_stable_rank(self):
        # The stable rank is the ratio between the squared Frobenius norm and the squared operator norm


        if self.stored_frob == 0:
            self.compute_frob()

        frob2 = (self.frob)**2

        if self.stored_norm == 0:
            self.compute_norm()

        norm2 = (self.norm)**2
        stable_rank = frob2/norm2
        self.stable_rank = stable_rank


        return stable_rank


    def compute_nuc_norm(self, use_torch = 1):


        if self.stored_nuc_norm == 0:   # Check if we have not computed the nuclear norm already
            if self.stored_svd == 0:    # To compute the nuclear norm one needs the SVD
                self.compute_svd(use_torch = use_torch)

            svd = self.svd
            N_nuc = np.linalg.norm(svd, ord=1)
            self.nuc_norm = N_nuc
            self.stored_nuc_norm = 1

        nuc_norm = self.nuc_norm

        return nuc_norm
    

    def compute_robust_rank(self):
        # The robust rank is the normalized nuclear norm

        # Get the nuclear norm
        if self.stored_nuc_norm == 0:   # Check if we have not computed the nuclear norm already
            self.compute_nuc_norm()
        nuc_norm = self.nuc_norm

        # Get the operator norm
        if self.stored_norm == 0:       # Check if we have not computed the operator norm already
            self.compute_norm()
        norm = self.norm

        # Compute the robust rank
        robust_rank = nuc_norm/norm     
        self.robust_rank = robust_rank


        return robust_rank
    
    

    def compute_eff_rank(self, use_torch = 1):


        # Get the SVD
        if self.stored_svd == 0:          # Check if we have already computed the svd
            self.compute_svd(use_torch = use_torch)

        svd = self.svd

        # Get the nuclear norm
        if self.stored_nuc_norm == 0:   # Check if we have already computed the nuclear norm
            nuc_norm = np.linalg.norm(svd, ord=1)
            self.nuc_norm = nuc_norm
            self.stored_nuc_norm = 1

        nuc_norm = self.nuc_norm

        # Transform the singular values as a probability vector
        normalized_svd = svd/nuc_norm

        # Compute the Shannon entropy
        p_tensor = torch.Tensor(normalized_svd)
        eff_rank = Categorical(probs = p_tensor).entropy()
        eff_rank = eff_rank.numpy()

        return eff_rank
    
    