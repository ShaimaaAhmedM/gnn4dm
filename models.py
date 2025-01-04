import torch

import torch_geometric

import torch.nn.functional as F
from torch.nn import Parameter
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv


import math

from utils import cosine_self_similarity, inv_softplus

EPS = 1e-15

def reset(value):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    #else The reset function recursively traverses containers to reset their child layers.
    else:
        #Checks for children value.children() in a container of modules like torch.nn.Sequential
        #torch.nn.Sequential is a simple container that holds a sequence of layers, applied one after another.
        #calls reset for each child if value has 'children' method
        #If value doesn’t have children (i.e., it’s not a container module), this returns an empty list.
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)

"""
The base class torch.nn.Linear class implements a fully connected (dense) layer. Its main features are
Weight Matrix: Multiplies the input by a learnable weight matrix.
Bias Vector: Adds a learnable bias term (optional).
Use torch.nn.Linear when you need:
Feature transformation: Map input features (not raw input in this case) to output features.
Linear relationships: Capture dependencies between input and output.
Dimensionality adjustment: Reduce or expand feature dimensions.
Inter-layer connections: Connect layers in a neural network.
Output predictions: Generate logits or values for regression, classification, or other tasks.
Why Use torch.nn.Linear in GNNs?In this GNN example As an output layer to map graph features (module_representation_channels) to task-specific outputs in PositiveLinear.
"""
class PositiveLinear(torch.nn.Linear):
    #constructor Calls the parent class constructor to initialize the linear layer.
    #Refers to the processed features produced by the GNN and passed to the task-specific linear layer.
    def __init__(self, in_features: int, out_features: int, bias: bool = True, ids: list = None, 
                 device = None, dtype = None):
        #When super() is used in the __init__ method It calls the constructor (__init__) of the parent class (torch.nn.Linear), ensuring that the attributes and behavior of the parent class are properly initialized in the child class.
        super(PositiveLinear, self).__init__(in_features, out_features, bias, device, dtype)

        assert out_features == len(ids), "Number of ids doesn't match with the number of out_features."
        
       
        #storing ids as self.id_list allows the layer to associate its outputs with meaningful terms or pathways.
        self.id_list = ids
        #populate mappings of pathways "value" to indices of the weight parameter matrix {'Pathway1': 0, 'Pathway2': 1,..}
        #enumerate function generates pairs of (index, value) for each element in ids.
        self.id_to_position = {value: index for index, value in enumerate(ids)} if ids is not None else None
        #Redundant check to ensure ids matches the number of out_features.
        if ids != None:
            assert len(ids) == out_features, "The number of ids doesn't match with the number of output features."

    # Performs the forward pass and ensures weights remain non-negative.
    def forward(self, x):
        #apply the constraint on weights it ensuring all weights in the layer are non-negative by clamping values below 0.0 to 0.0.
        #This maintains constraints for specific use cases, such as modeling pathways or gene expressions that always have to be positive to be meaningful.
        self.weight.data.clamp_(min=0.0)
        #Calls the parent class's forward method to compute y=xW(transpose)+b
        return super(PositiveLinear, self).forward(x)

    #These norms are often used for regularization, which adds a penalty to the loss function in training based on the magnitude of the weights
    #L1 Norm: Sum of the absolute values of the positive weights.Promotes sparsity, making many weights zero
    #Useful when you want the model to focus on a smaller subset of important features. Example: Feature selection tasks.
    #L2 Norm: Euclidean magnitude (square root of sum of squares) of the positive weights.
    #Encourages small, evenly distributed weights.Helps prevent overfitting by reducing the impact of individual weights.Example: General regression or classification tasks.
    def l1_l2_losses(self):
        #torch.where(condition, x, y)This function works like an "element-wise if-else"
        #condition results in tensor of boolean values (True or False) specifying the condition for each element.
        #If the condition is True for a particular element, take the value from x The tensor x has original weights.
        #Otherwise, take the value from y, y tensor is the same size tensor as x but all zeros.
        positive_weights = torch.where(self.weight > 0, self.weight, torch.zeros_like(self.weight))

        #torch.norm calculates l1 or l2 based on p=?
        l1 = torch.norm(positive_weights, p=1)
        l2 = torch.norm(positive_weights, p=2)

        return l1, l2

    
#this class defines the GNN model as a PyTorch module, allowing it to: Integrate seamlessly into PyTorch’s training framework.
#We use torch.nn.Module as the base class for all neural network models in PyTorch. It provides a framework for defining, managing, and training deep learning models by handling critical aspects like parameter management, forward propagation, and model evaluation.
class GNN(torch.nn.Module):
    #in_channels in GNN refers to the raw input features at the start of the model.
    #The hidden_channels_before_module_representation parameter refers to the sizes of the intermediate layers (hidden layers) in the Graph Neural Network (GNN) before it produces the final output, which is called the module representation.
    #if hidden_channels_before_module_representation is a single number like 128, it means there is only one hidden layer before the module representation layer, this hidden layer had 128 features.
    #this simplifies the architecure to input->hidden layer-> module representation
    #if batchnorm=True, adds a BatchNorm1d layer after the hidden layer.
    #module_representation_channels: Size of the final GNN output (number of features per node in the module representation).
    #out_models: Dictionary of task-specific output models (e.g., PositiveLinear layers).
    #batchnorm: Whether to apply batch normalization to stabilize training.
    #transform_probability_method: Method for transforming module representations into probabilities (tanh in call).
    #type: The type of GNN layer (e.g., GCN, GAT, MLP).
    #threshold a parameter that scales or normalizes GNN outputs before transforming them into probabilities. Ensures outputs are mapped into a valid range (e.g., 0 to 1),
    def __init__(self, 
                 in_channels, hidden_channels_before_module_representation, module_representation_channels, out_models : dict, 
                 dropout = 0.0, batchnorm = False, transform_probability_method : str = "tanh", threshold = 1.0,
                 type : str = "GCN" ):
        super(GNN, self).__init__()

        #self.convs creates a container to hold the GNN layers (graph convolution layers or linear layers).
        #ModuleList is used because it allows us to dynamically add layers and ensures they are properly registered as part of the model.
        self.convs = torch.nn.ModuleList()
        #sefl.batchnorms_convs creates a container for batch normalization layers.
        #Batch normalization is used after each GNN layer (if enabled) to normalize the output of the layer, making training more stable.
        self.batchnorms_convs = torch.nn.ModuleList()
        #The self.output_models dictionary connects the module representation output of the GNN to the specific tasks (pathways) you want to analyze or predict.
        # A dictionary that stores the output models (e.g., PositiveLinear layers) for each pathway
        # torch.nn.ModuleDict Registers the models from out_models into a single dictionary, ensuring they are part of the GNN and will be updated during training.            
        self.output_models = torch.nn.ModuleDict({ key: out_model for key, out_model in out_models.items() })

        self.type = type

        # this block creates the intermediate (hidden) layers for the GNN based on the sizes specified in hidden_channels_before_module_representation.
        # clean zeros from hidden channel counts
        if 0 in hidden_channels_before_module_representation: hidden_channels_before_module_representation.remove(0)
        
        # 1. GNN layers from input to module_representation
        channels = in_channels
        for hidden_channel in hidden_channels_before_module_representation:
            if type == "GCN":
                #Checks the type of GNN layer specified (e.g., GCN, MLP, etc.) and appends the corresponding layer to self.convs
                #A standard Graph Convolutional Network (GCN) layer that aggregates features from neighboring nodes.
                #improved=True: Uses an improved version of GCN with better accuracy.
                #cached=True: Speeds up repeated computations for static graphs.
                self.convs.append( torch_geometric.nn.GCNConv(channels, hidden_channel, improved=True, cached=True) )
            elif type == "MLP":
                self.convs.append( torch.nn.Linear(channels, hidden_channel) )
            elif type == "GAT":
                self.convs.append( torch_geometric.nn.GATv2Conv(channels, hidden_channel) )
            elif type == "SAGE":
                self.convs.append( torch_geometric.nn.SAGEConv(channels, hidden_channel) )
            if batchnorm:
                self.batchnorms_convs.append( torch.nn.BatchNorm1d(hidden_channel) )
            #Updates the number of input features for the next layer to match the output size of the current layer.
            channels = hidden_channel
        
        #this block adds the Final GNN Layer
        #This is the last step before transforming data into pathway-specific predictions.
        #This layer transforms channels-dimensional features into module_representation-dimensional features for each node.
        if type == "GCN":
            self.conv_last = torch_geometric.nn.GCNConv(channels, module_representation_channels, improved=True, cached=True)
        elif type == "MLP":
            self.conv_last = torch.nn.Linear(channels, module_representation_channels)
        elif type == "GAT":
            self.conv_last = torch_geometric.nn.GATv2Conv(channels, module_representation_channels)
        elif type == "SAGE":
            self.conv_last = torch_geometric.nn.SAGEConv(channels, module_representation_channels)

        channels = module_representation_channels
        #Dropout: Adds regularization to the model by randomly dropping some features during training
        #Example: dropout = 0.5 means 50% of features will be randomly set to zero.
        self.dropout = dropout

        #Threshold: Controls how the final GNN outputs are transformed into probabilities.
        self.transform_probability_method = transform_probability_method
        if threshold == 'auto':
            if self.transform_probability_method == 'tanh':
                self.threshold = torch.nn.Parameter(torch.tensor(inv_softplus(1.0)), requires_grad = True)
            else:
                self.threshold = torch.nn.Parameter(torch.tensor(inv_softplus(0.5)), requires_grad = True)
        else:
            self.threshold = torch.nn.Parameter(torch.tensor(inv_softplus(threshold)), requires_grad = False)

        #This line creates a string representation of the model’s architecture, called _short_name, which provides a concise summary of the GNN’s structure.
        #an example of the output string could be self._short_name = "GCN_10-[64-128]--Modules:32--LIN[Pathway1|Pathway2]"            
        self._short_name = f"{self.type}_{in_channels}-[{'-'.join([str(h) for h in hidden_channels_before_module_representation])}]--Modules:{module_representation_channels}--LIN[{'|'.join([k for k in out_models])}"
    
    #This method initializes or resets the parameters (weights and biases) of all layers in the GNN.
    #Ensures all layers start with random, properly initialized weights.
    #Proper initialization (e.g., orthogonal) helps prevent issues like vanishing or exploding gradients.
    #Allows consistent re-initialization during experiments.
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
            if self.type == "GCN": 
                #Ensures that the weight matrix is orthogonal (its columns are linearly independent).
                #Helps stabilize training, especially for deep models.
                torch.nn.init.orthogonal_( conv.lin.weight )        # orthogonal initialization
        self.conv_last.reset_parameters()
        if self.type == "GCN":
            torch.nn.init.orthogonal_( self.conv_last.lin.weight )  # orthogonal initialization
        #Loops through all task-specific output models (PositiveLinear layers) and calls their reset_parameters() method.
        for ll in self.output_models.values():
            ll.reset_parameters()
        for bn in self.batchnorms_convs:
            bn.reset_parameters()

    #Defines how data flows through the GNN during the forward pass (inference or training).
    #edge_index is a tensor of size (2, num_edges) that specifies the graph's connections (edges).row 1 is nodeA connected to node B in row 2 for example same column index.
    #A tensor of size (num_nodes, in_channels) containing the input features for each node.
    #output: module_representation(The learned node embeddings) and y_preds(Predictions for each pathway given x).
    #Predictions for each pathway, indicating how strongly each gene in the network is associated with the genes that belong to a specific pathway.
    #The predicted score for how strongly the graph node (gene) contributes to the specific genes in the pathway.
    #A high value means the graph node (gene) has a strong connection or relevance to the pathway column (gene in the pathway).
    #The model is predicting the extent to which every gene in the graph is associated with specific genes in the pathway.
    #Biological Insight:Even genes not explicitly in the pathway might have indirect effects or associations due to their graph connections.
    #The y_preds structure allows a detailed analysis of how all graph genes contribute to specific pathways, not just the genes explicitly in the pathway.
    def forward(self, x, edge_index):
        #Iterate Over GNN hidden Layers seld.convs
        for i in range(len(self.convs)):
            if self.dropout != 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            if self.type == "MLP":
                x = self.convs[i](x) 
            else:
                #For graph-based layers (e.g., GCNConv), the graph structure (edge_index) is used.
                x = self.convs[i](x, edge_index) 
            x = F.relu(x)
            if i < len(self.batchnorms_convs):
                x = self.batchnorms_convs[i](x)
       #apply dropout to the final layer    
        if self.dropout != 0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.type == "MLP":
            x = self.conv_last(x)
        else:
            x = self.conv_last(x, edge_index)

        # computing the module representations by applying a softplus activation (to enforce non-negative values)
        #This is important for some tasks where negative values are not meaningful.
        x = F.softplus(x)
        # this leads to the module representations which will be returned
        module_representation = x

        # transforming the module representations into probabilities 
        # x = tanh( k * x )
        # where k is a learnable scaling parameter
        # tanh(k*x) > 0.5 <=> x > 0.549306/k
        #Ensures the outputs are in a range interpretable as probabilities (e.g., 0 to 1).
        if self.transform_probability_method == 'tanh':
            x = torch.tanh( F.softplus(self.threshold) * x )
        else:
            x = x / (x + F.softplus(self.threshold))

        # applying the final "interpretable" modules
        #Creates an empty dictionary y_preds to store predictions for each pathway.
        y_preds = dict()
        #For each pathway (key in self.output_models) Passes the probabilities (x) through the corresponding output model.
        #Example: For "Pathway1", self.output_models["Pathway1"](x) computes the predictions ex pathway1: (num_nodes,num_genes_in_pathway1).
        for key in self.output_models:
            y_preds[key] = self.output_models[key](x)

        return module_representation, y_preds
    """
    L1 and L2 regularization are techniques used to prevent overfitting and improve the generalization of a machine learning model. 
    They work by adding penalties to the model's loss function based on the magnitude of the model's weights. 
    These penalties encourage smaller (or sparse) weights, helping the model avoid relying too heavily on specific features.
    L1 Loss: The sum of the absolute values of the model's weights.
    L2 Loss: The sum of the squared values of the model's weights.
    L1 regularization forces many weights to become exactly zero deaming them not important. Helps identify the most important features or interactions in the pathway.
    L2 regularization discourages large weights but doesn’t force them to zero.This makes the model more robust to small changes in the input data.
    In this GNN architecture, the output models are responsible for mapping the module representation (general embeddings) into pathway-specific predictions. 
    Applying L1 and L2 regularization to these models serves two purposes:
    
    Encourages Interpretability:
    Regularizing the weights helps ensure that the pathway-specific predictions are meaningful.
    For example, sparsity from L1 can highlight which nodes (genes) are most critical for a specific pathway.
    
    Reduces Overfitting:
    Output models deal with the final predictions, where overfitting is more likely since they directly relate to the target (e.g., pathway associations).
    Regularization ensures these layers don’t over-rely on specific features.

    The weights of the output models are regularized using L1 and L2 penalties.
    These penalties are added to the main loss function during training in forward pass.
    The total loss (including the regularization penalties) is used to calculate gradients during backpropagation:
    """
    def output_models_l1_l2_losses(self):
        l1_positives_loss = 0.0
        l2_positives_loss = 0.0

        for model in self.output_models.values():
            l1, l2 = model.l1_l2_losses()          
            l1_positives_loss += l1 
            l2_positives_loss += l2 

        return l1_positives_loss, l2_positives_loss
    #Declares this function as a property, meaning it can be accessed like an attribute (e.g., model.name).
    @property
    def name(self):
        return self._short_name

    #Creates a string representation of the GNN model’s architecture.
    def __str__(self):
        #Initializes an empty string to build the architecture description.
        arch = ""
        #Iterates over all hidden GNN layers in self.convs.
        for i in range(len(self.convs)):
            if self.dropout != 0.0:
                #If dropout is nonzero, appends the dropout rate to the architecture string:
                arch += f"(dropout:{self.dropout}); "
            #repr in python is like java's toString
            #If a class does not explicitly define __repr__, Python provides a default implementation, which simply shows the object’s memory address, e.g., <GCNConv object at 0x...>.
            #the output of repr here might be "GCNConv(128 -> 64); "
            arch += self.convs[i].__repr__() + "; "
            #Appends the activation function (ReLU) used after the current layer.
            arch += "ReLU; "
            #Checks if there is a batch normalization layer corresponding to the current GNN layer.
            if i < len(self.batchnorms_convs):
                arch += self.batchnorms_convs[i].__repr__() + "; "
        #ppends a string representation of the final GNN layer (self.conv_last) to the architecture description.
        if self.dropout != 0.0:
            arch += f"(dropout:{self.dropout}); "
        arch += self.conv_last.__repr__() + "; "
        arch += "SoftPlus; "
        #Appends a description of the pathway-specific output models.
        #output might append "{PositiveLinear(128 -> 10)|PositiveLinear(128 -> 5)}"
        arch += "{" + '|'.join( [self.output_models[key].__repr__() for key in self.output_models] ) + "}"
        return arch

"""
The decoder uses the inner product between node embeddings z to calculate:
Pairwise similarity (or probability of edges) for specific node pairs.
A dense adjacency matrix for all nodes.
z is  matrix where each row represents the latent embedding of a node.
"""
class InnerProductDecoder(torch.nn.Module):
    r"""Inner product decoder.

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})
        The inner product (z @ z.T) measures similarity between node embeddings.
        The sigmoid function maps these similarities into probabilities.
        
    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""
    def __init__(self):
        super().__init__()
        #Sets a shorthand name for the decoder, useful for logging or debugging.
        self._short_name = "IP"

    def forward(self, z: torch.Tensor, source: torch.Tensor, destination: torch.Tensor, sigmoid: bool = False) -> torch.Tensor:
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.
        The forward method computes edge probabilities for specific pairs of nodes.
        Edge probabilities are the model’s predictions about the likelihood of an edge existing between two nodes in the graph.
        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
                Sigmoid function to map the value to the range [0,1]
        """
        #z[source] extracts embeddings for all source nodes ex [0, 1, 2]
        #z[destination] extracts embeddings for all destination nodes ex [1, 2, 3]
        # shape of z source or destination is (num_edges, embedding_dim)
        #we multiply both z'sand end up with a matrix (num_edges, embedding_dim)
        # Sums the element-wise products across the embedding dimension (dim=1).
        # this reduces the tensor shape from (num_edges, embedding_dim) to 1D tensor of shape(num_edges,).
        #Each value represents the similarity score (inner product) for a specific edge between a source and destination node.
        value = (z[source] * z[destination]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_full(self, z: torch.Tensor, sigmoid: bool = False) -> torch.Tensor:
        #The r before a string literal makes it a raw string in Python.
        # It is useful when you want to include special characters like backslashes \ in a string without worrying about them being interpreted.
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.
        Computes pairwise similarities for all node pairs
        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`False`)
        """
        #adj is the output it's a dense adjacency matrix of shape (N, N)
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj
    
    @property
    def name(self):
        return self._short_name

    def __str__(self):
        return f"InnerProductDecoder"

class GAEL(torch.nn.Module):
    r"""Graph Autoencoder for Link Community Prediction.
    Encodes graph data into latent representations (via an encoder).
    Decodes these representations into edge probabilities or reconstructed adjacency matrices (via a decoder).
    Computes various losses for training the model and optimizes it.
    Args:
        encoder (Module): The encoder module.
        decoder (Module): The decoder module. 
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        #Defines a binary cross-entropy loss with logits. This is often used for binary classification problems (like edge prediction).
        self.bceloss = torch.nn.BCEWithLogitsLoss()
        GAEL.reset_parameters(self)
    #Calls the reset_parameters method to reinitialize all model weights.
    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)
    
    def getThresholdOfCommunities(self):
        r"""Returns the threshold above which the probability of being in a module is greater than 0.5."""
        if self.encoder.transform_probability_method == 'tanh':
            return 0.549306 / F.softplus(self.encoder.threshold)
        else:
            return F.softplus(self.encoder.threshold)

    def edgeTriplet_loss(self, L_pos, pos_index, neg_index, margin=0.5):
        r"""Computes triplet loss for anchor, positive and negative (but existing) edges."""
        
        assert L_pos.shape[0] == len(pos_index), "The length of positive edge indices does not match the number of anchor edges."
        assert L_pos.shape[0] == len(neg_index), "The length of negative edge indices does not match the number of anchor edges."
        
        E_anchor = torch.exp( -L_pos )
        E_positive = E_anchor[pos_index,:]
        E_negative = E_anchor[neg_index,:]

        probability_pos_edges = 1-torch.prod(E_anchor + E_positive - E_anchor*E_positive, dim=1)
        probability_neg_edges = 1-torch.prod(E_anchor + E_negative - E_anchor*E_negative, dim=1)

        prob_diff = probability_neg_edges - probability_pos_edges + margin

        losses = torch.max( prob_diff, torch.zeros_like(prob_diff) )
        return torch.mean(losses)
    
    def bce_loss(self, y_pred, target):
        return self.bceloss( y_pred, target )
    
    def cosine_similarity_loss(self, F, type : str, min_threshold : float = 0.0):
        valid_type_values = ['max','l1','l2']

        if type not in valid_type_values:
            raise ValueError(f"Invalid 'type' argument. Must be one of: {', '.join(valid_type_values)}")
    
        # Calculate cosine similarities for each column of the matrix
        cosine_similarities = cosine_self_similarity(F)

        eye_matrix = torch.eye(cosine_similarities.size(0), device=cosine_similarities.device)
        off_diagonal_clamped_cosine_similarities = torch.clamp( ( cosine_similarities - eye_matrix) - min_threshold, min = 0.0 )

        if type == 'max':
            loss = off_diagonal_clamped_cosine_similarities.max()
        elif type == 'l1':
            loss = off_diagonal_clamped_cosine_similarities.sum()
        elif type == 'l2':
            eye_matrix = torch.eye(cosine_similarities.size(0), device=cosine_similarities.device)
            loss = torch.norm(off_diagonal_clamped_cosine_similarities, p=2)

        return loss

    def rmse_of_module_size_loss(self, F, threshold, expected_mean):
        """
        Calculates the root mean squared error of the observed module sizes and the expected mean module size.
        """
        # Calculate the column-wise count of elements greater than the threshold
        observed_values = torch.sum(F >= threshold, dim=0, dtype=torch.float)
        observed_values = observed_values[ observed_values > 0.0 ]
        # Calculate root mean squared error of the observed module sizes and the expected mean module size
        rmse = torch.sqrt(torch.mean((observed_values - expected_mean)**2))
        return rmse

    def gcn_l1_l2_losses(self):
        l1_loss = 0.0
        l2_loss = 0.0

        for layer in self.encoder.modules():
            if isinstance(layer, torch_geometric.nn.GCNConv):
                l1_loss += torch.norm(layer.lin.weight, p=1)
                l2_loss += torch.norm(layer.lin.weight, p=2)

        return l1_loss, l2_loss
    
    def output_models_l1_l2_losses(self):
        return self.encoder.output_models_l1_l2_losses()
    
    def nll_BernoulliPoisson_loss(self, strength_pos_edges, strength_neg_edges, epsilon = 1e-8):
        r"""Given latent variables :obj:`H`, computes the Bernoulli-Poisson 
        loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges :obj:`neg_edge_index`.

        Args:
            H (Tensor): The latent space :math:`\mathbf{H}`.
            pos_edge_index (LongTensor): The positive edges to train against.
            neg_edge_index (LongTensor): The negative edges to train against.
        """
        
        ll_pos_edges = -torch.mean( torch.log( -torch.expm1( -strength_pos_edges - epsilon ) ) )
        ll_neg_edges = torch.mean( strength_neg_edges )

        ll = (ll_pos_edges + ll_neg_edges) / 2.0
        return ll
    
    def nll_BernoulliPoisson_loss_full(self, H, L_pos, num_edges, num_nonedges, epsilon = 1e-8):
        """Compute full loss."""
        strength_pos_edges = torch.sum( L_pos, dim=1 )
        strength_all_possible_edges = torch.sum( self.decoder.compute_all_possible_edge_strengths( H ) )
        loss_nonedges = strength_all_possible_edges - torch.sum( strength_pos_edges )

        ll_pos_edges = -torch.sum( torch.log( -torch.expm1( -strength_pos_edges - epsilon ) ) )

        # if self.balance_loss:
        #     neg_scale = 1.0
        # else:
        #     neg_scale = num_nonedges / num_edges
        # ll = (ll_pos_edges / num_edges + neg_scale * loss_nonedges / num_nonedges) / (1 + neg_scale)
        ll = (ll_pos_edges / num_edges + loss_nonedges / num_nonedges)
        return ll

    def configure_optimizers(self, params):
        """
        This implementation is based on https://github.com/karpathy/minGPT
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, PositiveLinear, torch_geometric.nn.GCNConv, torch_geometric.nn.GATv2Conv, torch_geometric.nn.SAGEConv)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm1d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('att') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif fpn == 'encoder.threshold':
                    # special cases as not decayed
                    no_decay.add(fpn)
                elif 'adjacency_tensor_weights' in pn:
                    # special cases as not decayed
                    no_decay.add(fpn)

        #print( f"decay: {str(decay)}")
        #print( f"no_decay: {str(no_decay)}")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": params.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=params.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.learning_rate_decay_step_size, gamma=0.85)

        return optimizer, scheduler

    @property
    def name(self):
        return f"{self.encoder.name}--{self.decoder.name}"

    def __str__(self):
        return f"GAEL( Encoder: {str(self.encoder)}, Decoder: {str(self.decoder)} )"

