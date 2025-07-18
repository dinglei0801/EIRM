import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GaussianLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigmas = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.centres, -1, 1)
        nn.init.constant_(self.sigmas, 1)

    def forward(self, input):
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) / self.sigmas.unsqueeze(0)
        return torch.exp(-(distances.pow(2)))

class GaussianNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GaussianNetwork, self).__init__()
        self.gaussian1 = GaussianLayer(input_dim, hidden_dim)
        self.gaussian2 = GaussianLayer(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.gaussian1(x)
        x = self.gaussian2(x)
        return x

class RelationModule(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RelationModule, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x):
        return self.layer(x)

class DualGaussianNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, relation_hidden_dim):
        super(DualGaussianNetwork, self).__init__()
        self.encoder = GaussianNetwork(input_dim, hidden_dim, output_dim)
        self.relation_module = RelationModule(output_dim, relation_hidden_dim)
    
    def forward(self, x):
        features = self.encoder(x)
        enhanced_features = self.relation_module(features)
        return features, enhanced_features

# Function to create the model (replacing resnet12)
def create_gaussian_network(input_dim, hidden_dim, output_dim, relation_hidden_dim):
    return DualGaussianNetwork(input_dim, hidden_dim, output_dim, relation_hidden_dim)
