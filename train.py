import torch
from mlp import MLP, MLP_OneLayer

def save_weights_bias(model, weights_path, bias_path):
    weight = []
    bias = []
    
    model_keys = model.state_dict().keys()
    has_bias = False
    
    for key in model_keys:
        if 'weight' in key:
            weight.append(model.state_dict()[key].T.flatten())
        elif 'bias' in key:
            has_bias = True
            bias.append(model.state_dict()[key].flatten())
            
    weight = torch.cat(weight, dim=0)
    
    with open(weights_path, 'wb') as f:
        weight.cpu().numpy().tofile(f)
        f.close()
    
    if has_bias:
        bias = torch.cat(bias, dim=0)
        with open(bias_path, 'wb') as f:
            bias.cpu().numpy().tofile(f)
            f.close()

if __name__ == "__main__":
    in_dims = 3
    hidden_dims = 64
    out_dims = 4
    hidden_layers = 2
    batch_size = 2
    model = MLP(in_dims, hidden_dims, out_dims, hidden_layers)
    # model = MLP_OneLayer(in_dims, out_dims)
    
    x = torch.linspace(0, (in_dims * batch_size - 1) * 0.1, in_dims * batch_size)
    x = x.reshape(batch_size, in_dims)
    print("Input:")
    print(x)
    
    y = model(x)
    print("Output:")
    print(y)
    
    weights_path = 'weights.bin'
    bias_path = 'bias.bin'
    
    save_weights_bias(model, weights_path, bias_path)
    
    # weight = model.state_dict()['net.weight']
    # bias = model.state_dict()['net.bias']
    
    # print("Weight:")
    # print(weight)
    
    # print("Bias:")
    # print(bias)