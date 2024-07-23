import numpy as np
import torch

def print_comparable_bits(int1, int2):
    if (int1==int2):
        print("Equal")
        return None
    bin1 = f'{int1:064b}'
    bin2 = f'{int2:064b}'
    print(bin1)
    print(bin2)

def float_bits_to_int(x):
    x = x.squeeze(0).round().int().tolist()
    out = 0
    for bit in x:
        out = (out << 1) | bit
    return out
def int_to_bits_tensor(x,width=64):
    out = torch.zeros(width)
    for i in range(width):
        out[width-i-1] = x & 1
        x >>= 1
    return out
def visualize_loss_and_gradient(model, p1_struct,p2_struct, p1_range, p2_range,X,title,resolution=30,target_coordinate=None):
    p1_param,p1_index = p1_struct
    p2_param,p2_index = p2_struct
    starting_p1 = p1_param[p1_index].item()
    starting_p2 = p2_param[p2_index].item()
    P1 = np.linspace (p1_range[0], p1_range[1], resolution)
    P2 = np.linspace (p2_range[0], p2_range[1], resolution)
    X_axis, Y_axis = np.meshgrid(P1, P2)
    losses = np.zeros_like(X_axis)
    grads_x = np.zeros_like(X_axis)
    grads_y = np.zeros_like(X_axis)
    for i in range(resolution):
        for j in range(resolution):
            with torch.no_grad():
                p1_param[p1_index] = P1[i]
                p2_param[p2_index] = P2[j]
            Y_pred = model(X)

            loss.backward(retain_graph=True)
            losses[i, j] = loss.item()
            grads_x[i, j] = p1_param.grad[p1_index]
            grads_y[i, j] = p2_param.grad[p2_index]
            model.zero_grad()

    # Plotting the loss surface
    plt.figure(figsize=(10, 8))
    plt.axis('equal')
    contour = plt.contourf(P1, P2, losses, levels=50, cmap='viridis')
    plt.colorbar(contour)  # Add a color bar to the contour plot
    plt.title(title)
    starting_coordinate = (starting_p1, starting_p2)  # Define your starting P1, P2 coordinates
    plt.plot(starting_coordinate[0], starting_coordinate[1], 'ro')  # 'ro' for red circle
    if target_coordinate is not None:
        plt.plot(target_coordinate[0], target_coordinate[1], 'go')
    # Overlay the gradient field as a quiver plot
    # Use a contrasting color for the arrows, such as white
    plt.show()
    #restore previous values for model:
    with torch.no_grad():
        p1_param[p1_index] = starting_p1
        p2_param[p2_index] = starting_p2
