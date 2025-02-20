import torch 
import torch.nn as nn

# Defining input size, hidden layer size, output size and batch size respectively
n_in, n_h, n_out, batch_size = 10, 5, 1, 10

# Create dummy input and target tensors (data)
x = torch.randn(batch_size, n_in)
y = torch.tensor([[1.0], [0.0], [0.0], 
[1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])

# Create a model
model = nn.Sequential(nn.Linear(n_in, n_h),
   nn.ReLU(),
   nn.Linear(n_h, n_out),
   nn.Sigmoid())

Construct the loss function
criterion = torch.nn.MSELoss()
# Construct the optimizer (Stochastic Gradient Descent in this case)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# Gradient Descent
for epoch in range(50):
   # Forward pass: Compute predicted y by passing x to the model
   y_pred = model(x)

   # Compute and print loss
   loss = criterion(y_pred, y)
   print('epoch: ', epoch,' loss: ', loss.item())

   # Zero gradients, perform a backward pass, and update the weights.
   optimizer.zero_grad()

   # perform a backward pass (backpropagation)
   loss.backward()

   # Update the parameters
   optimizer.step()

#OUTPUT
epoch: 0 loss: 0.2545787990093231
epoch: 1 loss: 0.2545052170753479
epoch: 2 loss: 0.254431813955307
epoch: 3 loss: 0.25435858964920044
epoch: 4 loss: 0.2542854845523834
.
.
.
.

epoch: 47 loss: 0.2512802183628082
epoch: 48 loss: 0.2512132525444031
epoch: 49 loss: 0.251146405935287
