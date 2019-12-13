import numpy as np


timesteps = 100
input_features = 32
output_features = 64

inputs = np.random.random((timesteps, input_features))
state_t = np.zeros((output_features,))

W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))

successive_outputs = []
for input_t in inputs:		# 取出每一行 (循环 timesteps 次)
	output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
	successive_outputs.append(output_t)
	state_t = output_t 		# Updates the state of the network for the next timestep

# The final output is a 2D tensor of shape (timesteps, output_features).
final_output_sequence = np.concatenate(successive_outputs, axis=0)
print(final_output_sequence.shape, len(successive_outputs), len(successive_outputs[-1]))
print(final_output_sequence[-64:])