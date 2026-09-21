import torch 

batch_size = 1
seq_len = 4
d_k = 8

Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_k)

print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)

K_transposed = K.transpose(-2, -1)

scores = torch.matmul(Q, K_transposed)

print("Attention scores shape:", scores.shape)
print(scores)

import math 
scaled_scores =  scores / math.sqrt(d_k)

print(f"scaled scores: //n {scaled_scores}" )

attention_weights = torch.softmax(scaled_scores, dim =-1)

print(f"attention_weights: {attention_weights}" )

output = torch.matmul(attention_weights, V)

print(f"output shape: {output.shape}" )
print(output)