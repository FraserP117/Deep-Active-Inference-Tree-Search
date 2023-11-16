import torch

rnd_idx = torch.randint(0, 1, (1,))

unused_actions = torch.tensor([0, 1])

# Extract the scalar value from the tensor
a_prime_scalar = unused_actions[rnd_idx].item()

# Now create a 1D tensor from the scalar value
a_prime = torch.tensor([a_prime_scalar], dtype=torch.float32, device='cpu')

print(f"\na_prime_scalar: {a_prime_scalar}, type(a_prime_scalar): {type(a_prime_scalar)}")
print(f"a_prime: {a_prime}, type(a_prime): {type(a_prime)}\n")
