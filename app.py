import hashlib
import subprocess

# Replace with the data you want to hash
data_to_hash = "Hello from Python!"

# Function to compute SHA-256 hash
def compute_hash(data):
    hash_value = hashlib.sha256(data.encode()).hexdigest()
    return hash_value

# Compute hash of data
hash_to_send = compute_hash(data_to_hash)

# Print the computed hash (optional, for verification)
print(f"Hash computed by Python: {hash_to_send}")

# Output the hash to a file or use in further processing
with open('hash_output.txt', 'w') as f:
    f.write(hash_to_send)

# Run JavaScript code using Node.js subprocess
try:
    subprocess.run(['node', 'Interaction.js'], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error running Node.js script: {e}")
