import secrets

# Generate a secure random key
key = secrets.token_hex(24)
print(key)
