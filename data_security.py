import os
import logging
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Secure key derivation using PBKDF2
def derive_key(password: str, salt: bytes) -> bytes:
    try:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        return kdf.derive(password.encode())
    except Exception as e:
        logging.error(f"Key derivation failed: {e}")
        raise

# Generate a strong RSA public-private key pair
def generate_rsa_keys():
    try:
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        public_key = private_key.public_key()

        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        )

        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        return private_pem, public_pem
    except Exception as e:
        logging.error(f"RSA key generation failed: {e}")
        raise

# Encrypt data using AES
def aes_encrypt(data: bytes, key: bytes) -> bytes:
    try:
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        padded_data = padder.update(data) + padder.finalize()

        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        return iv + encrypted_data
    except Exception as e:
        logging.error(f"AES encryption failed: {e}")
        raise

# Decrypt data using AES
def aes_decrypt(encrypted_data: bytes, key: bytes) -> bytes:
    try:
        iv = encrypted_data[:16]
        cipher_data = encrypted_data[16:]

        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        
        decrypted_data = decryptor.update(cipher_data) + decryptor.finalize()
        
        unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
        unpadded_data = unpadder.update(decrypted_data) + unpadder.finalize()

        return unpadded_data
    except Exception as e:
        logging.error(f"AES decryption failed: {e}")
        raise

# RSA encryption and decryption
def rsa_encrypt(public_key: bytes, data: bytes) -> bytes:
    try:
        public_key_obj = serialization.load_pem_public_key(public_key, backend=default_backend())
        encrypted_data = public_key_obj.encrypt(
            data,
            rsa.OAEP(
                mgf=rsa.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return encrypted_data
    except Exception as e:
        logging.error(f"RSA encryption failed: {e}")
        raise

def rsa_decrypt(private_key: bytes, encrypted_data: bytes) -> bytes:
    try:
        private_key_obj = serialization.load_pem_private_key(private_key, password=None, backend=default_backend())
        decrypted_data = private_key_obj.decrypt(
            encrypted_data,
            rsa.OAEP(
                mgf=rsa.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return decrypted_data
    except Exception as e:
        logging.error(f"RSA decryption failed: {e}")
        raise

# Securely store and retrieve sensitive information
def store_encrypted_data(file_path: str, data: bytes, encryption_key: bytes):
    try:
        encrypted_data = aes_encrypt(data, encryption_key)
        with open(file_path, 'wb') as file:
            file.write(encrypted_data)
        logging.info(f"Data securely stored at {file_path}")
    except Exception as e:
        logging.error(f"Failed to store encrypted data: {e}")
        raise

def retrieve_encrypted_data(file_path: str, encryption_key: bytes) -> bytes:
    try:
        with open(file_path, 'rb') as file:
            encrypted_data = file.read()
        return aes_decrypt(encrypted_data, encryption_key)
    except Exception as e:
        logging.error(f"Failed to retrieve encrypted data: {e}")
        raise

# Ensure strong random number generation for cryptographic operations
def secure_random_bytes(size: int) -> bytes:
    try:
        return os.urandom(size)
    except Exception as e:
        logging.error(f"Random byte generation failed: {e}")
        raise

# Salting passwords for extra protection
def salt_password(password: str) -> tuple:
    try:
        salt = secure_random_bytes(16)
        salted_password = derive_key(password, salt)
        return salted_password, salt
    except Exception as e:
        logging.error(f"Password salting failed: {e}")
        raise

# Key storage for private key management (secure storage)
def store_rsa_private_key(private_key: bytes, file_path: str):
    try:
        with open(file_path, 'wb') as file:
            file.write(private_key)
        logging.info(f"RSA private key securely stored at {file_path}")
    except Exception as e:
        logging.error(f"Failed to store RSA private key: {e}")
        raise

def retrieve_rsa_private_key(file_path: str) -> bytes:
    try:
        with open(file_path, 'rb') as file:
            private_key = file.read()
        return private_key
    except Exception as e:
        logging.error(f"Failed to retrieve RSA private key: {e}")
        raise