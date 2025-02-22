import hashlib
import os
import pyotp
import face_recognition
from cryptography.fernet import Fernet
import logging

# Security Configuration
SALT = os.urandom(32)
FERNET_KEY = Fernet.generate_key()
fernet = Fernet(FERNET_KEY)
SECRET_KEY = pyotp.random_base32()

# Logging setup
logging.basicConfig(filename="security.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Secure User Database (Encrypted)
user_data = {}

# Utility Functions
def hash_password(password: str) -> str:
    """Hash the password using SHA-256 with a unique salt."""
    salted_password = password.encode() + SALT
    return hashlib.sha256(salted_password).hexdigest()

def encrypt_data(data: str) -> str:
    """Encrypt sensitive data using AES-256 (Fernet)."""
    return fernet.encrypt(data.encode()).decode()

def decrypt_data(encrypted_data: str) -> str:
    """Decrypt sensitive data using AES-256 (Fernet)."""
    return fernet.decrypt(encrypted_data.encode()).decode()

# User Registration
def register_user(username: str, password: str):
    """Register a user with hashed password and encrypted storage."""
    if username in user_data:
        logging.warning(f"User '{username}' already exists.")
        return "[Error] User already exists."
    hashed_password = hash_password(password)
    user_data[username] = {"password": encrypt_data(hashed_password), "2FA": None}
    logging.info(f"User '{username}' registered securely.")
    return f"[Success] User '{username}' registered securely."

# Two-Factor Authentication (2FA) Setup
def setup_2fa(username: str):
    """Generate and store 2FA secret key for the user."""
    if username not in user_data:
        logging.warning(f"User '{username}' not found.")
        return "[Error] User not found."
    user_data[username]["2FA"] = SECRET_KEY
    logging.info(f"2FA setup for user '{username}'.")
    return f"[2FA] Scan this OTP Key in your Authenticator: {SECRET_KEY}"

# Login with Identity Verification
def login(username: str, password: str, otp_code: str):
    """Verify user identity using password and 2FA."""
    if username not in user_data:
        logging.warning(f"User '{username}' not found.")
        return "[Error] User not found."

    # Verify Password
    stored_password = decrypt_data(user_data[username]["password"])
    if stored_password != hash_password(password):
        logging.warning(f"Invalid password for user '{username}'.")
        return "[Error] Invalid password."

    # Verify 2FA (Time-based OTP)
    totp = pyotp.TOTP(user_data[username]["2FA"])
    if not totp.verify(otp_code):
        logging.warning(f"Invalid OTP for user '{username}'.")
        return "[Error] Invalid OTP."

    logging.info(f"User '{username}' logged in securely.")
    return f"[Success] User '{username}' logged in securely."

# Biometric Face Recognition
def verify_face():
    """Verify user face against saved authorized face."""
    try:
        known_image = face_recognition.load_image_file("authorized_face.jpg")
        unknown_image = face_recognition.load_image_file("attempt.jpg")

        known_encoding = face_recognition.face_encodings(known_image)[0]
        unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

        result = face_recognition.compare_faces([known_encoding], unknown_encoding)[0]
        logging.info(f"Face verification result: {result}")
        return result
    except Exception as e:
        logging.error(f"Face verification failed: {e}")
        return False