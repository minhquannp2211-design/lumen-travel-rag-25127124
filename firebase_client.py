"""Small Firebase Email/Password wrapper based on the Lab 03 REST approach.

No Firestore, Firebase Admin SDK, or database is used in this assignment.
"""

import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

AUTH_URL = "https://identitytoolkit.googleapis.com/v1/accounts:"


class FirebaseError(Exception):
    pass


def _auth(action: str, email: str, password: str) -> dict:
    api_key = os.getenv("FIREBASE_API_KEY", "").strip()
    if not api_key:
        raise FirebaseError("FIREBASE_API_KEY is missing from .env")
    if not email.strip() or not password:
        raise FirebaseError("Email and password are required.")

    try:
        response = requests.post(
            AUTH_URL + action,
            params={"key": api_key},
            json={"email": email.strip(), "password": password, "returnSecureToken": True},
            timeout=15,
        )
        data = response.json()
    except (requests.RequestException, ValueError):
        # Do not display network errors: they may contain the API key in the URL.
        raise FirebaseError("Cannot reach Firebase or read its response.") from None

    if not response.ok:
        code = str(data.get("error", {}).get("message", ""))
        messages = {
            "EMAIL_EXISTS": "This email is already registered.",
            "INVALID_EMAIL": "Enter a valid email address.",
            "OPERATION_NOT_ALLOWED": "Enable Email/Password in Firebase Authentication.",
            "INVALID_LOGIN_CREDENTIALS": "Incorrect email or password.",
            "INVALID_PASSWORD": "Incorrect email or password.",
            "EMAIL_NOT_FOUND": "Incorrect email or password.",
            "WEAK_PASSWORD": "Password does not meet Firebase requirements.",
        }
        if "TOO_MANY_ATTEMPTS" in code:
            raise FirebaseError("Too many attempts. Please try again later.")
        if "PASSWORD_DOES_NOT_MEET_REQUIREMENTS" in code:
            raise FirebaseError("Password does not meet Firebase requirements.")
        raise FirebaseError(messages.get(code, "Firebase authentication failed. Check your setup."))
    return data


def create_account(email: str, password: str) -> None:
    """Register an account; users sign in explicitly afterward (like Lab 03)."""
    _auth("signUp", email, password)


def sign_in(email: str, password: str) -> dict:
    """Save only this session's user information; never retain the password."""
    data = _auth("signInWithPassword", email, password)
    try:
        return {
            "uid": data["localId"],
            "email": data["email"],
            "expires_at": time.time() + int(data["expiresIn"]),
        }
    except (KeyError, TypeError, ValueError):
        raise FirebaseError("Firebase returned incomplete sign-in information.") from None
