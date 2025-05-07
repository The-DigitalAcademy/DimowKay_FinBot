import streamlit as st
from sqlalchemy.orm import sessionmaker
import secrets
from signup import User, engine

SessionLocal = sessionmaker(bind=engine)

def forgot_password():
    st.title("Forgot Password")
    email = st.text_input("Enter your email")
    if st.button("Reset Password"):
        session = SessionLocal()
        user = session.query(User).filter_by(email=email).first()
        if user:
            user.reset_token = secrets.token_hex(16)
            session.commit()
            st.success(f"Password reset link sent! Token: {user.reset_token}")  # Replace with actual email logic
        else:
            st.error("Email not found.")
        session.close()

    st.write("[Reset Password](pages/reset_password.py)")  # Link to reset page

forgot_password()
