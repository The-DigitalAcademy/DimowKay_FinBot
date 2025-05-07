import streamlit as st
from sqlalchemy.orm import sessionmaker
from passlib.hash import bcrypt
from signup import User, engine

SessionLocal = sessionmaker(bind=engine)

def reset_password():
    st.title("Reset Password")
    token = st.text_input("Enter reset token")
    new_password = st.text_input("New Password", type="password")
    if st.button("Update Password"):
        session = SessionLocal()
        user = session.query(User).filter_by(reset_token=token).first()
        if user:
            user.password = bcrypt.hash(new_password)
            user.reset_token = None  # Clear token after reset
            session.commit()
            st.success("Password updated! Redirecting to login...")
            st.switch_page("pages/login.py")  
        else:
            st.error("Invalid reset token.")
        session.close()

reset_password()
