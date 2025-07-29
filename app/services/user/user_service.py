from sqlalchemy.orm import Session
from app.models.user_model import User
from app.repository import user_repository
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def register_user(db: Session, name: str, email: str, password: str):
    hashed_password = pwd_context.hash(password)
    user = User(name=name, email=email, password=hashed_password)
    return user_repository.create_user(db, user)


def login_user(db: Session, email: str, password: str):
    user = user_repository.get_user_by_email(db, email)
    if not user or not pwd_context.verify(password, user.password):
        return None
    return user
