from pydantic import BaseModel


class UserDto(BaseModel):
    name: str
    email: str
    password: str


class LoginDto(BaseModel):
    email: str
    password: str
