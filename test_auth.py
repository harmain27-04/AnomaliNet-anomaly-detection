from backend.auth import (
    register_user,
    login_user
)

print(
    register_user(
        "Harmain",
        "harmain@gmail.com",
        "9876543210",
        "123456"
    )
)

print(
    login_user(
        "harmain@gmail.com",
        "123456"
    )
)