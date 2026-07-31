from functools import wraps

from flask import (
    session,
    redirect,
    url_for,
    flash
)

from backend.auth import (
    login_user
)

# =====================================
# LOGIN
# =====================================

def authenticate_user(email, password):

    user = login_user(
        email,
        password
    )

    if user:

        session["user_id"] = user["id"]

        session["user_name"] = user["full_name"]

        session["user_email"] = user["email"]

        return True

    return False


# =====================================
# LOGOUT
# =====================================

def logout_user():

    session.clear()


# =====================================
# CURRENT USER
# =====================================

def current_user():

    if "user_id" in session:

        return {

            "id": session["user_id"],

            "name": session["user_name"],

            "email": session["user_email"]

        }

    return None


# =====================================
# LOGIN REQUIRED DECORATOR
# =====================================

def login_required(route):

    @wraps(route)

    def wrapper(*args, **kwargs):

        if "user_id" not in session:

            flash(
                "Please login first."
            )

            return redirect(
                url_for("login")
            )

        return route(
            *args,
            **kwargs
        )

    return wrapper