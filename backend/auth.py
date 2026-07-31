import sqlite3
from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from backend.database import (
    get_connection,
    log_event
)

# =====================================================
# REGISTER USER
# =====================================================

def register_user(
    full_name,
    email,
    phone,
    password
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id
        FROM users
        WHERE email=?
        """,
        (email,)
    )

    existing_user = cursor.fetchone()

    if existing_user:

        connection.close()

        return False

    hashed_password = generate_password_hash(
        password
    )

    cursor.execute(
        """
        INSERT INTO users
        (
            full_name,
            email,
            phone,
            password
        )

        VALUES
        (
            ?,?,?,?
        )
        """,
        (
            full_name,
            email,
            phone,
            hashed_password
        )
    )

    connection.commit()

    log_event(
        f"New User Registered : {email}"
    )

    connection.close()

    return True


# =====================================================
# LOGIN USER
# =====================================================

def login_user(
    email,
    password
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM users
        WHERE email=?
        """,
        (email,)
    )

    user = cursor.fetchone()

    connection.close()

    if user is None:

        return None

    if check_password_hash(
        user["password"],
        password
    ):

        log_event(
            f"User Logged In : {email}"
        )

        return dict(user)

    return None


# =====================================================
# GET USER
# =====================================================

def get_user(user_id):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM users
        WHERE id=?
        """,
        (user_id,)
    )

    user = cursor.fetchone()

    connection.close()

    if user:

        return dict(user)

    return None


# =====================================================
# UPDATE PROFILE
# =====================================================

def update_user(
    user_id,
    full_name,
    phone
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        UPDATE users

        SET

        full_name=?,
        phone=?

        WHERE id=?
        """,
        (
            full_name,
            phone,
            user_id
        )
    )

    connection.commit()

    connection.close()

    log_event(
        f"Profile Updated : {user_id}"
    )


# =====================================================
# DELETE USER
# =====================================================

def delete_user(user_id):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        DELETE
        FROM users

        WHERE id=?
        """,
        (user_id,)
    )

    connection.commit()

    connection.close()

    log_event(
        f"User Deleted : {user_id}"
    )