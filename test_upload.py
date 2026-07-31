from backend.upload_video import allowed_file

print(

    allowed_file(

        "fight.mp4"

    )

)

print(

    allowed_file(

        "abc.jpg"

    )

)