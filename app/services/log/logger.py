from datetime import datetime


def log_event(message: str, to_file: bool = False):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)

    if to_file:
        with open("app.log", "a") as f:
            f.write(full_message + "\n")
