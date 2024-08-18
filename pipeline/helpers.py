from colorama import init, Fore, Back, Style
def log_progress(ti, message):
    ti.xcom_push(key='progress_log', value=message)
    print(Back.GREEN + message + Style.RESET_ALL)