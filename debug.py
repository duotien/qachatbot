import os

from qachatbot import PROJECT_DIR

APP_PATH = os.path.join(PROJECT_DIR, "app.py")

if __name__ == "__main__":
    from chainlit.cli import config, run_chainlit

    config.run.watch = True
    config.run.headless = True
    config.run.debug = False
    run_chainlit(APP_PATH)
