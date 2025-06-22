import logging
from tkinter import messagebox

from data_collection.config import Config
from data_collection.user_interface import ASLDataCollectorUI

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    try:
        config = Config()
        app = ASLDataCollectorUI(config)
        app.run()

    except Exception as e:
        logging.error(f"APPLICATION ERROR: {e}")
        messagebox.showerror("Application Error", f"An error occurred: {e}")


if __name__ == '__main__':
    main()