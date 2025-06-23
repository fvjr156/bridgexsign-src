from tkinter import StringVar, IntVar, Tk, messagebox, Label, OptionMenu, Entry, Button
from typing import Any

import cv2
from data_collection.observers import AbstractObserver
from data_collection.config import Config
from data_collection.engine import DataCollectionEngine

class ASLDataCollectorUI(AbstractObserver): # da UI
    def __init__(self, config: Config):
        self.config = config
        self.engine = DataCollectionEngine(config)
        self.engine.attach(self)
        self.root = Tk()
        self.root.title('ASL Data Collector')
        self.root.geometry("400x300")



        self.label_var = StringVar(value=config.LABELS[0])
        self.num_samples_var = IntVar(value=config.DEFAULT_SAMPLES)
        self.delay_var = IntVar(value=config.DEFAULT_DELAY)
        
        self.start_button = None
        self.stop_button = None
        self.status_label = None
        
        self._setup_ui()

    def _setup_ui(self):
        # tanginang layout code to
        Label(
            self.root, 
            text="Sign Label:", 
            font=("Arial", 10, "bold")
            ).grid(row=0, column=0, padx=10, pady=10, sticky="e")
        OptionMenu(
            self.root, 
            self.label_var, 
            *self.config.LABELS
            ).grid( row=0, column=1, padx=10, pady=10, sticky="w")
        Label(
            self.root, 
            text="Samples to collect:", 
            font=("Arial", 10, "bold")
            ).grid( row=1, column=0, padx=10, pady=5, sticky="e") 
        Entry(
            self.root, 
            textvariable=self.num_samples_var, 
            width=10
            ).grid( row=1, column=1, padx=10, pady=5, sticky="w")
        Label(
            self.root, 
            text="Countdown delay (s):", 
            font=("Arial", 10, "bold")
            ).grid( row=2, column=0, padx=10, pady=5, sticky="e")
        Entry(
            self.root, 
            textvariable=self.delay_var, 
            width=10
            ).grid( row=2, column=1, padx=10, pady=5, sticky="w")
        
        button_frame = Label(self.root)
        button_frame.grid(row=3, column=0, columnspan=2, pady=20)
        
        self.start_button = Button(
            button_frame, 
            text="START COLLECTION", 
            command=self._start_collection, 
            bg="green", 
            fg="white", 
            font=("Arial", 12, "bold")
            )
        self.start_button.pack(side="left", padx=10)
        
        self.stop_button = Button(
            button_frame, 
            text="STOP", 
            command=self._stop_collection, 
            bg="red", 
            fg="white", 
            font=("Arial", 12, "bold"), 
            state="disabled"
            )
        self.stop_button.pack(side="left", padx=10)
        
        self.status_label = Label(
            self.root, 
            text="Ready to collect data", 
            font=("Arial", 10), 
            fg="blue"
            )
        self.status_label.grid( row=4, column=0, columnspan=2, pady=10 )
        
        instructions = ("Instructions:\n"
                       "1. Select the sign label\n"
                       "2. Set number of samples and countdown delay\n"
                       "3. Click START COLLECTION\n"
                       "4. Perform the sign when countdown ends\n"
                       "5. Press 'q' during collection to quit early")
        Label(
            self.root, 
            text=instructions, 
            font=("Arial", 9), 
            justify="left", 
            fg="gray"
            ).grid(row=5, column=0, columnspan=2, padx=20, pady=20, sticky="w")
        
        # kaya lowkey ayaw ko mag layout design sa tkinter eh


    def _start_collection(self):
        if self._validate_inputs():
            success = self.engine.start_session(
                self.label_var.get(),
                self.num_samples_var.get(),
                self.delay_var.get()
            )
            if success:
                self.start_button.config(state="disabled")
                self.stop_button.config(state="normal")

    def _stop_collection(self):
        self.engine.stop_session()
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")

    def _validate_inputs(self) -> bool:
        try:
            if self.num_samples_var.get() <= 0:
                raise ValueError("Number of samples must be positive")
            if self.delay_var.get() < 0:
                raise ValueError("Delay cannot be negative")
            return True
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
            return False

    def update(self, event_type: str, data: Any):
        # naga-handle dito ng messages by engine
        # event_type: error, sample_completed, frame_update, session_completed
        if event_type == "error":
            messagebox.showerror("Error", data)
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.status_label.config(text="Error occurred - Ready to collect data")
        elif event_type == "sample_completed":
            session = data
            self.status_label.config(
                text=f"Collected {session.collected_samples}/{session.target_samples} samples"
            )
        elif event_type == "frame_update":
            cv2.imshow("ASL Data Collector", data)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.engine.stop_session()
        
        elif event_type == "session_completed":
            session = data
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.status_label.config(
                text=f"Collection complete! Collected {session.collected_samples} samples"
            )
            messagebox.showinfo("Complete", 
                              f"Successfully collected {session.collected_samples} samples "
                              f"for sign '{session.label}'")

    def run(self):
        try:
            self.root.mainloop()
        finally:
            self.engine.stop_session()