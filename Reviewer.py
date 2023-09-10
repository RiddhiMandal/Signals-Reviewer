import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from scipy.signal import butter, lfilter

class ReviewApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Review")

        # Create a frame for the buttons at the bottom
        button_frame = tk.Frame(root)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        # Create a frame for the plot
        plot_frame = tk.Frame(root)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.load_button = tk.Button(button_frame, text="Load CSV", command=self.load_csv)
        self.load_button.pack(side=tk.LEFT, padx=10)

        # Add the preprocessing toggle button
        self.preprocess_button = tk.Button(button_frame, text="Preprocessing", command=self.toggle_preprocess)
        self.preprocess_button.pack(side=tk.LEFT, padx=10)
        self.preprocess_enabled = False

        self.patient_index = 0
        self.data = None
        self.responses = []

        self.play_button = tk.Button(button_frame, text="Play", state=tk.DISABLED, command=self.toggle_play_pause)
        self.play_button.pack(side=tk.LEFT, padx=10)

        self.yes_button = tk.Button(button_frame, text="Yes", state=tk.DISABLED, command=self.mark_yes)
        self.yes_button.pack(side=tk.LEFT, padx=10)

        self.no_button = tk.Button(button_frame, text="No", state=tk.DISABLED, command=self.mark_no)
        self.no_button.pack(side=tk.LEFT, padx=10)

        self.prev_button = tk.Button(button_frame, text="Previous", state=tk.DISABLED, command=self.prev_patient)
        self.prev_button.pack(side=tk.LEFT, padx=10)

        self.next_button = tk.Button(button_frame, text="Next", state=tk.DISABLED, command=self.next_patient)
        self.next_button.pack(side=tk.RIGHT, padx=10)

        self.save_button = tk.Button(button_frame, text="Save Responses", state=tk.DISABLED, command=self.save_responses)
        self.save_button.pack(side=tk.RIGHT, padx=10)

        self.fig, self.ax = plt.subplots(figsize=(10, 6))  # Enlarge the figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()

        self.play_speed = 1.0
        self.playing = False
        self.current_time = 0
        self.update_interval = 100
        self.flow_length = 3

    def toggle_preprocess(self):
        self.preprocess_enabled = not self.preprocess_enabled

        if self.preprocess_enabled:
            # Apply the Butterworth filter to the ECG data
            cutoff = 30  # Cutoff frequency (in Hz)
            order = 4    # Order of the filter
            fs = 1000    # Sampling frequency of the ECG signal (in Hz)
            b, a = butter(order, cutoff / (fs / 2), btype='low')
            signal = self.data.iloc[self.patient_index]
            filtered = lfilter(b, a, signal)

            # Show a pop-up message when preprocessing is completed
            messagebox.showinfo("Preprocessing", "Data preprocessing is completed.")

        self.show_patient_data()

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.data = pd.read_csv(file_path)
            self.patient_index = 0
            self.responses = [""] * len(self.data)
            self.show_patient_data()

    def show_patient_data(self):
        if self.data is not None:
            patient_data = self.data.iloc[self.patient_index]

            # Calculate time axis (assuming 20 seconds)
            time_axis = np.linspace(0, 20, len(patient_data))

            self.ax.clear()

            if self.preprocess_enabled:
                # Apply the Butterworth filter to the ECG data
                cutoff = 30  # Cutoff frequency (in Hz)
                order = 4    # Order of the filter
                fs = 1000    # Sampling frequency of the signal (in Hz)
                b, a = butter(order, cutoff / (fs / 2), btype='low')
                filtered = lfilter(b, a, patient_data)

                # Plot the filtered  signal
                self.ax.plot(time_axis, filtered, label="Filtered")
            else:
                # Plot the raw ECG signal
                self.ax.plot(time_axis, patient_data, label=" ECG")

            self.ax.set_title(f" Signal  {self.patient_index + 1}")
            self.ax.set_xlabel("Time (seconds)")
            self.ax.set_ylabel("Amplitude")
            self.ax.set_xlim(self.current_time - self.flow_length, self.current_time)  # Display the last N seconds
            self.ax.legend()
            self.canvas.draw()

            self.enable_buttons()

    def enable_buttons(self):
        if self.data is not None:
            self.play_button.config(state=tk.NORMAL)
            self.yes_button.config(state=tk.NORMAL)
            self.no_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.NORMAL)

            if self.patient_index < len(self.data) - 1:
                self.next_button.config(state=tk.NORMAL)
            if self.patient_index > 0:
                self.prev_button.config(state=tk.NORMAL)
            else:
                self.prev_button.config(state=tk.DISABLED)

    def mark_yes(self):
        if self.data is not None:
            self.responses[self.patient_index] = "Yes"
            self.play_button.config(text="Play")
            self.next_patient()

    def mark_no(self):
        if self.data is not None:
            self.responses[self.patient_index] = "No"
            self.play_button.config(text="Play")
            self.next_patient()

    def prev_patient(self):
        if self.playing:  # Check if it's playing
            self.playing = False  # Stop playback
            self.play_button.config(text="Play")  # Reset the button text to "Play"
        if self.patient_index > 0:
            self.patient_index -= 1
            self.current_time = 0  # Reset time to zero when moving to the previous row
            self.show_patient_data()

    def next_patient(self):
        if self.patient_index < len(self.data) - 1:
            self.patient_index += 1
            self.current_time = 0  # Reset time to zero when moving to the next row
            self.play_button.config(text="Play")  # Reset the button text to "Play" when moving to the next row
            self.show_patient_data()
            self.playing = False  # Stop playback when moving to the next row

    def save_responses(self):
        if self.data is not None:
            self.data["Response"] = self.responses  # Assuming a column named "Response" for saving responses
        if hasattr(self, 'output_file_path'):
            try:
                self.data.to_csv(self.output_file_path, mode='a', header=False, index=False)
                print(f"Responses saved to '{self.output_file_path}'")
            except Exception as e:
                print(f"An error occurred while saving responses: {str(e)}")
        else:
            self.output_file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
            if self.output_file_path:
                try:
                    self.data.to_csv(self.output_file_path, index=False)
                    print(f"Responses saved to '{self.output_file_path}'")
                except Exception as e:
                    print(f"An error occurred while saving responses: {str(e)}")

    def plot_ecg(self):
        if self.playing and self.current_time <= 20:  # Check if it's playing and within 20 seconds
            self.show_patient_data()
            self.current_time += self.update_interval / 1000.0
            self.root.after(self.update_interval, self.plot_ecg)  # Schedule the next update
        else:
            self.current_time = 0  # Reset to 0 seconds when reaching the end or 20 seconds
            self.show_patient_data()


    def toggle_play_pause(self):
        if not self.playing:
            self.playing = True
            self.play_button.config(text="Pause")
            self.enable_buttons()
            self.plot_ecg()
        else:
            self.playing = False
            self.play_button.config(text="Play")

if __name__ == "__main__":
    root = tk.Tk()
    app = ReviewApp(root)
    root.mainloop()
