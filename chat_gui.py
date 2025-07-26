import tkinter as tk
from tkinter import scrolledtext
import requests
from playsound import playsound
import threading

# URL for backend Flask server
SERVER_URL = "http://127.0.0.1:5000/predict"

def play_alert_sound():
    threading.Thread(target=lambda: playsound("alert.wav")).start()

def send_message():
    message = entry.get()
    if message.strip():
        chat_window.insert(tk.END, f"You: {message}\n")
        entry.delete(0, tk.END)

        try:
            response = requests.post(SERVER_URL, json={"text": message})
            result = response.json()

            if result["Aggressiveness"] == "Aggressive":
                chat_window.insert(tk.END, f"⚠️ Aggression Detected! Penalty Applied\n")
                chat_window.insert(tk.END, f"Categories: {', '.join(result['Categories'])}\n\n")
                play_alert_sound()
            else:
                chat_window.insert(tk.END, "Bot: Message is clean.\n\n")

        except Exception as e:
            chat_window.insert(tk.END, f"Error: {e}\n\n")

# GUI setup
root = tk.Tk()
root.title("Chat App with Aggression Penalty")
root.geometry("450x550")

chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='normal')
chat_window.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

bottom_frame = tk.Frame(root)
bottom_frame.pack(fill=tk.X, padx=10, pady=10)

entry = tk.Entry(bottom_frame)
entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
entry.bind("<Return>", lambda event: send_message())

send_button = tk.Button(bottom_frame, text="Send", command=send_message)
send_button.pack(side=tk.RIGHT)

root.mainloop()
