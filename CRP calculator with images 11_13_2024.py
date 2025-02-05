import os
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from scipy.signal import hilbert
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class CRPAnalyzer:
    def __init__(self, master):
        self.master = master
        self.master.title("CRP Analysis")
        self.master.geometry("1200x800")

        self.data_files = []
        self.merged_file = None
        self.variables = []
        self.crp_pairs = []
        self.analyze_button = None
        
        # Create a single figure with one plot
        self.fig, self.ax1 = plt.subplots(figsize=(10, 6))
        # Create a second y-axis that shares the same x-axis
        self.ax2 = self.ax1.twinx()
        self.canvas = None

        # Set up logging
        logging.basicConfig(filename='crp_analysis.log', level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')

        self.create_widgets()

    def create_widgets(self):
        # Create left panel for existing controls
        left_panel = ttk.Frame(self.master)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        self.file_frame = ttk.Frame(left_panel)
        self.file_frame.pack(pady=10)

        self.file_listbox = tk.Listbox(self.file_frame, width=50, height=10)
        self.file_listbox.pack()

        ttk.Button(self.file_frame, text="Add Data Files", command=self.add_files).pack(pady=5)
        ttk.Button(self.file_frame, text="Remove Selected", command=self.remove_files).pack(pady=5)

        ttk.Label(left_panel, text="Select Merged File:").pack()
        merged_frame = ttk.Frame(left_panel)
        merged_frame.pack(fill=tk.X, pady=5)
        
        self.merged_file_var = tk.StringVar()
        ttk.Entry(merged_frame, textvariable=self.merged_file_var, width=40).pack(side=tk.LEFT)
        ttk.Button(merged_frame, text="Browse", command=self.browse_merged_file).pack(side=tk.LEFT, padx=5)

        tk.Label(left_panel, text="Select variables for CRP:").pack()
        self.var_frame = ttk.Frame(left_panel)
        self.var_frame.pack(pady=10)

        tk.Button(left_panel, text="Add Variable Pair", command=self.add_variable_pair).pack(pady=5)

        self.analyze_button = tk.Button(left_panel, text="Analyze CRP", command=self.analyze_crp, state="disabled")
        self.analyze_button.pack(pady=20)

        # Create right panel for plots
        right_panel = ttk.Frame(self.master)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        # Initialize the canvas
        self.canvas = FigureCanvasTkAgg(self.fig, right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add initial variable pair
        self.add_variable_pair()

    def add_files(self):
        new_files = filedialog.askopenfilenames(filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")])
        for file in new_files:
            if file not in self.data_files:
                self.data_files.append(file)
                self.file_listbox.insert(tk.END, os.path.basename(file))
        
        if self.data_files:
            self.update_variable_options()

    def remove_files(self):
        selected_indices = self.file_listbox.curselection()
        for index in reversed(selected_indices):
            self.file_listbox.delete(index)
            del self.data_files[index]
        
        if not self.data_files:
            self.variables = []
            self.update_all_dropdowns()
            self.analyze_button['state'] = "disabled"

    def browse_merged_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if filename:
            self.merged_file_var.set(filename)
            self.merged_file = filename

    def update_variable_options(self):
        if self.data_files:
            first_file = self.data_files[0]
            df = pd.read_excel(first_file) if first_file.endswith('.xlsx') else pd.read_csv(first_file)
            self.variables = list(df.columns)
            self.update_all_dropdowns()
            if self.analyze_button:
                self.analyze_button['state'] = "normal"

    def update_all_dropdowns(self):
        for var1, var2, dropdown1, dropdown2, _ in self.crp_pairs:
            dropdown1['values'] = self.variables
            dropdown2['values'] = self.variables

    def add_variable_pair(self):
        pair_frame = ttk.Frame(self.var_frame)
        pair_frame.pack(pady=5)

        var1 = tk.StringVar()
        var2 = tk.StringVar()

        dropdown1 = ttk.Combobox(pair_frame, textvariable=var1, state="readonly", width=20)
        dropdown1.grid(row=0, column=0, padx=5)
        dropdown2 = ttk.Combobox(pair_frame, textvariable=var2, state="readonly", width=20)
        dropdown2.grid(row=0, column=1, padx=5)

        if self.variables:
            dropdown1['values'] = self.variables
            dropdown2['values'] = self.variables

        remove_button = ttk.Button(pair_frame, text="Remove", command=lambda: self.remove_variable_pair(pair_frame))
        remove_button.grid(row=0, column=2, padx=5)

        self.crp_pairs.append((var1, var2, dropdown1, dropdown2, pair_frame))
        
        if self.analyze_button and len(self.crp_pairs) > 0:
            self.analyze_button['state'] = "normal"

    def remove_variable_pair(self, pair_frame):
        for i, (var1, var2, dropdown1, dropdown2, frame) in enumerate(self.crp_pairs):
            if frame == pair_frame:
                self.crp_pairs.pop(i)
                pair_frame.destroy()
                break
        
        if len(self.crp_pairs) == 0:
            self.analyze_button['state'] = "disabled"
        else:
            self.analyze_button['state'] = "normal"

    def plot_crp_results(self, windowed_data, crp_data, var1_name, var2_name, file_basename):
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()

        # Convert time to percentage
        time = windowed_data['Time'].values
        time_percentage = ((time - time.min()) / (time.max() - time.min())) * 100
        
        # Plot original signals on left y-axis
        signal1_line, = self.ax1.plot(time_percentage, windowed_data[var1_name].values, 'b-', label=var1_name)
        signal2_line, = self.ax1.plot(time_percentage, windowed_data[var2_name].values, 'g-', label=var2_name)
        self.ax1.set_xlabel('Movement Completion (%)')
        self.ax1.set_ylabel('Force (N)', color='k')
        
        # Plot CRP on right y-axis
        crp_line, = self.ax2.plot(time_percentage, crp_data, 'r-', label='CRP', alpha=0.7)
        self.ax2.set_ylabel('CRP (radians/π)', color='r', rotation=270, labelpad=15)
        self.ax2.yaxis.set_label_position('right')
        self.ax2.set_ylim(0, 1)
        
        # Simplified title showing just the comparison
        self.ax1.set_title(f'{var1_name} vs {var2_name}', pad=20)
        
        # Add legends outside the plot
        lines = [signal1_line, signal2_line, crp_line]
        labels = [line.get_label() for line in lines]
        self.ax1.legend(lines, labels, 
                       loc='center left', 
                       bbox_to_anchor=(1.15, 0.5),  # Position legend to the right of the plot
                       frameon=True)

        # Set grid
        self.ax1.grid(True, alpha=0.3)
        
        # Color the tick labels and position ticks
        self.ax1.tick_params(axis='y', labelcolor='k')
        self.ax2.tick_params(axis='y', labelcolor='r', right=True, labelright=True)
        
        # Set x-axis limits from 0 to 100%
        self.ax1.set_xlim(0, 100)

        # Adjust layout with specific spacing for legend
        self.fig.tight_layout()
        # Add extra right margin for legend
        plt.subplots_adjust(top=0.9, right=0.85)
        
        self.canvas.draw()

        # Save the plot
        output_dir = os.path.dirname(self.data_files[0])
        plot_filename = f"CRP_plot_{file_basename}_{var1_name}_vs_{var2_name}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        self.fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        logging.info(f"Plot saved as: {plot_path}")

    def analyze_crp(self):
        if not self.merged_file:
            messagebox.showerror("Error", "Please select a merged file")
            return

        if not any(var1.get() and var2.get() for var1, var2, _, _, _ in self.crp_pairs):
            messagebox.showerror("Error", "Please select variables for at least one pair")
            return

        merged_df = pd.read_excel(self.merged_file)
        logging.info(f"Merged file loaded: {self.merged_file}")

        all_results = []

        for data_file in self.data_files:
            file_basename = os.path.basename(data_file)
            logging.info(f"Processing file: {file_basename}")
            
            metrics_row = self.find_matching_row(merged_df, file_basename)
            if metrics_row is None:
                continue

            file_results = {'Source File': file_basename}

            try:
                start_time = metrics_row['start_time']
                end_jump_time = metrics_row['end_jump_time']
                
                if data_file.endswith('.xlsx'):
                    data_df = pd.read_excel(data_file)
                else:
                    data_df = pd.read_csv(data_file)

                start_index = data_df[data_df['Time'] >= start_time].index[0]
                end_index = data_df[data_df['Time'] <= end_jump_time].index[-1]

                new_matrix = pd.DataFrame(index=data_df.index)
                new_matrix['Time'] = data_df['Time']

                for var1, var2, _, _, _ in self.crp_pairs:
                    if var1.get() and var2.get():
                        new_matrix[var1.get()] = data_df[var1.get()]
                        new_matrix[var2.get()] = data_df[var2.get()]

                windowed_data = new_matrix.loc[start_index:end_index]

                for var1, var2, _, _, _ in self.crp_pairs:
                    if var1.get() and var2.get():
                        try:
                            signal1 = self.normalize_signal(windowed_data[var1.get()].values)
                            signal2 = self.normalize_signal(windowed_data[var2.get()].values)

                            phase1 = self.calculate_phase_angle(signal1)
                            phase2 = self.calculate_phase_angle(signal2)

                            crp = self.calculate_crp(phase1, phase2)

                            # Plot the results
                            self.plot_crp_results(windowed_data, crp, var1.get(), var2.get(), file_basename)

                            mean_crp = np.mean(crp)
                            std_crp = np.std(crp)
                            rms_crp = np.sqrt(np.mean(np.square(crp)))

                            file_results.update({
                                f'{var1.get()} vs {var2.get()} Mean CRP': mean_crp,
                                f'{var1.get()} vs {var2.get()} Std CRP': std_crp,
                                f'{var1.get()} vs {var2.get()} RMS CRP': rms_crp
                            })

                        except Exception as e:
                            logging.error(f"Error processing {var1.get()} vs {var2.get()} for {file_basename}: {str(e)}")
                            file_results.update({
                                f'{var1.get()} vs {var2.get()} Mean CRP': 'Error',
                                f'{var1.get()} vs {var2.get()} Std CRP': 'Error',
                                f'{var1.get()} vs {var2.get()} RMS CRP': 'Error'
                            })

            except Exception as e:
                logging.error(f"Error processing file {file_basename}: {str(e)}")
                continue

            all_results.append(file_results)

        final_results_df = pd.DataFrame(all_results)
        
        numeric_columns = final_results_df.select_dtypes(include=['number']).columns
        averages = final_results_df[numeric_columns].mean()
        average_row = pd.DataFrame([averages], columns=numeric_columns)
        average_row['Source File'] = 'Average'
        final_results_df = pd.concat([final_results_df, average_row], ignore_index=True)
        
        self.save_results(final_results_df)

    def save_results(self, results_df):
        output_dir = os.path.dirname(self.data_files[0])
        output_filename = "CRP_analysis_results.xlsx"
        output_path = os.path.join(output_dir, output_filename)
        
        results_df.to_excel(output_path, index=False)
        messagebox.showinfo("Success", f"CRP analysis results saved as: {output_path}")

    def normalize_signal(self, signal):
        return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

    def calculate_phase_angle(self, signal):
        analytic_signal = hilbert(signal)
        phase_angle = np.angle(analytic_signal)
        return phase_angle

    def calculate_crp(self, phase1, phase2):
        crp = np.abs(phase1 - phase2)
        crp = np.where(crp > np.pi, 2*np.pi - crp, crp)
        return crp / np.pi

    def find_matching_row(self, merged_df, file_basename):
        # Try exact match first
        exact_match = merged_df[merged_df.iloc[:, 0] == file_basename]
        if not exact_match.empty:
            logging.info(f"Exact match found for {file_basename}")
            return exact_match.iloc[0]
        
        # If no exact match, try fuzzy matching
        logging.info(f"No exact match found for {file_basename}. Attempting fuzzy match...")
        for index, row in merged_df.iterrows():
            if self.fuzzy_match(row.iloc[0], file_basename):
                logging.info(f"Fuzzy match found for {file_basename}: {row.iloc[0]}")
                return row

        logging.warning(f"No match found for {file_basename}")
        return None

    def fuzzy_match(self, x, filename):
        # Remove file extensions and 'data' or 'metrics' suffix, but keep movement numbers
        x_clean = re.sub(r'_(data|metrics)(\.xlsx?|\.csv)$', '', x, flags=re.IGNORECASE)
        filename_clean = re.sub(r'_(data|metrics)(\.xlsx?|\.csv)$', '', filename, flags=re.IGNORECASE)
        
        return x_clean.lower() == filename_clean.lower()

if __name__ == "__main__":
    root = tk.Tk()
    app = CRPAnalyzer(root)
    root.mainloop()
