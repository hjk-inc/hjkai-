import customtkinter as ctk
import sqlite3
import mysql.connector
import psycopg2
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
import plotly.graph_objects as go
import pickle
import json
import os
import threading
import queue
import time
import psutil
import warnings
from datetime import datetime
from tkinter import ttk, Menu, filedialog, messagebox
from ydata_profiling import ProfileReport
import io
import zipfile
import base64
import logging
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(filename='visualsql.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==================== CORE APPLICATION ====================
class EnterpriseVisualSQL(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Enterprise VisualSQL AI 4.0")
        self.geometry("1800x1200")
        self.minsize(1200, 800)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Application state
        self.connection = None
        self.current_data = None
        self.data_pipeline = self.AdvancedDataPipeline()
        self.model_warehouse = self.ModelWarehouse()
        self.job_queue = queue.Queue()
        self.system_monitor = self.SystemMonitor(self)
        self.current_model = None
        self.current_metrics = None
        self.recent_connections = []
        self.query_history = []
        
        # UI Setup
        self.setup_ui()
        self.start_background_workers()
        
        # Initialize components
        self.load_default_settings()
        self.setup_keybindings()
        logging.info("Application initialized")
        
    # ==================== UI SETUP ====================
    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Theme toggle
        self.theme_var = ctk.StringVar(value="System")
        theme_menu = ctk.CTkOptionMenu(self, values=["System", "Light", "Dark"], variable=self.theme_var, command=lambda _: ctk.set_appearance_mode(self.theme_var.get()))
        theme_menu.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=2)
        
        # ===== LEFT SIDEBAR =====
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar.grid_propagate(False)
        
        # Database Explorer
        self.setup_database_explorer()
        
        # AI Tools Section
        self.setup_ai_workbench()
        
        # Data Pipeline Config
        self.setup_data_pipeline_ui()
        
        # Collaboration Tools
        self.setup_collaboration_tools()
        
        # ===== MAIN NOTEBOOK =====
        self.notebook = ctk.CTkTabview(self)
        self.notebook.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # SQL Workspace
        self.setup_sql_workspace()
        
        # Data Explorer
        self.setup_professional_data_viewer()
        
        # Visualization Studio
        self.setup_visualization_studio()
        
        # Model Training Center
        self.setup_model_training_center()
        
        # Data Profiling Dashboard
        self.setup_data_profiling()
        
        # Feature Engineering Toolkit
        self.setup_feature_engineering()
        
        # ===== STATUS BAR =====
        self.status_bar = ctk.CTkLabel(self, text="System Ready | CPU: 0% | RAM: 0MB | Disk: 0MB", anchor="w", height=20)
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        
        # Start system monitoring
        self.system_monitor.start()
        
        # Show onboarding
        self.show_onboarding()
        
    def setup_database_explorer(self):
        """Enhanced database connection and table explorer"""
        explorer_frame = ctk.CTkFrame(self.sidebar)
        explorer_frame.pack(fill="x", padx=5, pady=5, expand=False)
        
        ctk.CTkLabel(explorer_frame, text="Database Explorer", font=ctk.CTkFont(weight="bold")).pack(pady=(0,10))
        
        # Database type selection
        self.db_type = ctk.CTkComboBox(explorer_frame, values=["SQLite", "MySQL", "PostgreSQL", "CSV"], state="readonly")
        self.db_type.set("SQLite")
        self.db_type.pack(fill="x", padx=5, pady=2)
        
        # Connection controls
        conn_frame = ctk.CTkFrame(explorer_frame)
        conn_frame.pack(fill="x", pady=(0,10))
        
        self.db_path_entry = ctk.CTkEntry(conn_frame, placeholder_text="Database path or connection string")
        self.db_path_entry.pack(fill="x", padx=5, pady=2)
        self.db_path_entry.configure(tooltip="Enter SQLite file path, MySQL/PostgreSQL connection string, or CSV file path")
        
        browse_btn = ctk.CTkButton(conn_frame, text="Browse", width=80, command=self.browse_database)
        browse_btn.pack(side="right", padx=5)
        browse_btn.configure(tooltip="Browse for database or CSV file")
        
        connect_btn = ctk.CTkButton(conn_frame, text="Connect", width=80, command=self.connect_db)
        connect_btn.pack(side="right", padx=5)
        connect_btn.configure(tooltip="Connect to the selected database")
        
        # Recent connections
        self.recent_connections_combo = ctk.CTkComboBox(explorer_frame, values=[], state="readonly", command=self.load_recent_connection)
        self.recent_connections_combo.pack(fill="x", padx=5, pady=2)
        
        # Table list
        table_list_frame = ctk.CTkFrame(explorer_frame)
        table_list_frame.pack(fill="both", expand=True)
        
        self.table_tree = ttk.Treeview(table_list_frame, columns=("type", "rows", "size"), show="tree headings", height=15)
        self.table_tree.heading("#0", text="Tables", anchor="w")
        self.table_tree.heading("type", text="Type")
        self.table_tree.heading("rows", text="Rows")
        self.table_tree.heading("size", text="Size")
        
        vsb = ttk.Scrollbar(table_list_frame, orient="vertical", command=self.table_tree.yview)
        hsb = ttk.Scrollbar(table_list_frame, orient="horizontal", command=self.table_tree.xview)
        self.table_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        self.table_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        table_list_frame.grid_columnconfigure(0, weight=1)
        table_list_frame.grid_rowconfigure(0, weight=1)
        
        # Context menu
        self.table_menu = Menu(self, tearoff=0)
        self.table_menu.add_command(label="Preview Data", command=self.preview_table)
        self.table_menu.add_command(label="Show Schema", command=self.show_schema)
        self.table_menu.add_command(label="Export Table", command=self.export_table)
        self.table_menu.add_command(label="Visualize Schema", command=self.visualize_schema)
        self.table_tree.bind("<Button-3>", self.show_table_context_menu)
        
    def setup_ai_workbench(self):
        """Professional AI toolset with model management"""
        ai_frame = ctk.CTkFrame(self.sidebar)
        ai_frame.pack(fill="x", padx=5, pady=5, expand=False)
        
        ctk.CTkLabel(ai_frame, text="AI Workbench", font=ctk.CTkFont(weight="bold")).pack(pady=(0,10))
        
        # Model selection
        model_select_frame = ctk.CTkFrame(ai_frame)
        model_select_frame.pack(fill="x", pady=(0,5))
        
        self.model_type = ctk.CTkComboBox(model_select_frame, values=["Random Forest", "Isolation Forest"], state="readonly")
        self.model_type.pack(fill="x", padx=5, pady=2)
        self.model_type.configure(tooltip="Select machine learning model type")
        
        # Target selection
        self.target_var = ctk.CTkComboBox(model_select_frame, values=[], state="readonly")
        self.target_var.pack(fill="x", padx=5, pady=2)
        self.target_var.configure(tooltip="Select target variable for supervised learning")
        
        # Training controls
        train_btn = ctk.CTkButton(ai_frame, text="Train Model", command=self.train_selected_model)
        train_btn.pack(fill="x", padx=5, pady=2)
        train_btn.configure(tooltip="Train the selected model")
        
        # Model management
        model_mgmt_frame = ctk.CTkFrame(ai_frame)
        model_mgmt_frame.pack(fill="x", pady=(5,0))
        
        save_btn = ctk.CTkButton(model_mgmt_frame, text="Save Model", width=100, command=self.save_current_model)
        save_btn.pack(side="left", padx=2)
        save_btn.configure(tooltip="Save the trained model to disk")
        
        load_btn = ctk.CTkButton(model_mgmt_frame, text="Load Model", width=100, command=self.load_saved_model)
        load_btn.pack(side="left", padx=2)
        load_btn.configure(tooltip="Load a saved model from disk")
        
        predict_btn = ctk.CTkButton(model_mgmt_frame, text="Predict", width=100, command=self.make_predictions)
        predict_btn.pack(side="left", padx=2)
        predict_btn.configure(tooltip="Make predictions with the current model")
        
    def setup_data_pipeline_ui(self):
        """Data pipeline configuration UI"""
        pipeline_frame = ctk.CTkFrame(self.sidebar)
        pipeline_frame.pack(fill="x", padx=5, pady=5, expand=False)
        
        ctk.CTkLabel(pipeline_frame, text="Data Pipeline", font=ctk.CTkFont(weight="bold")).pack(pady=(0,10))
        
        self.impute_strategy = ctk.CTkComboBox(pipeline_frame, values=["mean", "median", "most_frequent"], state="readonly")
        self.impute_strategy.set("median")
        self.impute_strategy.pack(fill="x", padx=5, pady=2)
        self.impute_strategy.configure(tooltip="Select imputation strategy for missing values")
        
        self.scaler_type = ctk.CTkComboBox(pipeline_frame, values=["StandardScaler", "MinMaxScaler", "RobustScaler"], state="readonly")
        self.scaler_type.set("StandardScaler")
        self.scaler_type.pack(fill="x", padx=5, pady=2)
        self.scaler_type.configure(tooltip="Select scaling method for numerical features")
        
        outlier_btn = ctk.CTkButton(pipeline_frame, text="Remove Outliers", command=self.remove_outliers)
        outlier_btn.pack(fill="x", padx=5, pady=2)
        outlier_btn.configure(tooltip="Remove outliers using Isolation Forest")
        
    def setup_collaboration_tools(self):
        """Collaboration and project management tools"""
        collab_frame = ctk.CTkFrame(self.sidebar)
        collab_frame.pack(fill="x", padx=5, pady=5, expand=False)
        
        ctk.CTkLabel(collab_frame, text="Collaboration", font=ctk.CTkFont(weight="bold")).pack(pady=(0,10))
        
        save_project_btn = ctk.CTkButton(collab_frame, text="Save Project", command=self.save_project)
        save_project_btn.pack(fill="x", padx=5, pady=2)
        save_project_btn.configure(tooltip="Save current workspace as a project")
        
        load_project_btn = ctk.CTkButton(collab_frame, text="Load Project", command=self.load_project)
        load_project_btn.pack(fill="x", padx=5, pady=2)
        load_project_btn.configure(tooltip="Load a saved project")
        
        feedback_btn = ctk.CTkButton(collab_frame, text="Send Feedback", command=self.send_feedback)
        feedback_btn.pack(fill="x", padx=5, pady=2)
        feedback_btn.configure(tooltip="Send feedback to developers")
        
    def setup_sql_workspace(self):
        """SQL query editor and execution"""
        tab = self.notebook.add("SQL Workspace")
        query_frame = ctk.CTkFrame(tab)
        query_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.query_editor = ctk.CTkTextbox(query_frame, height=200)
        self.query_editor.pack(fill="x", padx=5, pady=5)
        self.query_editor.configure(tooltip="Enter SQL query here")
        
        query_btn_frame = ctk.CTkFrame(query_frame)
        query_btn_frame.pack(fill="x", padx=5, pady=2)
        
        execute_btn = ctk.CTkButton(query_btn_frame, text="Execute Query (Ctrl+E)", command=self.execute_query)
        execute_btn.pack(side="left", padx=5)
        execute_btn.configure(tooltip="Execute the current query")
        
        history_btn = ctk.CTkButton(query_btn_frame, text="Query History", command=self.show_query_history)
        history_btn.pack(side="left", padx=5)
        history_btn.configure(tooltip="View and reuse past queries")
        
        self.query_progress = ctk.CTkProgressBar(query_frame)
        self.query_progress.pack(fill="x", padx=5, pady=5)
        
    def setup_professional_data_viewer(self):
        """Data viewer with treeview"""
        tab = self.notebook.add("Data Explorer")
        self.data_tree = ttk.Treeview(tab, show="headings")
        self.data_tree.pack(fill="both", expand=True, padx=5, pady=5)
        vsb = ttk.Scrollbar(tab, orient="vertical", command=self.data_tree.yview)
        hsb = ttk.Scrollbar(tab, orient="horizontal", command=self.data_tree.xview)
        self.data_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        
        export_btn = ctk.CTkButton(tab, text="Export Data", command=self.export_data)
        export_btn.pack(pady=5)
        export_btn.configure(tooltip="Export current data to CSV")
        
    def setup_visualization_studio(self):
        """Visualization studio with Matplotlib and Plotly"""
        tab = self.notebook.add("Visualization Studio")
        self.vis_frame = ctk.CTkFrame(tab)
        self.vis_frame.pack(fill="both", expand=True, padx=5, pady=5)
        fig, ax = plt.subplots(figsize=(8, 6))
        self.vis_canvas = FigureCanvasTkAgg(fig, master=self.vis_frame)
        self.vis_canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.vis_canvas, self.vis_frame)
        toolbar.update()
        self.vis_ax = ax
        self.vis_canvas.draw()
        
        # Plot controls
        plot_frame = ctk.CTkFrame(tab)
        plot_frame.pack(fill="x", padx=5, pady=5)
        
        self.plot_type = ctk.CTkComboBox(plot_frame, values=["Scatter", "Histogram", "Box", "Correlation Matrix", "Line"], state="readonly")
        self.plot_type.pack(side="left", padx=5)
        self.plot_type.configure(tooltip="Select plot type")
        
        self.x_var = ctk.CTkComboBox(plot_frame, values=[], state="readonly")
        self.x_var.pack(side="left", padx=5)
        self.x_var.configure(tooltip="Select X-axis variable")
        
        self.y_var = ctk.CTkComboBox(plot_frame, values=[], state="readonly")
        self.y_var.pack(side="left", padx=5)
        self.y_var.configure(tooltip="Select Y-axis variable")
        
        plot_btn = ctk.CTkButton(plot_frame, text="Plot", command=self.generate_plot)
        plot_btn.pack(side="left", padx=5)
        plot_btn.configure(tooltip="Generate the selected plot")
        
        export_plot_btn = ctk.CTkButton(plot_frame, text="Export Plot", command=self.export_plot)
        export_plot_btn.pack(side="left", padx=5)
        export_plot_btn.configure(tooltip="Export plot as PNG or HTML")
        
    def setup_model_training_center(self):
        """Model training interface"""
        tab = self.notebook.add("Model Training Center")
        self.train_progress = ctk.CTkProgressBar(tab)
        self.train_progress.pack(fill="x", padx=5, pady=5)
        self.metrics_label = ctk.CTkLabel(tab, text="No model trained")
        self.metrics_label.pack(pady=5)
        
        # Model comparison
        compare_frame = ctk.CTkFrame(tab)
        compare_frame.pack(fill="x", padx=5, pady=5)
        compare_btn = ctk.CTkButton(compare_frame, text="Compare Models", command=self.compare_models)
        compare_btn.pack(pady=5)
        compare_btn.configure(tooltip="Compare all saved models")
        
    def setup_data_profiling(self):
        """Data profiling dashboard"""
        tab = self.notebook.add("Data Profiling")
        self.profile_text = ctk.CTkTextbox(tab, height=400)
        self.profile_text.pack(fill="both", expand=True, padx=5, pady=5)
        profile_btn = ctk.CTkButton(tab, text="Generate Profile", command=self.generate_data_profile)
        profile_btn.pack(pady=5)
        profile_btn.configure(tooltip="Generate detailed data profile")
        
    def setup_feature_engineering(self):
        """Feature engineering toolkit"""
        tab = self.notebook.add("Feature Engineering")
        fe_frame = ctk.CTkFrame(tab)
        fe_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(fe_frame, text="Feature Engineering", font=ctk.CTkFont(weight="bold")).pack(pady=(0,10))
        
        poly_btn = ctk.CTkButton(fe_frame, text="Add Polynomial Features", command=self.add_polynomial_features)
        poly_btn.pack(fill="x", padx=5, pady=2)
        poly_btn.configure(tooltip="Add polynomial features up to degree 2")
        
        interaction_btn = ctk.CTkButton(fe_frame, text="Add Interaction Terms", command=self.add_interaction_terms)
        interaction_btn.pack(fill="x", padx=5, pady=2)
        interaction_btn.configure(tooltip="Add interaction terms between features")
        
    # ==================== CORE FUNCTIONALITY ====================
    def connect_db(self):
        """Connect to database or load CSV"""
        db_type = self.db_type.get()
        db_path = self.db_path_entry.get().strip()
        if not db_path:
            self.show_error("Please enter a database path or connection string")
            return
            
        try:
            if self.connection:
                self.connection.close()
                
            if db_type == "SQLite":
                self.connection = sqlite3.connect(
                    db_path,
                    timeout=30,
                    detect_types=sqlite3.PARSE_DECLTYPES,
                    isolation_level=None,
                    check_same_thread=False
                )
                self.connection.execute("PRAGMA journal_mode=WAL")
                self.connection.execute("PRAGMA synchronous=NORMAL")
            elif db_type == "MySQL":
                self.connection = mysql.connector.connect(
                    host=db_path.split(":")[0],
                    user="user",  # Replace with user input or config
                    password="password",
                    database=db_path.split(":")[-1]
                )
            elif db_type == "PostgreSQL":
                self.connection = psycopg2.connect(
                    host=db_path.split(":")[0],
                    user="user",
                    password="password",
                    dbname=db_path.split(":")[-1]
                )
            elif db_type == "CSV":
                self.current_data = pd.read_csv(db_path)
                self.update_data_viewer()
                self.update_target_var()
                self.status_bar.configure(text=f"Loaded CSV: {os.path.basename(db_path)}")
                self.add_recent_connection(f"CSV:{db_path}")
                return
                
            self.populate_table_list()
            self.status_bar.configure(text=f"Connected to {os.path.basename(db_path)} ({db_type})")
            self.add_recent_connection(f"{db_type}:{db_path}")
            logging.info(f"Connected to {db_type}: {db_path}")
            
        except Exception as e:
            self.show_error(f"Connection failed: {str(e)}")
            if self.connection:
                self.connection.close()
            self.connection = None
            logging.error(f"Connection failed: {str(e)}")
            
    def execute_query(self):
        """Thread-safe query execution"""
        query = self.query_editor.get("1.0", "end").strip()
        if not query or not self.connection:
            self.show_error("No query or database connection")
            return
            
        self.query_history.append(query)
        self.status_bar.configure(text="Executing query...")
        self.query_progress.start()
        
        def query_task():
            try:
                start_time = time.time()
                chunks = []
                for chunk in pd.read_sql(query, self.connection, chunksize=10000, parse_dates=True, coerce_float=True):
                    chunks.append(chunk)
                self.current_data = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
                
                self.job_queue.put((
                    "query_complete",
                    {
                        "row_count": len(self.current_data),
                        "time_elapsed": time.time() - start_time
                    }
                ))
                
            except Exception as e:
                self.job_queue.put(("error", f"Query failed: {str(e)}"))
            finally:
                self.job_queue.put(("progress_complete", None))
                
        threading.Thread(target=query_task, daemon=True).start()
        
    def train_selected_model(self):
        """Train selected model"""
        if self.current_data is None or len(self.current_data) < 10:
            self.show_error("Not enough data for training")
            return
            
        model_type = self.model_type.get()
        target = self.target_var.get() if model_type == "Random Forest" else None
        
        try:
            X, y = self.prepare_training_data(target) if model_type == "Random Forest" else (self.current_data, None)
            self.status_bar.configure(text=f"Training {model_type}...")
            self.train_progress.start()
            
            def train_task():
                try:
                    start_time = time.time()
                    result = None
                    
                    if model_type == "Random Forest":
                        model = RandomForestRegressor(n_estimators=100, n_jobs=-1)
                        model.fit(X, y)
                        result = {
                            "model": model,
                            "metrics": {
                                "r2_score": model.score(X, y),
                                "mse": mean_squared_error(y, model.predict(X)),
                                "feature_importance": dict(zip(self.current_data.drop(columns=[target]).columns, model.feature_importances_))
                            }
                        }
                    elif model_type == "Isolation Forest":
                        model = IsolationForest(contamination=0.1)
                        model.fit(X)
                        result = {
                            "model": model,
                            "metrics": {"outliers_detected": np.sum(model.predict(X) == -1)}
                        }
                        
                    self.job_queue.put((
                        "training_complete",
                        {
                            "model_type": model_type,
                            "result": result,
                            "time_elapsed": time.time() - start_time
                        }
                    ))
                    
                except Exception as e:
                    self.job_queue.put(("error", f"Training failed: {str(e)}"))
                finally:
                    self.job_queue.put(("progress_complete", None))
                    
            threading.Thread(target=train_task, daemon=True).start()
            
        except Exception as e:
            self.show_error(f"Training setup failed: {str(e)}")
            
    def browse_database(self):
        """Browse for database or CSV file"""
        db_type = self.db_type.get()
        filetypes = [("All Files", "*.*")]
        if db_type == "SQLite":
            filetypes = [("SQLite Database", "*.db *.sqlite *.sqlite3")]
        elif db_type == "CSV":
            filetypes = [("CSV Files", "*.csv")]
        db_path = filedialog.askopenfilename(filetypes=filetypes)
        if db_path:
            self.db_path_entry.delete(0, "end")
            self.db_path_entry.insert(0, db_path)
            
    def preview_table(self):
        """Preview selected table data"""
        selected = self.table_tree.selection()
        if not selected:
            self.show_error("No table selected")
            return
        table_name = self.table_tree.item(selected[0])["text"]
        query = f"SELECT * FROM {table_name} LIMIT 100"
        self.query_editor.delete("1.0", "end")
        self.query_editor.insert("1.0", query)
        self.execute_query()
        
    def show_schema(self):
        """Show schema of selected table"""
        selected = self.table_tree.selection()
        if not selected:
            self.show_error("No table selected")
            return
        table_name = self.table_tree.item(selected[0])["text"]
        cursor = self.connection.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        schema = cursor.fetchall()
        schema_text = f"Schema for {table_name}:\n" + "\n".join([f"{col[1]} ({col[2]})" for col in schema])
        self.profile_text.delete("1.0", "end")
        self.profile_text.insert("1.0", schema_text)
        self.notebook.set("Data Profiling")
        
    def export_table(self):
        """Export selected table to CSV"""
        selected = self.table_tree.selection()
        if not selected:
            self.show_error("No table selected")
            return
        table_name = self.table_tree.item(selected[0])["text"]
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                df = pd.read_sql(f"SELECT * FROM {table_name}", self.connection)
                df.to_csv(file_path, index=False)
                self.status_bar.configure(text=f"Table {table_name} exported to {file_path}")
                logging.info(f"Exported table {table_name} to {file_path}")
            except Exception as e:
                self.show_error(f"Export failed: {str(e)}")
                
    def visualize_schema(self):
        """Visualize database schema (placeholder)"""
        self.show_error("Schema visualization not implemented yet")
        # Future: Implement ER diagram using Matplotlib or Graphviz
        
    def show_table_context_menu(self, event):
        """Show context menu for table treeview"""
        self.table_tree.selection_set(self.table_tree.identify_row(event.y))
        self.table_menu.post(event.x_root, event.y_root)
        
    def save_current_model(self):
        """Save the current model to disk"""
        if not hasattr(self, "current_model"):
            self.show_error("No model to save")
            return
        model_type = self.model_type.get()
        file_path = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle Files", "*.pkl")])
        if file_path:
            try:
                metadata = {
                    "metrics": self.current_metrics,
                    "features": self.current_data.columns.tolist()
                }
                model_key = self.model_warehouse.save_model(model_type, self.current_model, metadata)
                with open(file_path, "wb") as f:
                    pickle.dump(self.current_model, f)
                with open(f"{file_path}_meta.json", "w") as f:
                    json.dump(self.model_warehouse.metrics[model_key], f)
                self.status_bar.configure(text=f"Model saved as {model_key}")
                logging.info(f"Model saved: {model_key}")
            except Exception as e:
                self.show_error(f"Save failed: {str(e)}")
                
    def load_saved_model(self):
        """Load a saved model from disk"""
        file_path = filedialog.askopenfilename(filetypes=[("Pickle Files", "*.pkl")])
        if file_path:
            try:
                with open(file_path, "rb") as f:
                    model = pickle.load(f)
                model_type = self.model_type.get()
                model_key = self.model_warehouse.save_model(model_type, model, {})
                self.current_model = model
                self.status_bar.configure(text=f"Model loaded: {model_key}")
                logging.info(f"Model loaded: {model_key}")
            except Exception as e:
                self.show_error(f"Failed to load model: {str(e)}")
                
    def make_predictions(self):
        """Make predictions with the current model"""
        if not hasattr(self, "current_model") or self.current_data is None:
            self.show_error("No model or data available")
            return
        try:
            X = self.data_pipeline.build_preprocessor(self.current_data).fit_transform(self.current_data)
            predictions = self.current_model.predict(X)
            self.current_data["Predictions"] = predictions
            self.update_data_viewer()
            self.status_bar.configure(text="Predictions added to data")
            logging.info("Predictions made and added to data")
        except Exception as e:
            self.show_error(f"Prediction failed: {str(e)}")
            
    def generate_plot(self):
        """Generate visualization based on selected plot type"""
        if self.current_data is None:
            self.show_error("No data to plot")
            return
        self.vis_ax.clear()
        plot_type = self.plot_type.get()
        x_var = self.x_var.get() if self.x_var.get() else self.current_data.columns[0]
        y_var = self.y_var.get() if self.y_var.get() else (self.current_data.columns[1] if len(self.current_data.columns) > 1 else None)
        
        try:
            if plot_type == "Scatter" and y_var:
                sns.scatterplot(data=self.current_data, x=x_var, y=y_var, ax=self.vis_ax)
            elif plot_type == "Histogram":
                self.current_data[x_var].hist(bins=30, ax=self.vis_ax)
            elif plot_type == "Box" and y_var:
                sns.boxplot(data=self.current_data, x=x_var, y=y_var, ax=self.vis_ax)
            elif plot_type == "Correlation Matrix":
                corr = self.current_data.corr(numeric_only=True)
                sns.heatmap(corr, annot=True, cmap="coolwarm", ax=self.vis_ax)
            elif plot_type == "Line" and y_var:
                sns.lineplot(data=self.current_data, x=x_var, y=y_var, ax=self.vis_ax)
            self.vis_ax.set_title(plot_type)
            self.vis_canvas.draw()
            logging.info(f"Generated {plot_type} plot")
        except Exception as e:
            self.show_error(f"Plotting failed: {str(e)}")
            
    def export_plot(self):
        """Export current plot"""
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("HTML", "*.html")])
        if file_path:
            try:
                if file_path.endswith(".png"):
                    self.vis_canvas.figure.savefig(file_path)
                else:
                    fig = go.Figure()
                    plot_type = self.plot_type.get()
                    x_var = self.x_var.get() or self.current_data.columns[0]
                    y_var = self.y_var.get() or (self.current_data.columns[1] if len(self.current_data.columns) > 1 else None)
                    if plot_type == "Scatter" and y_var:
                        fig.add_trace(go.Scatter(x=self.current_data[x_var], y=self.current_data[y_var], mode="markers"))
                    fig.write_html(file_path)
                self.status_bar.configure(text=f"Plot exported to {file_path}")
                logging.info(f"Plot exported to {file_path}")
            except Exception as e:
                self.show_error(f"Export failed: {str(e)}")
                
    def generate_data_profile(self):
        """Generate data profile report"""
        if self.current_data is None:
            self.show_error("No data to profile")
            return
        try:
            profile = ProfileReport(self.current_data, minimal=True)
            profile_text = profile.to_notebook_iframe()
            self.profile_text.delete("1.0", "end")
            self.profile_text.insert("1.0", "Profile generated. View in browser or export as HTML.")
            profile.to_file("profile.html")
            self.status_bar.configure(text="Data profile generated. Open profile.html for details.")
            logging.info("Data profile generated")
        except Exception as e:
            self.show_error(f"Profiling failed: {str(e)}")
            
    def add_polynomial_features(self):
        """Add polynomial features to data"""
        if self.current_data is None:
            self.show_error("No data available")
            return
        try:
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(degree=2, include_bias=False)
            numeric_cols = self.current_data.select_dtypes(include=['int64', 'float64']).columns
            poly_features = poly.fit_transform(self.current_data[numeric_cols])
            poly_cols = poly.get_feature_names_out(numeric_cols)
            self.current_data = pd.concat([self.current_data, pd.DataFrame(poly_features, columns=poly_cols)], axis=1)
            self.update_data_viewer()
            self.status_bar.configure(text="Polynomial features added")
            logging.info("Added polynomial features")
        except Exception as e:
            self.show_error(f"Failed to add polynomial features: {str(e)}")
            
    def add_interaction_terms(self):
        """Add interaction terms to data"""
        if self.current_data is None:
            self.show_error("No data available")
            return
        try:
            numeric_cols = self.current_data.select_dtypes(include=['int64', 'float64']).columns
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    self.current_data[f"{col1}_{col2}_interaction"] = self.current_data[col1] * self.current_data[col2]
            self.update_data_viewer()
            self.status_bar.configure(text="Interaction terms added")
            logging.info("Added interaction terms")
        except Exception as e:
            self.show_error(f"Failed to add interaction terms: {str(e)}")
            
    def remove_outliers(self):
        """Remove outliers using Isolation Forest"""
        if self.current_data is None:
            self.show_error("No data available")
            return
        try:
            X = self.data_pipeline.build_preprocessor(self.current_data).fit_transform(self.current_data)
            model = IsolationForest(contamination=0.1)
            outliers = model.fit_predict(X)
            self.current_data = self.current_data[outliers == 1].reset_index(drop=True)
            self.update_data_viewer()
            self.status_bar.configure(text=f"Removed {np.sum(outliers == -1)} outliers")
            logging.info(f"Removed {np.sum(outliers == -1)} outliers")
        except Exception as e:
            self.show_error(f"Outlier removal failed: {str(e)}")
            
    def save_project(self):
        """Save current workspace as a project"""
        file_path = filedialog.asksaveasfilename(defaultextension=".zip", filetypes=[("ZIP Files", "*.zip")])
        if file_path:
            try:
                with zipfile.ZipFile(file_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    if self.current_data is not None:
                        zf.writestr("data.csv", self.current_data.to_csv(index=False))
                    zf.writestr("queries.json", json.dumps(self.query_history))
                    if hasattr(self, "current_model"):
                        zf.writestr("model.pkl", pickle.dumps(self.current_model))
                        zf.writestr("model_meta.json", json.dumps(self.current_metrics))
                self.status_bar.configure(text=f"Project saved to {file_path}")
                logging.info(f"Project saved to {file_path}")
            except Exception as e:
                self.show_error(f"Save project failed: {str(e)}")
                
    def load_project(self):
        """Load a saved project"""
        file_path = filedialog.askopenfilename(filetypes=[("ZIP Files", "*.zip")])
        if file_path:
            try:
                with zipfile.ZipFile(file_path, "r") as zf:
                    if "data.csv" in zf.namelist():
                        self.current_data = pd.read_csv(zf.open("data.csv"))
                        self.update_data_viewer()
                        self.update_target_var()
                    if "queries.json" in zf.namelist():
                        self.query_history = json.loads(zf.read("queries.json").decode())
                    if "model.pkl" in zf.namelist():
                        self.current_model = pickle.load(zf.open("model.pkl"))
                        self.current_metrics = json.loads(zf.read("model_meta.json").decode())
                self.status_bar.configure(text=f"Project loaded from {file_path}")
                logging.info(f"Project loaded from {file_path}")
            except Exception as e:
                self.show_error(f"Load project failed: {str(e)}")
                
    def send_feedback(self):
        """Send user feedback (placeholder)"""
        feedback = ctk.CTkInputDialog(title="Feedback", text="Enter your feedback:")
        if feedback.get_input():
            logging.info(f"User feedback: {feedback.get_input()}")
            self.status_bar.configure(text="Feedback submitted")
            
    def show_query_history(self):
        """Show query history in a dialog"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Query History")
        history_list = ctk.CTkTextbox(dialog, height=300, width=600)
        history_list.pack(padx=10, pady=10)
        history_list.insert("1.0", "\n".join(self.query_history[-10:]))
        use_btn = ctk.CTkButton(dialog, text="Use Query", command=lambda: self.use_query_from_history(history_list.get("sel.first", "sel.last")))
        use_btn.pack(pady=5)
        
    def use_query_from_history(self, query):
        """Use selected query from history"""
        self.query_editor.delete("1.0", "end")
        self.query_editor.insert("1.0", query.strip())
        
    def export_data(self):
        """Export current data to CSV"""
        if self.current_data is None:
            self.show_error("No data to export")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.current_data.to_csv(file_path, index=False)
                self.status_bar.configure(text=f"Data exported to {file_path}")
                logging.info(f"Data exported to {file_path}")
            except Exception as e:
                self.show_error(f"Export failed: {str(e)}")
                
    def compare_models(self):
        """Compare all saved models"""
        report = self.model_warehouse.compare_models()
        self.profile_text.delete("1.0", "end")
        self.profile_text.insert("1.0", report.to_string())
        self.notebook.set("Data Profiling")
        logging.info("Generated model comparison report")
        
    def show_error(self, message):
        """Show error message in a dialog"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Error")
        ctk.CTkLabel(dialog, text=message).pack(padx=10, pady=10)
        ctk.CTkButton(dialog, text="OK", command=dialog.destroy).pack(pady=5)
        logging.error(f"Error: {message}")
        
    def prepare_training_data(self, target):
        """Prepare data for model training"""
        if target not in self.current_data.columns:
            raise ValueError("Target variable not found")
        X = self.current_data.drop(columns=[target])
        y = self.current_data[target]
        preprocessor = self.data_pipeline.build_preprocessor(X)
        X_transformed = preprocessor.fit_transform(X)
        return X_transformed, y
        
    def populate_table_list(self):
        """Populate table list in treeview"""
        if not self.connection:
            return
        self.table_tree.delete(*self.table_tree.get_children())
        cursor = self.connection.cursor()
        cursor.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view')")
        for name, type_ in cursor.fetchall():
            cursor.execute(f"SELECT COUNT(*) FROM {name}")
            rows = cursor.fetchone()[0]
            size = os.path.getsize(self.db_path_entry.get()) / (1024 ** 2) if self.db_type.get() == "SQLite" else "N/A"
            self.table_tree.insert("", "end", text=name, values=(type_, rows, f"{size:.2f} MB" if size != "N/A" else size))
            
    def add_recent_connection(self, db_path):
        """Add database to recent connections"""
        if db_path not in self.recent_connections:
            self.recent_connections.append(db_path)
            self.recent_connections_combo.configure(values=self.recent_connections[-5:])
            with open("recent_connections.json", "w") as f:
                json.dump(self.recent_connections, f)
                
    def load_recent_connection(self, db_path):
        """Load recent connection"""
        self.db_path_entry.delete(0, "end")
        self.db_path_entry.insert(0, db_path.split(":", 1)[1])
        self.db_type.set(db_path.split(":", 1)[0])
        
    def load_default_settings(self):
        """Load default application settings"""
        try:
            with open("recent_connections.json", "r") as f:
                self.recent_connections = json.load(f)
                self.recent_connections_combo.configure(values=self.recent_connections[-5:])
        except FileNotFoundError:
            pass
            
    def setup_keybindings(self):
        """Setup keyboard shortcuts"""
        self.bind("<Control-e>", lambda event: self.execute_query())
        self.bind("<Control-s>", lambda event: self.save_project())
        self.bind("<Control-o>", lambda event: self.load_project())
        
    def start_background_workers(self):
        """Start background queue processing"""
        self.process_job_queue()
        
    def process_job_queue(self):
        """Process background job queue"""
        try:
            while True:
                msg_type, data = self.job_queue.get_nowait()
                if msg_type == "query_complete":
                    self.status_bar.configure(text=f"Query completed: {data['row_count']} rows in {data['time_elapsed']:.2f}s")
                    self.update_data_viewer()
                    self.update_target_var()
                    self.update_plot_vars()
                elif msg_type == "training_complete":
                    self.status_bar.configure(text=f"{data['model_type']} trained in {data['time_elapsed']:.2f}s")
                    self.current_model = data["result"]["model"]
                    self.current_metrics = data["result"]["metrics"]
                    metrics_text = "\n".join([f"{k}: {v}" for k, v in data["result"]["metrics"].items()])
                    self.metrics_label.configure(text=f"Metrics:\n{metrics_text}")
                    self.model_warehouse.save_model(data['model_type'], data['result']['model'], {"metrics": data['result']['metrics']})
                elif msg_type == "system_update":
                    self.status_bar.configure(text=f"CPU: {data['cpu']:.1f}% | RAM: {data['memory']:.0f}MB | Disk: {data['disk']:.0f}MB")
                elif msg_type == "error":
                    self.show_error(data)
                elif msg_type == "progress_complete":
                    self.query_progress.stop()
                    self.train_progress.stop()
        except queue.Empty:
            pass
        self.after(100, self.process_job_queue)
        
    def update_data_viewer(self):
        """Update data viewer with current data"""
        if self.current_data is not None:
            self.data_tree["columns"] = list(self.current_data.columns)
            for col in self.current_data.columns:
                self.data_tree.heading(col, text=col)
                self.data_tree.column(col, width=100, anchor="w")
            self.data_tree.delete(*self.data_tree.get_children())
            for row in self.current_data.head(100).itertuples(index=False):
                self.data_tree.insert("", "end", values=row)
                
    def update_target_var(self):
        """Update target variable dropdown"""
        if self.current_data is not None:
            self.target_var.configure(values=list(self.current_data.columns))
            if self.current_data.columns.any():
                self.target_var.set(self.current_data.columns[0])
                
    def update_plot_vars(self):
        """Update plot variable dropdowns"""
        if self.current_data is not None:
            self.x_var.configure(values=list(self.current_data.columns))
            self.y_var.configure(values=list(self.current_data.columns))
            if self.current_data.columns.any():
                self.x_var.set(self.current_data.columns[0])
                self.y_var.set(self.current_data.columns[1] if len(self.current_data.columns) > 1 else self.current_data.columns[0])
                
    def show_onboarding(self):
        """Show onboarding tutorial"""
        if not os.path.exists("onboarding_shown.txt"):
            dialog = ctk.CTkToplevel(self)
            dialog.title("Welcome to Enterprise VisualSQL AI 4.0")
            ctk.CTkLabel(dialog, text="Welcome! Here's a quick guide:\n1. Connect to a database in the sidebar.\n2. Write and execute SQL queries.\n3. Explore data, train models, and create visualizations.\n4. Use collaboration tools to save/load projects.").pack(padx=10, pady=10)
            ctk.CTkButton(dialog, text="OK", command=lambda: [dialog.destroy(), open("onboarding_shown.txt", "w").close()]).pack(pady=5)
            
    def on_close(self):
        """Handle application closing"""
        if self.connection:
            self.connection.close()
        self.system_monitor.stop()
        self.destroy()
        logging.info("Application closed")

    # ==================== ENTERPRISE COMPONENTS ====================
    class AdvancedDataPipeline:
        """Industrial-strength data preprocessing"""
        def __init__(self):
            self.impute_strategy = "median"
            self.scaler_type = "StandardScaler"
            
        def build_preprocessor(self, X):
            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = X.select_dtypes(include=['object', 'category']).columns
            
            numeric_transformer = make_pipeline(
                SimpleImputer(strategy=self.impute_strategy),
                globals()[self.scaler_type]()
            )
            
            categorical_transformer = make_pipeline(
                SimpleImputer(strategy='most_frequent'),
                OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            )
            
            return ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ],
                remainder='passthrough'
            )
            
        def auto_preprocess(self, df):
            """Automatically handle common data issues"""
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        pass
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df
            
        def set_impute_strategy(self, strategy):
            """Set imputation strategy"""
            self.impute_strategy = strategy
            
        def set_scaler_type(self, scaler_type):
            """Set scaler type"""
            self.scaler_type = scaler_type
            
    class ModelWarehouse:
        """Centralized model management with versioning"""
        def __init__(self):
            self.models = {}
            self.history = {}
            self.metrics = {}
            
        def save_model(self, name, model, metadata=None):
            """Save model with version control"""
            version = 1
            while f"{name}_v{version}" in self.models:
                version += 1
                
            model_key = f"{name}_v{version}"
            self.models[model_key] = model
            
            self.metrics[model_key] = {
                "created": datetime.now().isoformat(),
                "type": type(model).__name__,
                "metrics": metadata.get("metrics", {}) if metadata else {},
                "features": metadata.get("features", []) if metadata else [],
                "version": version
            }
            
            self.history[model_key] = {"saved": datetime.now().isoformat()}
            return model_key
            
        def load_model(self, name):
            """Load model by name"""
            return self.models.get(name)
            
        def compare_models(self):
            """Generate comparison report for all models"""
            report = []
            for name, metrics in self.metrics.items():
                report.append({
                    "name": name,
                    "type": metrics["type"],
                    "version": metrics["version"],
                    "metrics": metrics["metrics"]
                })
            return pd.DataFrame(report)
            
    class SystemMonitor:
        """Real-time system resource monitoring"""
        def __init__(self, app):
            self.app = app
            self.running = False
            
        def start(self):
            self.running = True
            threading.Thread(target=self.monitor_loop, daemon=True).start()
            
        def stop(self):
            self.running = False
            
        def monitor_loop(self):
            while self.running:
                cpu = psutil.cpu_percent()
                mem = psutil.virtual_memory().used / (1024 ** 2)
                disk = psutil.disk_usage("/").used / (1024 ** 2)
                self.app.job_queue.put((
                    "system_update",
                    {"cpu": cpu, "memory": mem, "disk": disk}
                ))
                time.sleep(2)

# ==================== APPLICATION ENTRY POINT ====================
if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    app = EnterpriseVisualSQL()
    try:
        app.mainloop()
    except KeyboardInterrupt:
        app.on_close()