import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

# --- Page Configuration ---
st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="📊",
    layout="wide"
)

# --- Helper Functions ---

def clean_sheet_data(df, sheet_name):
    """
    Cleans a single sheet based on the project structure.
    Handles 'Weightage' and 'Total' meta-rows and standardizes column names.
    """
    df = df.copy()
    
    # Remove metadata rows (Weightage/Total) based on Sr.# column
    if 'Sr.#' in df.columns:
        df = df[pd.to_numeric(df['Sr.#'], errors='coerce').notnull()]
    
    # Standardize Column Names
    new_cols = {}
    for col in df.columns:
        col_str = str(col).strip()
        if 'Qz' in col_str or 'Q' in col_str and 'Tot' not in col_str:
            new_cols[col] = col_str
        elif 'As' in col_str or 'A' in col_str and 'Tot' not in col_str:
            new_cols[col] = col_str
        elif 'S-I' in col_str and 'S-II' not in col_str:
            new_cols[col] = 'Midterm1'
        elif 'S-II' in col_str:
            new_cols[col] = 'Midterm2'
        elif 'Final' in col_str:
            new_cols[col] = 'Final_Score'
        elif 'Proj' in col_str:
            new_cols[col] = 'Project'
            
    # Calculate Averages (Feature Engineering)
    # We treat NaNs as 0 for calculation
    temp_df = df.copy()
    for col in temp_df.columns:
        temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce').fillna(0)
    
    quiz_cols = [c for c in df.columns if 'Qz' in str(c) or 'Q' in str(c)]
    assign_cols = [c for c in df.columns if 'As' in str(c) or 'A' in str(c)]
    
    # Avoid div by zero if no columns found
    if quiz_cols:
        df['Avg_Quiz'] = temp_df[quiz_cols].mean(axis=1)
    else:
        df['Avg_Quiz'] = 0
        
    if assign_cols:
        df['Avg_Assignment'] = temp_df[assign_cols].mean(axis=1)
    else:
        df['Avg_Assignment'] = 0
        
    # Extract Exams (Logic to find the summary columns)
    # Midterm 1
    if 'S-I' in df.columns: df['Midterm1'] = temp_df['S-I']
    elif 'S-I ' in df.columns: df['Midterm1'] = temp_df['S-I ']
    else:
        # heuristic: take the last column containing S-I
        s1 = [c for c in df.columns if 'S-I' in str(c)]
        df['Midterm1'] = temp_df[s1[-1]] if s1 else 0

    # Midterm 2
    if 'S-II' in df.columns: df['Midterm2'] = temp_df['S-II']
    else:
        s2 = [c for c in df.columns if 'S-II' in str(c)]
        df['Midterm2'] = temp_df[s2[-1]] if s2 else 0

    # Final
    if 'Final' in df.columns: df['Final_Score'] = temp_df['Final']
    else:
        fn = [c for c in df.columns if 'Final' in str(c)]
        df['Final_Score'] = temp_df[fn[-1]] if fn else 0
        
    # Project (Optional, not in all sheets)
    proj_cols = [c for c in df.columns if 'Proj' in str(c)]
    if proj_cols:
        df['Project'] = temp_df[proj_cols[-1]]
    else:
        df['Project'] = 0
        
    # Return clean subset
    cols_to_keep = ['Avg_Quiz', 'Avg_Assignment', 'Midterm1', 'Midterm2', 'Project', 'Final_Score']
    final_df = df[cols_to_keep].copy()
    
    # Ensure numeric
    for c in cols_to_keep:
        final_df[c] = pd.to_numeric(final_df[c], errors='coerce').fillna(0)
        
    final_df['Sheet'] = sheet_name
    return final_df

def run_bootstrap_ci(model, X_train, y_train, n_iterations=500):
    """Calculates 95% Confidence Interval for MAE using Bootstrapping"""
    stats = []
    for _ in range(n_iterations):
        # Prepare bootstrap sample
        X_sample, y_sample = resample(X_train, y_train)
        model.fit(X_sample, y_sample)
        pred = model.predict(X_sample) 
        score = mean_absolute_error(y_sample, pred)
        stats.append(score)
    
    alpha = 0.95
    p = ((1.0 - alpha) / 2.0) * 100
    lower = np.percentile(stats, p)
    upper = np.percentile(stats, 100 - p)
    return lower, upper

# --- Main App Logic ---

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["1. Pipeline & Workflow", "2. Data Exploration (EDA)", "3. Model Comparison & RQ Analysis", "4. Prediction System"])

st.sidebar.markdown("---")
st.sidebar.subheader("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload 'marks_dataset.xlsx'", type=['xlsx'])

if uploaded_file:
    # Load Data Once
    xls = pd.ExcelFile(uploaded_file)
    all_data = []
    for sheet in xls.sheet_names:
        raw_df = pd.read_excel(xls, sheet_name=sheet)
        all_data.append(clean_sheet_data(raw_df, sheet))
    main_df = pd.concat(all_data, ignore_index=True)
    
    # Filter valid rows (simple cleanup)
    main_df = main_df[main_df['Final_Score'] > 0] # Remove likely dropouts for better viz
    
    if page == "1. Pipeline & Workflow":
        st.title("Project Workflow Pipeline")
        st.markdown("As requested in the project deliverables, below is the visual representation of the data science workflow used in this application.")
        
        # Graphviz Diagram
        st.graphviz_chart('''
            digraph {
                rankdir=LR;
                node [shape=box, style=filled, fillcolor=lightblue];
                
                Data [label="Raw Excel Sheets\n(Multiple Years)", shape=folder, fillcolor=gold];
                Clean [label="Data Cleaning\n(Remove Meta-rows,\nStandardize Cols)"];
                Feat [label="Feature Engineering\n(Avg Quiz, Avg Assign)"];
                Split [label="Data Split\n(Train / Test)"];
                
                subgraph cluster_models {
                    label = "Model Training (RQs)";
                    style=dashed;
                    LR [label="Linear Regression"];
                    Poly [label="Polynomial Reg (Deg 2)"];
                    Dummy [label="Dummy Regressor\n(Baseline)"];
                }
                
                Eval [label="Evaluation\n(MAE, RMSE, R², CI)"];
                App [label="Streamlit Dashboard\n(Interactive Prediction)", fillcolor=lightgreen];

                Data -> Clean -> Feat -> Split;
                Split -> LR;
                Split -> Poly;
                Split -> Dummy;
                LR -> Eval;
                Poly -> Eval;
                Dummy -> Eval;
                Eval -> App;
            }
        ''')
        
        st.info("**Note on Data Leakage prevention:** In the Prediction System (Page 4), features are restricted based on the Exam context (e.g., Midterm 2 is NOT used to predict Midterm 1).")

    elif page == "2. Data Exploration (EDA)":
        st.title("Exploratory Data Analysis")
        
        # --- Descriptive Statistics ---
        st.subheader("1. Descriptive Statistics")
        st.markdown("Summary statistics of the aggregated data (mean, std, min, max, quartiles).")
        st.dataframe(main_df.describe().style.format("{:.2f}"))
        
        st.markdown("---")
        
        # --- Visualizations ---
        st.subheader("2. Detailed Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Correlation Matrix**")
            fig, ax = plt.subplots(figsize=(6, 5))
            corr = main_df[['Avg_Quiz', 'Avg_Assignment', 'Midterm1', 'Midterm2', 'Project', 'Final_Score']].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig)
            st.caption("Shows how strongly different exams are related. (1.0 = perfect positive correlation)")

        with col2:
            st.markdown("**Distribution of Final Scores**")
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            sns.histplot(main_df['Final_Score'], kde=True, color='purple', ax=ax2)
            ax2.set_title("Frequency of Final Scores")
            st.pyplot(fig2)
            st.caption("Distribution of student grades in the final exam.")

        # --- Row 2: Relationships and Outliers ---
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**Outlier Detection (Box Plot)**")
            viz_feat = st.selectbox("Select Feature for Box Plot", ['Midterm1', 'Midterm2', 'Final_Score', 'Avg_Assignment'])
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            sns.boxplot(x=main_df[viz_feat], color='orange', ax=ax3)
            ax3.set_title(f"Box Plot of {viz_feat}")
            st.pyplot(fig3)
            st.caption("Dots outside the whiskers represent potential outliers.")

        with col4:
            st.markdown("**Feature vs. Target Relationship**")
            x_axis = st.selectbox("Select X-Axis", ['Midterm1', 'Midterm2', 'Avg_Quiz'])
            y_axis = st.selectbox("Select Y-Axis", ['Final_Score', 'Midterm2'])
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x=main_df[x_axis], y=main_df[y_axis], hue=main_df['Sheet'], palette='viridis', ax=ax4)
            ax4.set_title(f"{x_axis} vs {y_axis}")
            st.pyplot(fig4)
            st.caption("Scatter plot showing relationships between assessment stages.")

    elif page == "3. Model Comparison & RQ Analysis":
        st.title("Model Analysis & Evaluation")
        st.markdown("Train and evaluate models for specific research questions (RQ).")

        # 1. Configuration
        rq_mode = st.selectbox("Select Research Question:", [
            "RQ1: Predict Midterm 1", 
            "RQ2: Predict Midterm 2", 
            "RQ3: Predict Final Exam"
        ])
        
        # Define Features (Preventing Data Leakage)
        if "RQ1" in rq_mode:
            features = ['Avg_Quiz', 'Avg_Assignment']
            target = 'Midterm1'
        elif "RQ2" in rq_mode:
            features = ['Avg_Quiz', 'Avg_Assignment', 'Midterm1']
            target = 'Midterm2'
        else: # RQ3
            features = ['Avg_Quiz', 'Avg_Assignment', 'Midterm1', 'Midterm2', 'Project']
            target = 'Final_Score'
            
        st.info(f"**Target:** `{target}` | **Features:** `{features}`")
        
        # 2. Training Button
        if st.button("Train Models"):
            X = main_df[features]
            y = main_df[target]
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Define Models
            models = {
                "Linear Regression": LinearRegression(),
                "Polynomial (Deg 2)": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
                "Dummy (Baseline)": DummyRegressor(strategy="mean")
            }
            
            # Train and Store Results in Session State
            results_list = []
            trained_models = {}
            predictions = {}
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                train_r2 = r2_score(y_train, model.predict(X_train))
                
                results_list.append({
                    "Model": name,
                    "MAE": mae,
                    "RMSE": rmse,
                    "Test R²": r2,
                    "Train R²": train_r2
                })
                
                trained_models[name] = model
                predictions[name] = y_pred
            
            # Store in session state for persistence
            st.session_state['results_df'] = pd.DataFrame(results_list).set_index("Model")
            st.session_state['y_test'] = y_test
            st.session_state['predictions'] = predictions
            st.session_state['rq_mode'] = rq_mode
            st.session_state['X_train'] = X_train # For bootstrap
            st.session_state['y_train'] = y_train

        # 3. Display Results (if available)
        if 'results_df' in st.session_state:
            st.markdown("### Performance Metrics")
            st.table(st.session_state['results_df'].style.format("{:.4f}"))
            
            # Overfitting Warning
            poly_row = st.session_state['results_df'].loc["Polynomial (Deg 2)"]
            if (poly_row['Train R²'] - poly_row['Test R²']) > 0.15:
                st.warning(f"⚠️ **Overfitting Detected:** Polynomial Regression Train R² ({poly_row['Train R²']:.2f}) >> Test R² ({poly_row['Test R²']:.2f}).")

            st.markdown("---")
            
            # 4. Actual vs Predicted Graph
            st.subheader("Actual vs Predicted Visualization")
            
            col_viz, col_opts = st.columns([3, 1])
            
            with col_opts:
                selected_viz_model = st.radio(
                    "Select Model to Visualize:", 
                    ["Linear Regression", "Polynomial (Deg 2)", "Dummy (Baseline)"]
                )
            
            with col_viz:
                y_test_viz = st.session_state['y_test']
                y_pred_viz = st.session_state['predictions'][selected_viz_model]
                
                fig_viz, ax_viz = plt.subplots(figsize=(8, 6))
                
                # Scatter plot of data
                ax_viz.scatter(y_test_viz, y_pred_viz, alpha=0.6, edgecolors='b', label='Prediction Points')
                
                # Red Line (Perfect Prediction Line)
                min_val = min(y_test_viz.min(), y_pred_viz.min())
                max_val = max(y_test_viz.max(), y_pred_viz.max())
                ax_viz.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y=x)')
                
                ax_viz.set_xlabel("Actual Values")
                ax_viz.set_ylabel("Predicted Values")
                ax_viz.set_title(f"Actual vs Predicted: {selected_viz_model}")
                ax_viz.legend()
                ax_viz.grid(True, linestyle='--', alpha=0.5)
                
                st.pyplot(fig_viz)

    elif page == "4. Prediction System":
        st.title("Interactive Prediction System")
        st.markdown("Predict scores for specific exams. Inputs change based on the selected target.")
        
        pred_type = st.radio("What do you want to predict?", ["Midterm 1", "Midterm 2", "Final Exam"], horizontal=True)
        
        col1, col2, col3 = st.columns(3)
        
        # Dynamic Inputs
        with col1:
            val_quiz = st.number_input("Average Quiz Score (0-10)", 0.0, 10.0, 5.0)
            val_assign = st.number_input("Average Assignment Score (0-10)", 0.0, 10.0, 5.0)
        
        val_m1 = 0.0
        val_m2 = 0.0
        val_proj = 0.0
        
        # Conditional Rendering of Inputs
        if pred_type in ["Midterm 2", "Final Exam"]:
            with col2:
                val_m1 = st.number_input("Midterm 1 Score (0-50)", 0.0, 50.0, 20.0)
        
        if pred_type == "Final Exam":
            with col3:
                val_m2 = st.number_input("Midterm 2 Score (0-50)", 0.0, 50.0, 20.0)
                val_proj = st.number_input("Project Score (0-10)", 0.0, 10.0, 5.0)
        
        if st.button("Predict Score"):
            # Set up correct model and inputs
            if pred_type == "Midterm 1":
                features = ['Avg_Quiz', 'Avg_Assignment']
                target = 'Midterm1'
                input_data = [[val_quiz, val_assign]]
            elif pred_type == "Midterm 2":
                features = ['Avg_Quiz', 'Avg_Assignment', 'Midterm1']
                target = 'Midterm2'
                input_data = [[val_quiz, val_assign, val_m1]]
            else:
                features = ['Avg_Quiz', 'Avg_Assignment', 'Midterm1', 'Midterm2', 'Project']
                target = 'Final_Score'
                input_data = [[val_quiz, val_assign, val_m1, val_m2, val_proj]]
            
            # Train specific model on full data for prediction
            model = LinearRegression()
            model.fit(main_df[features], main_df[target])
            pred = model.predict(input_data)[0]
            
            st.metric(label=f"Predicted {pred_type}", value=f"{pred:.2f}")

else:
    st.warning("Please upload the project dataset (marks_dataset.xlsx) in the sidebar to proceed.")