import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import sys
from io import StringIO
from openai import OpenAI

# Page configuration
st.set_page_config(page_title="Student Data Chat", layout="wide")

# Set up OpenAI API
def initialize_openai():
    api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
    if api_key:
        return api_key
    else:
        st.sidebar.warning("Please enter your OpenAI API key to enable chat functionality.")
        return None

# Function to load CSV data
@st.cache_data
def load_data(file_path=None):
    if file_path:
        return pd.read_csv(file_path)
    else:
        # Default to using the student_habits_performance.csv file
        return pd.read_csv("student_habits_performance.csv")

# Function to get relevant data context based on query
def get_query_context(df, query):
    # Prepare initial context with dataframe info
    context = f"""
    DataFrame Information:
    - {len(df)} students
    - Columns: {', '.join(df.columns.tolist())}
    """
    
    # Analyze query to determine what data to include
    query_lower = query.lower()
    
    # Add basic statistics for relevant columns
    mentioned_columns = []
    
    # Check for column mentions
    for col in df.columns:
        col_variants = [col, col.replace('_', ' ')]
        if any(variant in query_lower for variant in col_variants):
            mentioned_columns.append(col)
    
    # If no specific columns are mentioned, include overall statistics
    if not mentioned_columns:
        # Include overall dataframe statistics
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        context += "\nOverall Statistics:\n"
        for col in numeric_cols:
            context += f"- Average {col}: {df[col].mean():.2f}\n"
    else:
        # Add statistics for mentioned columns
        context += "\nRelevant Column Statistics:\n"
        
        for col in mentioned_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                context += f"""
                {col}:
                - Mean: {df[col].mean():.2f}
                - Median: {df[col].median():.2f}
                - Min: {df[col].min():.2f}
                - Max: {df[col].max():.2f}
                - Std Dev: {df[col].std():.2f}
                """
            else:
                context += f"""
                {col}:
                - Value counts: {df[col].value_counts().to_dict()}
                """
    
    # Check for specific analysis requests
    if any(term in query_lower for term in ["average", "mean", "avg"]):
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            if col in query_lower or col.replace('_', ' ') in query_lower:
                context += f"\nAverage {col}: {df[col].mean():.2f}\n"
    
    if any(term in query_lower for term in ["correlation", "correlate", "relationship"]):
        if len(mentioned_columns) >= 2 and all(pd.api.types.is_numeric_dtype(df[col]) for col in mentioned_columns):
            # If specific columns are mentioned, show their correlation
            corr_value = df[mentioned_columns].corr().iloc[0, 1]
            context += f"\nCorrelation between {mentioned_columns[0]} and {mentioned_columns[1]}: {corr_value:.3f}\n"
        else:
            # Otherwise show correlations with exam_score or other important metrics
            important_cols = ['exam_score', 'attendance_percentage', 'mental_health_rating']
            for col in important_cols:
                if col in df.columns:
                    correlations = df.corr()[col].sort_values(ascending=False).drop(col)
                    context += f"\nTop correlations with {col}:\n"
                    for corr_col, corr_val in correlations.head(5).items():
                        context += f"- {corr_col}: {corr_val:.3f}\n"
                    break
    
    if any(term in query_lower for term in ["group", "grouped by", "compare"]):
        # Try to identify grouping column
        group_cols = ['gender', 'part_time_job', 'diet_quality', 'parental_education_level', 
                      'internet_quality', 'extracurricular_participation']
        
        target_cols = ['exam_score', 'mental_health_rating', 'attendance_percentage', 'study_hours_per_day']
        
        for group_col in group_cols:
            if group_col in query_lower or group_col.replace('_', ' ') in query_lower:
                context += f"\nGrouped statistics by {group_col}:\n"
                
                for target_col in target_cols:
                    if target_col in query_lower or target_col.replace('_', ' ') in query_lower:
                        grouped_stats = df.groupby(group_col)[target_col].agg(['mean', 'median', 'count'])
                        context += f"\n{target_col} by {group_col}:\n"
                        context += str(grouped_stats) + "\n"
                        break
                
                # If no specific target column found, use exam_score as default
                if not any(target_col in query_lower or target_col.replace('_', ' ') in query_lower for target_col in target_cols):
                    grouped_stats = df.groupby(group_col)['exam_score'].agg(['mean', 'median', 'count'])
                    context += f"\nexam_score by {group_col}:\n"
                    context += str(grouped_stats) + "\n"
    
    return context

# RAG-based chat with data
def chat_with_data(client, df, user_query, chat_history):
    # Get relevant data context based on the query
    data_context = get_query_context(df, user_query)
    
    # Prepare system message with instructions and data context
    system_message = f"""You are a data analyst assistant specialized in analyzing student data.
    You're working with a student habits and performance dataset.
    
    Here's relevant data based on the user's query:
    {data_context}
    
    When answering questions:
    1. Focus on the data provided in the context
    2. Be specific with numbers and statistics
    3. Provide clear insights and interpretations
    4. If the user asks for code, provide the exact pandas code that would answer the question
    5. When suggesting visualizations, include complete code that will execute properly and create the plot
    6. All code should be in Python using pandas, matplotlib, or seaborn
    """
    
    # Prepare the messages for the API call
    if not chat_history:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_query}
        ]
    else:
        messages = [{"role": "system", "content": system_message}]
        
        # Add chat history
        for message in chat_history:
            messages.append({"role": message["role"], "content": message["content"]})
        
        # Add the new user query
        messages.append({"role": "user", "content": user_query})
    
    # Get response from OpenAI
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting response: {e}"

# Function to execute code from response
def execute_code_from_response(df, response_text):
    # Extract code blocks from the response
    code_blocks = []
    lines = response_text.split('\n')
    i = 0
    
    while i < len(lines):
        if re.match(r'^```(?:python)?$', lines[i].strip()):
            # Start of a code block
            code_block = []
            i += 1
            
            # Collect lines until the end of the code block
            while i < len(lines) and not re.match(r'^```$', lines[i].strip()):
                code_block.append(lines[i])
                i += 1
                
            code_blocks.append('\n'.join(code_block))
        
        i += 1
    
    # Execute code blocks and collect results
    results = []
    for code in code_blocks:
        if not code.strip():
            continue
            
        try:
            # Create a local namespace with the dataframe
            local_namespace = {
                'df': df,
                'pd': pd,
                'np': np,
                'plt': plt,
                'sns': sns
            }
            
            # Redirect stdout to capture print statements
            stdout_buffer = StringIO()
            original_stdout = sys.stdout
            sys.stdout = stdout_buffer
            
            # Execute the code
            exec(code, local_namespace)
            
            # Restore stdout
            sys.stdout = original_stdout
            stdout_output = stdout_buffer.getvalue()
            
            # Check if there were any printed outputs
            if stdout_output.strip():
                results.append(("text", stdout_output))
            
            # Check if a plot was generated
            if plt.get_fignums():
                fig = plt.gcf()
                results.append(("figure", fig))
                plt.close()
                
        except Exception as e:
            results.append(("error", f"Error executing code: {str(e)}"))
    
    return results

# Main app
def main():
    st.title("Student Data Analysis Chat")
    st.markdown("Ask questions about student habits and performance data using natural language")
    
    # Sidebar for settings
    st.sidebar.title("Settings")
    api_key = initialize_openai()
    
    # Load data
    st.sidebar.header("Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file (optional)", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Custom data loaded successfully!")
    else:
        st.sidebar.info("Using default student_habits_performance.csv")
        try:
            df = load_data()
        except FileNotFoundError:
            st.error("Error: student_habits_performance.csv not found. Please upload a CSV file.")
            df = None
    
    # Display basic data info
    if df is not None:
        with st.expander("Preview Data"):
            st.dataframe(df.head())
            
            st.subheader("Basic Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of Students", f"{len(df)}")
            with col2:
                st.metric("Average Exam Score", f"{df['exam_score'].mean():.1f}")
            with col3:
                st.metric("Average Study Hours", f"{df['study_hours_per_day'].mean():.1f}")
    
        # Initialize OpenAI client if API key is provided
        if api_key:
            client = OpenAI(api_key=api_key)
            
            # Chat interface
            st.header("Ask about your data")
            st.markdown("""
            **Example questions:**
            - What's the average exam score?
            - Is there a correlation between study hours and exam scores?
            - How does sleep affect mental health ratings?
            - Show me exam scores grouped by gender
            - What percentage of students have a part-time job?
            - Visualize the relationship between social media usage and exam scores
            """)
            
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "results" in message:
                        for result_type, result in message.get("results", []):
                            if result_type == "text":
                                st.code(result)
                            elif result_type == "figure":
                                st.pyplot(result)
                            elif result_type == "error":
                                st.error(result)
            
            # Input for new question
            if prompt := st.chat_input("Ask something about the student data..."):
                # Add user message to chat
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Display assistant response
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    message_placeholder.text("Analyzing data...")
                    
                    # Get chat response
                    chat_history = [{"role": msg["role"], "content": msg["content"]} 
                                    for msg in st.session_state.messages[:-1]]  # Exclude the latest user message
                    
                    response_text = chat_with_data(client, df, prompt, chat_history)
                    message_placeholder.markdown(response_text)
                    
                    # Execute any code in the response
                    results = execute_code_from_response(df, response_text)
                    
                    # Display results (tables, charts, etc.)
                    for result_type, result in results:
                        if result_type == "text":
                            st.code(result)
                        elif result_type == "figure":
                            st.pyplot(result)
                        elif result_type == "error":
                            st.error(result)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "results": results
                })
        else:
            st.warning("Please enter your OpenAI API key in the sidebar to enable chat functionality.")

if __name__ == "__main__":
    main()