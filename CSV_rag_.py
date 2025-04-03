import gradio as gr
import pandas as pd
import os
import matplotlib.pyplot as plt
import io
from PIL import Image
import base64
import re
import numpy as np
from llama_index.llms.groq import Groq
from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
)
from llama_index.experimental.query_engine.pandas import (
    PandasInstructionParser,
)
from llama_index.core import PromptTemplate

# Example datasets
EXAMPLE_DATASETS = {
    "Hotel Bookings": "hotel_bookings.csv",
}

def load_dataframe(file_path):
    try:
        if isinstance(file_path, str):
            # If it's a URL or file path
            df = pd.read_csv(file_path)
        else:
            # If it's an uploaded file
            df = pd.read_csv(file_path.name)
        return df, f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns."
    except Exception as e:
        return None, f"Error loading dataset: {str(e)}"

def create_query_pipeline(df, api_key, model="llama-3.3-70b-versatile"):
    # Create Groq LLM with the provided API key
    try:
        llm = Groq(model=model, api_key=api_key)
    except Exception as e:
        return None, f"Error initializing Groq LLM: {str(e)}"
    
    instruction_str = (
        "1. Convert the query to executable Python code using Pandas.\n"
        "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
        "3. The code should represent a solution to the query.\n"
        "4. PRINT ONLY THE EXPRESSION.\n"
        "5. Do not quote the expression.\n"
    )

    pandas_prompt_str = (
        "You are working with a pandas dataframe in Python.\n"
        "The name of the dataframe is `df`.\n"
        "This is the result of `print(df.head())`:\n"
        "{df_str}\n\n"
        "Follow these instructions:\n"
        "{instruction_str}\n"
        "Query: {query_str}\n\n"
        "Expression:"
    )
    
    response_synthesis_prompt_str = (
        "Given an input question, synthesize a response from the query results.\n"
        "Query: {query_str}\n\n"
        "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
        "Pandas Output: {pandas_output}\n\n"
        "Response: "
    )

    pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
        instruction_str=instruction_str, df_str=df.head(5)
    )
    pandas_output_parser = PandasInstructionParser(df)
    response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)

    qp = QP(
        modules={
            "input": InputComponent(),
            "pandas_prompt": pandas_prompt,
            "llm1": llm,
            "pandas_output_parser": pandas_output_parser,
            "response_synthesis_prompt": response_synthesis_prompt,
            "llm2": llm,
        },
        verbose=True,
    )
    qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
    qp.add_links(
        [
            Link("input", "response_synthesis_prompt", dest_key="query_str"),
            Link(
                "llm1", "response_synthesis_prompt", dest_key="pandas_instructions"
            ),
            Link(
                "pandas_output_parser",
                "response_synthesis_prompt",
                dest_key="pandas_output",
            ),
        ]
    )
    qp.add_link("response_synthesis_prompt", "llm2")
    
    return qp, "Query pipeline created successfully!"

def enhance_visualization(df, query):
    """
    Create an enhanced visualization based on the dataframe and query
    This function attempts to create a better visualization with proper labels and formatting
    """
    try:
        # Close any existing figures to avoid conflicts
        plt.close('all')
        
        # Create a new figure with larger size for better quality
        plt.figure(figsize=(12, 8), dpi=100)
        
        # Time-related visualization handling (for bookings over time, trends, etc.)
        if any(term in query.lower() for term in ['trend', 'time', 'year', 'month', 'booking', 'reservation']):
            # Try to detect date columns
            date_cols = [col for col in df.columns if any(term in col.lower() for term in 
                        ['date', 'year', 'month', 'time', 'arrival', 'reservation'])]
            
            if 'arrival_date_year' in df.columns and 'arrival_date_month' in df.columns:
                try:
                    # Create a year-month based visualization
                    # Convert month names to numbers for sorting
                    month_order = {
                        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
                    }
                    
                    # Count bookings by year and month
                    booking_counts = df.groupby(['arrival_date_year', 'arrival_date_month']).size().reset_index(name='count')
                    
                    # Add month order for sorting
                    booking_counts['month_order'] = booking_counts['arrival_date_month'].map(month_order)
                    booking_counts = booking_counts.sort_values(['arrival_date_year', 'month_order'])
                    
                    # Create pivot table for visualization
                    pivot_data = booking_counts.pivot(index='arrival_date_year', columns='arrival_date_month', values='count')
                    
                    # Reorder columns by month
                    months = sorted(booking_counts['arrival_date_month'].unique(), key=lambda x: month_order.get(x, 13))
                    
                    if len(months) > 0:  # Check if the months list is not empty
                        pivot_data = pivot_data[months]
                        
                        # Plot the data
                        ax = pivot_data.plot(kind='bar', figsize=(14, 8), width=0.8)
                        
                        # Enhance the plot
                        plt.title('Bookings by Month and Year', fontsize=16)
                        plt.xlabel('Year', fontsize=14)
                        plt.ylabel('Number of Bookings', fontsize=14)
                        plt.legend(title='Month', fontsize=12)
                        plt.grid(axis='y', linestyle='--', alpha=0.7)
                        plt.tight_layout()
                        
                        # Add value labels on top of bars
                        for container in ax.containers:
                            ax.bar_label(container, fontsize=9, fmt='%d')
                    else:
                        return None  # No months data found
                except Exception as e:
                    print(f"Error in time visualization: {str(e)}")
                    return None
                    
            elif len(date_cols) > 0 and any(col in df.columns for col in date_cols):
                try:
                    # Handle other time-based visualizations
                    date_col = [col for col in date_cols if col in df.columns][0]
                    df_count = df.groupby(date_col).size().reset_index(name='count')
                    
                    plt.bar(df_count[date_col], df_count['count'], color='steelblue')
                    plt.title(f'Distribution by {date_col}', fontsize=16)
                    plt.xlabel(date_col, fontsize=14)
                    plt.ylabel('Count', fontsize=14)
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                except Exception as e:
                    print(f"Error in date column visualization: {str(e)}")
                    return None
                
            else:
                # Default time visualization if we can't find specific columns
                return None  # Let matplotlib handle it
        
        # Distribution visualization (for questions about distributions)
        elif any(term in query.lower() for term in ['distribution', 'histogram', 'spread']):
            try:
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if len(numeric_cols) > 0:
                    # Choose a relevant column based on query or the first numeric column
                    target_col = None
                    for col in numeric_cols:
                        if col.lower() in query.lower():
                            target_col = col
                            break
                    
                    if target_col is None and numeric_cols:
                        target_col = numeric_cols[0]
                    
                    if target_col:
                        # Create histogram
                        plt.hist(df[target_col].dropna(), bins=30, color='steelblue', edgecolor='black', alpha=0.7)
                        plt.title(f'Distribution of {target_col}', fontsize=16)
                        plt.xlabel(target_col, fontsize=14)
                        plt.ylabel('Frequency', fontsize=14)
                        plt.grid(axis='y', linestyle='--', alpha=0.7)
                        plt.tight_layout()
                    else:
                        return None  # Let matplotlib handle it
                else:
                    return None  # Let matplotlib handle it
            except Exception as e:
                print(f"Error in distribution visualization: {str(e)}")
                return None
        
        # Comparison visualization (for questions comparing categories)
        elif any(term in query.lower() for term in ['compare', 'comparison', 'versus', 'vs', 'most', 'least']):
            try:
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                if len(categorical_cols) > 0:
                    # Choose a relevant column based on query or the first categorical column
                    target_col = None
                    for col in categorical_cols:
                        if col.lower() in query.lower():
                            target_col = col
                            break
                    
                    if target_col is None and categorical_cols:
                        target_col = categorical_cols[0]
                    
                    if target_col:
                        # Get top categories by count
                        top_categories = df[target_col].value_counts().nlargest(10)
                        
                        # Create bar chart
                        plt.bar(top_categories.index, top_categories.values, color='steelblue')
                        plt.title(f'Top Categories by {target_col}', fontsize=16)
                        plt.xlabel(target_col, fontsize=14)
                        plt.ylabel('Count', fontsize=14)
                        plt.grid(axis='y', linestyle='--', alpha=0.7)
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                    else:
                        return None  # Let matplotlib handle it
                else:
                    return None  # Let matplotlib handle it
            except Exception as e:
                print(f"Error in comparison visualization: {str(e)}")
                return None
        else:
            # For other types of queries, let the default matplotlib handle it
            return None
        
        # Save figure to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Create an image from the buffer
        img = Image.open(buf)
        plt.close('all')  # Close the figure to free memory
        
        return img
    except Exception as e:
        print(f"Error in enhance_visualization: {str(e)}")
        plt.close('all')  # Make sure to close any figures in case of error
        return None

def process_query(query, api_key, df, model_choice):
    if df is None:
        return "Please load a dataset first.", None
    
    if not api_key:
        return "Please provide your Groq API key.", None
    
    try:
        # First, try to create an enhanced visualization based on the query
        enhanced_img = enhance_visualization(df, query)
        
        # Create and run the query pipeline
        pipeline, message = create_query_pipeline(df, api_key, model_choice)
        if pipeline is None:
            return message, None
        
        # Run the query
        response = pipeline.run(query_str=query)
        
        # If we already have an enhanced visualization, use it
        if enhanced_img is not None:
            return response.message.content, enhanced_img
        
        # Otherwise check if any matplotlib figures were created by the query
        figures = plt.get_fignums()
        
        if figures:
            try:
                # Improve any existing figure if possible
                fig = plt.figure(figures[0])
                axes = fig.axes
                
                if axes and len(axes) > 0:  # Make sure axes list isn't empty
                    ax = axes[0]
                    # Add grid lines
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    # Enhance title and labels if they exist
                    if ax.get_title():
                        ax.set_title(ax.get_title(), fontsize=16)
                    if ax.get_xlabel():
                        ax.set_xlabel(ax.get_xlabel(), fontsize=14)
                    if ax.get_ylabel():
                        ax.set_ylabel(ax.get_ylabel(), fontsize=14)
                    # Handle legend if it exists
                    if ax.get_legend():
                        ax.legend(fontsize=12)
                    fig.tight_layout()
                
                # Save the figure to a bytes buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                
                # Create an image from the buffer
                img = Image.open(buf)
                plt.close('all')  # Close the figure to free memory
                
                return response.message.content, img
            except Exception as e:
                plt.close('all')
                # Log the error but continue without crashing
                print(f"Visualization error: {str(e)}")
                return response.message.content, None
        else:
            # No visualization was generated
            return response.message.content, None
            
    except Exception as e:
        plt.close('all')  # Make sure to close any figures in case of error
        return f"Error processing query: {str(e)}", None

def handle_example_selection(example_name):
    if example_name in EXAMPLE_DATASETS:
        file_path = EXAMPLE_DATASETS[example_name]
        df, message = load_dataframe(file_path)
        return df, message, gr.update(value=f"Dataset preview:\n{df.head().to_string()}")
    return None, "Please select a valid example dataset.", gr.update(value="")

def handle_file_upload(file):
    if file is not None:
        df, message = load_dataframe(file)
        return df, message, gr.update(value=f"Dataset preview:\n{df.head().to_string()}")
    return None, "No file uploaded.", gr.update(value="")

# Create Gradio interface
with gr.Blocks(title="Pandas Data Analysis with Groq LLM") as app:
    gr.Markdown("# Pandas Data Analysis with Groq LLM")
    gr.Markdown("Upload your CSV data or choose an example dataset, then ask questions about it.")
    
    # State variables
    df_state = gr.State(value=None)
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### Data Selection")
                with gr.Tab("Upload Data"):
                    file_input = gr.File(label="Upload CSV File", file_types=[".csv"])
                    upload_button = gr.Button("Load Uploaded Data")
                
                with gr.Tab("Example Datasets"):
                    example_dropdown = gr.Dropdown(
                        choices=list(EXAMPLE_DATASETS.keys()),
                        label="Select Example Dataset"
                    )
                    example_button = gr.Button("Load Example Dataset")
                
                data_status = gr.Textbox(label="Data Loading Status", interactive=False)
            
            with gr.Group():
                gr.Markdown("### Groq API Configuration")
                api_key = gr.Textbox(
                    label="Enter your Groq API Key", 
                    placeholder="gsk_...",
                    type="password"
                )
                model_choice = gr.Dropdown(
                    choices=["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma-7b-it"],
                    value="llama-3.3-70b-versatile",
                    label="Select Groq Model"
                )
        
        with gr.Column(scale=1):
            data_preview = gr.Textbox(label="Dataset Preview", interactive=False, lines=10)
            query_input = gr.Textbox(
                label="Ask a question about your data",
                placeholder="e.g., What is the trend of monthly bookings over time?",
                lines=2
            )
            query_button = gr.Button("Submit Query")
            
            # Output display with tabs for text and visualization
            with gr.Tabs():
                with gr.TabItem("Text Response"):
                    response_output = gr.Textbox(label="Response", interactive=False, lines=10)
                with gr.TabItem("Visualization"):
                    image_output = gr.Image(label="Data Visualization", interactive=False)
    
    # Handle events
    upload_button.click(
        handle_file_upload,
        inputs=[file_input],
        outputs=[df_state, data_status, data_preview]
    )
    
    example_button.click(
        handle_example_selection,
        inputs=[example_dropdown],
        outputs=[df_state, data_status, data_preview]
    )
    
    query_button.click(
        process_query,
        inputs=[query_input, api_key, df_state, model_choice],
        outputs=[response_output, image_output]
    )
    
    gr.Markdown("""
    ### Instructions
    1. Upload your CSV file or select an example dataset
    2. Enter your Groq API key (get one at [https://console.groq.com](https://console.groq.com))
    3. Ask questions about your data in natural language
    4. Get AI-powered insights and visualizations based on your data
    
    ### Example Questions
    - What is the trend of monthly bookings over time?
    - What's the distribution of stay duration?
    - Which country has the most bookings?
    - Is there a correlation between lead time and cancellations?
    - Show me bookings by month and year
    """)

# Launch the app
if __name__ == "__main__":
    app.launch()