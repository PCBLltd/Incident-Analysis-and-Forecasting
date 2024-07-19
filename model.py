import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import os
from datetime import datetime
from scipy.stats.mstats import winsorize

# Define allowed locations and available features
locations = ['Palej', 'Mundra', 'Kochi', 'Durgapur', 'Kolkata HO']
features = ['Total Tokens', 'Low', 'Medium (SAP) - 16 Hrs', 'No Inputs', 'Low (SAP) - 40 Hrs', 'Critical', 'Medium', 'High', 'High (SAP) - 6 Hrs', 'Very High (SAP) - 2 Hrs']
categories = ['Printing / Scanning', 'SAP', 'Information / Report Required', 'Asset Allocation', 'Software', 'Login / Access Issues', 'Creation / Addition / Modification', 'User Separation', 'Online Meeting', 'Server / Security activity', 'System Configuration', 'Network', 'Desktop / Laptop Hardware', 'No Inputs', 'Operating System Issue', 'Domain Joining', 'Asset De-allocation', 'Test Ticket', 'Preventive Checking', 'Procurement', 'User Transfer', 'User Onboarding']

# Function to preprocess data
def separate_date_time(created_time):
    try:
        datetime_obj = pd.to_datetime(created_time, format="%d-%m-%Y %H:%M")
        if isinstance(datetime_obj, (datetime, pd.Timestamp)):
            return datetime_obj.date(), datetime_obj.time()
        else:
            return None, None
    except ValueError:
        return None, None

# def preprocess_data(data):
#     data[['Creation Date', 'Creation Time']] = data['Created Time'].apply(separate_date_time).apply(pd.Series)
#     summary_df = pd.DataFrame(columns=['Site', 'Creation Date', 'Total Tokens', 'Low', 'Medium (SAP) - 16 Hrs', 'No Inputs', 'Low (SAP) - 40 Hrs', 'Critical', 'Medium', 'High', 'High (SAP) - 6 Hrs', 'Very High (SAP) - 2 Hrs'])

#     unique_dates = data['Creation Date'].unique()
#     for date in unique_dates:
#         date_df = data[data['Creation Date'] == date]
#         if not date_df.empty:
#             if 'No Inputs' in date_df.columns:
#                 date_df = date_df[date_df['No Inputs'] != 'No Input']
#             total_tokens = len(date_df)
#             priority_counts = date_df[date_df['Priority'] != 'No Input']['Priority'].value_counts().reindex(features, fill_value=0)
#             new_row = pd.DataFrame([{
#                 'Site': date_df['Site'].iloc[0],
#                 'Creation Date': date,
#                 'Total Tokens': total_tokens,
#                 'Low': priority_counts.get('Low', 0),
#                 'Medium (SAP) - 16 Hrs': priority_counts.get('Medium (SAP) - 16 Hrs', 0),
#                 'No Inputs': 0,
#                 'Low (SAP) - 40 Hrs': priority_counts.get('Low (SAP) - 40 Hrs', 0),
#                 'Critical': priority_counts.get('Critical', 0),
#                 'Medium': priority_counts.get('Medium', 0),
#                 'High': priority_counts.get('High', 0),
#                 'High (SAP) - 6 Hrs': priority_counts.get('High (SAP) - 6 Hrs', 0),
#                 'Very High (SAP) - 2 Hrs': priority_counts.get('Very High (SAP) - 2 Hrs', 0)
#             }])
#             summary_df = pd.concat([summary_df, new_row], ignore_index=True)
#     for location in summary_df['Site'].unique():
#         location_df = summary_df[summary_df['Site'] == location]
#         location_df.to_csv(f'token_summary_{location}.csv', index=False)
#     return summary_df

def preprocess_data(data):
    # Skip the first 5 rows if present
    #data = data.iloc[5:, :].reset_index(drop=True)

    data[['Creation Date', 'Creation Time']] = data['Created Time'].apply(separate_date_time).apply(pd.Series)
    summary_df = pd.DataFrame(columns=['Site', 'Creation Date', 'Total Tokens', 'Low', 'Medium (SAP) - 16 Hrs', 'No Inputs', 'Low (SAP) - 40 Hrs', 'Critical', 'Medium', 'High', 'High (SAP) - 6 Hrs', 'Very High (SAP) - 2 Hrs'])

    unique_dates = data['Creation Date'].unique()
    for date in unique_dates:
        date_df = data[data['Creation Date'] == date]
        if not date_df.empty:
            if 'No Inputs' in date_df.columns:
                date_df = date_df[date_df['No Inputs'] != 'No Input']
            total_tokens = len(date_df)
            priority_counts = date_df[date_df['Priority'] != 'No Input']['Priority'].value_counts().reindex(features, fill_value=0)
            new_row = pd.DataFrame([{
                'Site': date_df['Site'].iloc[0],
                'Creation Date': date,
                'Total Tokens': total_tokens,
                'Low': priority_counts.get('Low', 0),
                'Medium (SAP) - 16 Hrs': priority_counts.get('Medium (SAP) - 16 Hrs', 0),
                'No Inputs': 0,
                'Low (SAP) - 40 Hrs': priority_counts.get('Low (SAP) - 40 Hrs', 0),
                'Critical': priority_counts.get('Critical', 0),
                'Medium': priority_counts.get('Medium', 0),
                'High': priority_counts.get('High', 0),
                'High (SAP) - 6 Hrs': priority_counts.get('High (SAP) - 6 Hrs', 0),
                'Very High (SAP) - 2 Hrs': priority_counts.get('Very High (SAP) - 2 Hrs', 0)
            }])
            summary_df = pd.concat([summary_df, new_row], ignore_index=True)
    for location in summary_df['Site'].unique():
        location_df = summary_df[summary_df['Site'] == location]
        location_df.to_csv(f'token_summary_{location}.csv', index=False)
    return summary_df


# Function to preprocess data for new categories
def preprocess_data_new(data):
    #data = data.iloc[5:, :].reset_index(drop=True)
    data[['Creation Date', 'Creation Time']] = data['Created Time'].apply(separate_date_time).apply(pd.Series)
    summary_df = pd.DataFrame(columns=['Site', 'Creation Date'] + categories)

    unique_dates = data['Creation Date'].unique()
    unique_sites = data['Site'].unique()

    for date in unique_dates:
        for site in unique_sites:
            date_site_df = data[(data['Creation Date'] == date) & (data['Site'] == site)]
            if not date_site_df.empty:
                category_counts = date_site_df['Category'].value_counts().to_dict()
                new_row = {'Site': site, 'Creation Date': date}
                for category in categories:
                    new_row[category] = category_counts.get(category, 0)
                summary_df = pd.concat([summary_df, pd.DataFrame([new_row])], ignore_index=True)

    for location in summary_df['Site'].unique():
        location_df = summary_df[summary_df['Site'] == location]
        location_df.to_csv(f'category_token_{location}.csv', index=False)

    summary_df.to_csv('category_token.csv', index=False)
    return summary_df

# Function to fit and forecast with Prophet
def fit_and_forecast(df, periods):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)
    forecast['yhat'] = forecast['yhat'].apply(lambda x: max(0, x))
    return model, forecast

# Function to calculate RMSE
def calculate_rmse(test_data, forecast):
    rmse = np.sqrt(np.mean((test_data['y'] - forecast['yhat'].iloc[-len(test_data):].values) ** 2))
    return rmse

# Function to plot forecast with Plotly
def plot_forecast(df, test_data, forecast, future_forecast):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_data['ds'], y=test_data['Predicted'], mode='lines', name='Predicted', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'], mode='lines', name='Future Forecast', line=dict(color='green')))
    fig.update_layout(title='Model Prediction vs Reality', xaxis_title='Date', yaxis_title='Value', legend_title='Legend', hovermode='x unified')
    st.plotly_chart(fig)

# Function to plot bar chart with Plotly
def plot_bar_chart(location_data, location):
    category_counts = location_data['Category'].value_counts().reset_index()
    category_counts.columns = ['Category', 'Count']
    fig = px.bar(category_counts, x='Category', y='Count', title=f'Ticket Category Distribution for {location}')
    fig.update_layout(xaxis_title='Category', yaxis_title='Count')
    return fig

# Function to filter valid rows
def is_valid_row(row):
    return not (row['Created Time'] == 'No Inputs' or row['Resolved Time'] == 'No Inputs')

# Function to calculate Winsorized Mean
def winsorized_mean(group, limits=0.4):
    group['time_taken'] = (group['Resolved Time'] - group['Created Time']) / pd.Timedelta(hours=1)
    winsorized_data = winsorize(group['time_taken'], limits=limits)
    return winsorized_data.mean()

# Main function for Streamlit app
def main():
    st.set_page_config(layout="wide")
    st.title('Web View Forecasting')

    # Page navigation
    pages = {
        "Priority Distribution": page_upload_data,
        "Priority Forecasting": page_forecast_feature,
        "Resolution Time Analysis": page_winsorized_mean_time,
        "Category Distribution": page_upload_data_new,
        "Category Foracasting": page_forecast_feature_new
    }

    page = st.sidebar.radio("Select Page", tuple(pages.keys()))

    # Display selected page
    pages[page]()

# Page 1: Upload Data and Preprocess
def page_upload_data():
    st.header("Upload Data and Preprocess")

    sample_file_path = 'sample_Data.csv'
    if os.path.exists(sample_file_path):
        with open(sample_file_path, "rb") as f:
            st.download_button(
                label="Download Sample Dataset",
                data=f,
                file_name='sample_dataset.csv',
                mime='text/csv'
            )

    st.markdown("### Disclaimer:")
    st.markdown("Please ensure that the uploaded CSV file follows the correct format , as mentioned in the sample data provided. Any deviation may lead to errors.")
    
   

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        

        # Preprocess data
        summary_data = preprocess_data(data)
        st.write("Data Preprocessed Successfully")
        
        # Display and provide download links for each location-wise dataset
        st.subheader("Location-wise Datasets")
        for location in locations:
            file_path = f'token_summary_{location}.csv'
            if os.path.exists(file_path):
                st.write(f"Dataset for {location}:")

                # Create columns for dataset and pie chart
                col1, col2 = st.columns([2, 1])

                # Display dataset
                with col1:
                    location_data = pd.read_csv(file_path)
                    st.dataframe(location_data)

                    # Provide download link
                    with open(file_path, "rb") as f:
                        st.download_button(
                            label=f"Download {location} Dataset",
                            data=f,
                            file_name=f"token_summary_{location}.csv",
                            mime='text/csv'
                        )

                # Filter out 'No Inputs' in Category and Priority
                df_filtered = data[(data['Site'] == location) & 
                                   (data['Category'] != 'No Inputs') & 
                                   (data['Priority'] != 'No Inputs')]

                # Analyze tickets by Priority
                priority_counts = df_filtered['Priority'].value_counts()

                # Display pie chart
                with col2:
                    if not df_filtered.empty:
                        fig = px.pie(
                            priority_counts,
                            values=priority_counts.values,
                            names=priority_counts.index,
                            title=f'Priority Distribution for {location}',
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        st.plotly_chart(fig)

# Page 2: Forecast Feature
# def page_forecast_feature():
#     st.header("Forecast Feature")

#     uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

#     if uploaded_file is not None:
#         data = pd.read_csv(uploaded_file)
#         selected_feature = st.selectbox("Select Feature for Forecasting", features)

#         if selected_feature:
#             location = st.selectbox("Select Location", locations)
#             location_data = data[data['Site'] == location]
#             location_data['ds'] = pd.to_datetime(location_data['Creation Date'])
#             location_data = location_data.rename(columns={selected_feature: 'y'})

#             train_size = st.slider("Select Train Data Size", min_value=0.5, max_value=0.9, step=0.1, value=0.7)
#             train_data = location_data[:int(train_size * len(location_data))]
#             test_data = location_data[int(train_size * len(location_data)):]
#             model, forecast = fit_and_forecast(train_data, periods=len(test_data))

#             # Calculate RMSE
#             test_data['Predicted'] = forecast['yhat'].iloc[-len(test_data):].values
#             rmse = calculate_rmse(test_data, forecast)
#             st.write(f"RMSE: {rmse:.2f}")

#             # Plot forecast
#             future_forecast = model.make_future_dataframe(periods=12, freq='M')
#             future_forecast = model.predict(future_forecast)
#             future_forecast['yhat'] = future_forecast['yhat'].apply(lambda x: max(0, x))
#             plot_forecast(train_data, test_data, forecast, future_forecast)

def page_forecast_feature():
    st.header("Forecast Feature")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        selected_feature = st.selectbox("Select Feature for Forecasting", features)

        if selected_feature:
            location_data = data.rename(columns={selected_feature: 'y'})
            location_data['ds'] = pd.to_datetime(location_data['Creation Date'])

            train_size = st.slider("Select Train Data Size", min_value=0.5, max_value=0.9, step=0.1, value=0.7)
            train_data = location_data[:int(train_size * len(location_data))]
            test_data = location_data[int(train_size * len(location_data)):]
            model, forecast = fit_and_forecast(train_data, periods=len(test_data))

            # Calculate RMSE
            test_data['Predicted'] = forecast['yhat'].iloc[-len(test_data):].values
            rmse = calculate_rmse(test_data, forecast)
            st.write(f"RMSE: {rmse:.2f}")

            # Plot forecast
            future_forecast = model.make_future_dataframe(periods=12, freq='M')
            future_forecast = model.predict(future_forecast)
            future_forecast['yhat'] = future_forecast['yhat'].apply(lambda x: max(0, x))
            plot_forecast(train_data, test_data, forecast, future_forecast)
# Page 3: Winsorized Mean Time Taken
def page_winsorized_mean_time():
    st.header("Winsorized Mean Time Taken to Resolve Issues")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load the dataset
        df = pd.read_csv(uploaded_file)
        df = df.iloc[5:, :].reset_index(drop=True)
        # Append previous data if available
        prev_file = st.file_uploader("Upload previous dataset to append (optional)", type="csv")
        if prev_file is not None:
            prev_data = pd.read_csv(prev_file)
            df = pd.concat([prev_data, df], ignore_index=True)
        
        st.write("Dataset successfully loaded and appended (if applicable).")

        # Show data info
        st.write("Data Preview:")
        st.write(df.head())
        st.write("Data Info:")
        st.write(df.info())

        # Filter valid rows
        df_filtered = df[df.apply(is_valid_row, axis=1)]

        # Convert strings to datetime format
        df_filtered['Created Time'] = pd.to_datetime(df_filtered['Created Time'], format='%d-%m-%Y %H:%M')
        df_filtered['Resolved Time'] = pd.to_datetime(df_filtered['Resolved Time'], format='%d-%m-%Y %H:%M')

        # Winsorized Mean by Priority
        st.subheader("Winsorized Mean Time Taken to Resolve by Priority")
        winsorized_time_taken_priority = df_filtered.groupby('Priority').apply(winsorized_mean)

        # Plot by Priority
        fig_priority = px.bar(
            winsorized_time_taken_priority,
            x=winsorized_time_taken_priority.index,
            y=winsorized_time_taken_priority.values,
            labels={'x': 'Priority', 'y': 'Winsorized Mean Time Taken (hours)'},
            title='Winsorized Mean Time Taken to Resolve by Priority',
            color_discrete_sequence=['#FF4BD6']  # Change bar color to pink
        )
        st.plotly_chart(fig_priority)

        # Winsorized Mean by Category
        st.subheader("Winsorized Mean Time Taken to Resolve by Category")
        winsorized_time_taken_category = df_filtered.groupby('Category').apply(winsorized_mean)

        # Plot by Category
        fig_category = px.bar(
            winsorized_time_taken_category,
            x=winsorized_time_taken_category.index,
            y=winsorized_time_taken_category.values,
            labels={'x': 'Category', 'y': 'Winsorized Mean Time Taken (hours)'},
            title='Winsorized Mean Time Taken to Resolve by Category',
            color_discrete_sequence=['#FF4BD6']  # Change bar color to pink
        )
        st.plotly_chart(fig_category)

# Page 4: Upload Data and Preprocess for New Categories
def page_upload_data_new():
    st.header("Upload Data and Preprocess for New Categories")
    sample_file_path = 'sample_Data.csv'
    if os.path.exists(sample_file_path):
        with open(sample_file_path, "rb") as f:
            st.download_button(
                label="Download Sample Dataset",
                data=f,
                file_name='sample_dataset.csv',
                mime='text/csv'
            )
    st.markdown("### Disclaimer:")
    st.markdown("Please ensure that the uploaded CSV file follows the correct format,as mentioned in the sample data provided. Any deviation may lead to errors.")
    


    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
       
        data = pd.read_csv(uploaded_file)
        
        # Preprocess data for new categories
        summary_data = preprocess_data_new(data)
        st.write("Data Preprocessed Successfully for New Categories")

        # Display and provide download links for each location-wise dataset
        st.subheader("Location-wise Datasets")
        for location in locations:
            file_path = f'category_token_{location}.csv'
            if os.path.exists(file_path):
                st.write(f"Dataset for {location}:")

                # Create columns for dataset and pie chart
                col1, col2 = st.columns([2, 1])

                # Display dataset
                with col1:
                    location_data = pd.read_csv(file_path)
                    st.dataframe(location_data)

                    # Provide download link
                    with open(file_path, "rb") as f:
                        st.download_button(label=f"Download {location} Dataset", data=f, file_name=f"category_token_{location}.csv", mime='text/csv')

                # Filter out 'No Inputs' in Category
                df_filtered = data[(data['Site'] == location) & (data['Category'] != 'No Inputs')]

                with col2:
                    # Plot bar chart
                    bar_chart = plot_bar_chart(df_filtered, location)
                    st.plotly_chart(bar_chart)

# # Page 5: Feature Forecasting with Prophet for New Categories
# def page_forecast_feature_new():
#     st.header("Feature Forecasting with Prophet for New Categories")

#     uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

#     if uploaded_file is not None:
#         data = pd.read_csv(uploaded_file)
#         selected_category = st.selectbox("Select Category for Forecasting", categories)

#         if selected_category:
#             location = st.selectbox("Select Location", locations)
#             location_data = data[data['Site'] == location]
#             location_data['ds'] = pd.to_datetime(location_data['Creation Date'])
#             location_data = location_data.rename(columns={selected_category: 'y'})

#             train_size = st.slider("Select Train Data Size", min_value=0.5, max_value=0.9, step=0.1, value=0.7)
#             train_data = location_data[:int(train_size * len(location_data))]
#             test_data = location_data[int(train_size * len(location_data)):]
#             model, forecast = fit_and_forecast(train_data, periods=len(test_data))

#             # Calculate RMSE
#             test_data['Predicted'] = forecast['yhat'].iloc[-len(test_data):].values
#             rmse = calculate_rmse(test_data, forecast)
#             st.write(f"RMSE: {rmse:.2f}")

#             # Plot forecast
#             future_forecast = model.make_future_dataframe(periods=12, freq='M')
#             future_forecast = model.predict(future_forecast)
#             future_forecast['yhat'] = future_forecast['yhat'].apply(lambda x: max(0, x))
#             plot_forecast(train_data, test_data, forecast, future_forecast)
def page_forecast_feature_new():
    st.header("Feature Forecasting with Prophet for New Categories")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        selected_category = st.selectbox("Select Category for Forecasting", categories)

        if selected_category:
            location_data = data.rename(columns={selected_category: 'y'})
            location_data['ds'] = pd.to_datetime(location_data['Creation Date'])

            train_size = st.slider("Select Train Data Size", min_value=0.5, max_value=0.9, step=0.1, value=0.7)
            train_data = location_data[:int(train_size * len(location_data))]
            test_data = location_data[int(train_size * len(location_data)):]
            model, forecast = fit_and_forecast(train_data, periods=len(test_data))

            # Calculate RMSE
            test_data['Predicted'] = forecast['yhat'].iloc[-len(test_data):].values
            rmse = calculate_rmse(test_data, forecast)
            st.write(f"RMSE: {rmse:.2f}")

            # Plot forecast
            future_forecast = model.make_future_dataframe(periods=12, freq='M')
            future_forecast = model.predict(future_forecast)
            future_forecast['yhat'] = future_forecast['yhat'].apply(lambda x: max(0, x))
            plot_forecast(train_data, test_data, forecast, future_forecast)
if __name__ == "__main__":
    main()

