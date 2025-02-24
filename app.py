import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from flask import Flask, request, jsonify, render_template
import os
from datetime import datetime, timedelta
app = Flask(__name__)


# Default parameters
default_ar_params = [0.7, -0.2]
default_ma_params = [0.3, 0.5]
default_d = 1
default_base_value = 30
default_peak_value = 50
default_correlation_factor = 0.5

# Define the factors for electricity usage variation
factors = {
    "Low Usage": 10,
    "Medium Usage": 20,
    "Peak Usage": 30
}

# Define the events function
def events(usage):
    # Define probabilities for event to be True or False based on a geometric distribution
    prob_true = 0.2  # Probability for True event
    perform_variation = np.random.choice([True, False], p=[prob_true, 1-prob_true])
    if perform_variation:
        # Selecting appropriate event type and factor
        factor_choice = np.random.choice(list(factors.keys()))
        factor_value = factors[factor_choice]

        # Randomly choose the time slot
        choice = np.random.choice(["forenoon", "afternoon", "fullday"])

        # Vary usage based on the choice
        if choice == "forenoon":
            usage[8:12] += factor_value
        elif choice == "afternoon":
            usage[13:17] += factor_value
        elif choice == "fullday":
            usage[8:17] += factor_value
    return usage

# Function to generate ARIMA time series for electricity usage
def generate_electricity_usage(length, ar_params=default_ar_params, ma_params=default_ma_params, d=default_d, base_value=default_base_value, peak_value=default_peak_value, season='normal'):
    ar = np.r_[1, -np.array(ar_params)]
    ma = np.r_[1, np.array(ma_params)]
    arma_process = ArmaProcess(ar, ma)
    electricity_usage = arma_process.generate_sample(nsample=length)

    # Adjust usage levels based on season
    if season == 'summer':
        base_value *= 1.2  # Increase base value during summer
        peak_value *= 1.5  # Increase peak value during summer
    elif season == 'monsoon':
        base_value *= 0.8  # Decrease base value during monsoon
        peak_value *= 0.7  # Decrease peak value during monsoon
    elif season == 'holiday':
        peak_value *= 1.1  # Increase peak value during holidays

    # Adding a peak usage component similar to the stochastic model
    electricity_usage += base_value
    electricity_usage[17:] += peak_value
    # Add some random noise
    electricity_usage += np.random.normal(0, 5, size=length)  # Adjust standard deviation as needed

    # Integrate events
    electricity_usage = events(electricity_usage)

    return electricity_usage

# Function to generate ARIMA time series for water usage
def generate_water_usage(length, ar_params=default_ar_params, ma_params=default_ma_params, d=default_d, base_value=default_base_value, peak_value=default_peak_value, season='normal', electricity_usage=None):
    ar = np.r_[1, -np.array(ar_params)]
    ma = np.r_[1, np.array(ma_params)]
    arma_process = ArmaProcess(ar, ma)
    water_usage = arma_process.generate_sample(nsample=length)

    # Adjust usage levels based on season
    if season == 'summer':
        base_value *= 1.2  # Increase base value during summer
        peak_value *= 1.5  # Increase peak value during summer
    elif season == 'monsoon':
        base_value *= 0.8  # Decrease base value during monsoon
        peak_value *= 0.7  # Decrease peak value during monsoon
    elif season == 'holiday':
        peak_value *= 1.1  # Increase peak value during holidays

    # Adding a peak usage component similar to the stochastic model
    water_usage += base_value
    water_usage[17:] += peak_value
    water_usage[6:8] += peak_value
    # Add some random noise and correlation with electricity usage
    water_usage += np.random.normal(0, 5, size=length)  # Adjust standard deviation as needed
    if electricity_usage is not None:
        water_usage += default_correlation_factor * electricity_usage

    # Integrate events
    water_usage = events(water_usage)

    return water_usage

# Function to generate ARIMA time series for gas usage
def generate_gas_usage(length, ar_params=default_ar_params, ma_params=default_ma_params, d=default_d, base_value=default_base_value, peak_value=default_peak_value, season='normal', electricity_usage=None, water_usage=None):
    ar = np.r_[1, -np.array(ar_params)]
    ma = np.r_[1, np.array(ma_params)]
    arma_process = ArmaProcess(ar, ma)
    gas_usage = arma_process.generate_sample(nsample=length)

    # Adjust usage levels based on season
    if season == 'summer':
        base_value *= 1  # Increase base value during summer
        peak_value *= 1.2  # Increase peak value during summer
    elif season == 'monsoon':
        base_value *= 0.8  # Decrease base value during monsoon
        peak_value *= 0.7  # Decrease peak value during monsoon
    elif season == 'holiday':
        peak_value *= 1.3  # Increase peak value during holidays

    # Adding a peak usage component similar to the stochastic model
    gas_usage += base_value
    gas_usage[17:] += peak_value
    gas_usage[5:8] += peak_value
    # Add some random noise and correlation with electricity and water usage
    gas_usage += np.random.normal(0, 3, size=length)  # Adjust standard deviation as needed
    if electricity_usage is not None:
        gas_usage += default_correlation_factor * electricity_usage
    if water_usage is not None:
        gas_usage += default_correlation_factor * water_usage

    # Integrate events
    gas_usage = events(gas_usage)

    return gas_usage

# Function to generate weekly data
def generate_weekly_data(days=7, length=24, season='normal', utility='E', electricity_usage=None, water_usage=None):
    weekly_data = np.zeros((days, length))
    if utility == 'E':
        usage_function = generate_electricity_usage
    elif utility == 'W':
        usage_function = generate_water_usage
    elif utility == 'G':
        usage_function = generate_gas_usage
    else:
        print(f"Invalid utility choice: {utility}")
        return None
    for day in range(days):
        if utility == 'W' and electricity_usage is not None:
            weekly_data[day] = usage_function(length, season=season, electricity_usage=electricity_usage[day])
        elif utility == 'G' and electricity_usage is not None and water_usage is not None:
            weekly_data[day] = usage_function(length, season=season, electricity_usage=electricity_usage[day], water_usage=water_usage[day])
        else:
            weekly_data[day] = usage_function(length, season=season)
    return weekly_data

# Function to generate monthly data
def generate_monthly_data(weeks=4, days_per_week=7, length=24, season='normal', utility='E', electricity_usage=None, water_usage=None):
    monthly_data = np.zeros((weeks * days_per_week, length))
    for week in range(weeks):
        weekly_data = generate_weekly_data(days_per_week, length, season=season, utility=utility, electricity_usage=electricity_usage, water_usage=water_usage)
        monthly_data[week * days_per_week:(week + 1) * days_per_week] = weekly_data
    return monthly_data

# Function to generate yearly data
def generate_yearly_data(months=12, weeks_per_month=4, days_per_week=7, length=24, electricity_seasons=None, water_seasons=None):
    yearly_electricity_data = np.zeros((months * weeks_per_month * days_per_week, length))
    yearly_water_data = np.zeros((months * weeks_per_month * days_per_week, length))
    yearly_gas_data = np.zeros((months * weeks_per_month * days_per_week, length))

    for month in range(months):
        electricity_season = electricity_seasons[month] if electricity_seasons is not None else 'normal'
        water_season = water_seasons[month] if water_seasons is not None else 'normal'

        monthly_electricity_data = generate_monthly_data(weeks_per_month, days_per_week, length, season=electricity_season, utility='E')
        monthly_water_data = generate_monthly_data(weeks_per_month, days_per_week, length, season=water_season, utility='W', electricity_usage=monthly_electricity_data)
        monthly_gas_data = generate_monthly_data(weeks_per_month, days_per_week, length, season='normal', utility='G', electricity_usage=monthly_electricity_data, water_usage=monthly_water_data)

        yearly_electricity_data[month * weeks_per_month * days_per_week:(month + 1) * weeks_per_month * days_per_week] = monthly_electricity_data
        yearly_water_data[month * weeks_per_month * days_per_week:(month + 1) * weeks_per_month * days_per_week] = monthly_water_data
        yearly_gas_data[month * weeks_per_month * days_per_week:(month + 1) * weeks_per_month * days_per_week] = monthly_gas_data

    return yearly_electricity_data, yearly_water_data, yearly_gas_data

# Main function to generate and save utility data based on user input
def generate_and_save_data(nodes):
    utilities = ['E', 'W', 'G']
    for node in range(1, nodes + 1):
        print(f"Configurations for Node {node}:")
        utility_choices = input("Enter the utilities you want to generate data for (comma-separated, e.g., 'E,W,G'): ").upper().split(',')
        for choice in utility_choices:
            if choice not in utilities:
                print(f"Invalid utility choice: {choice}")
                continue
        generate_and_save_utility_data(node, utility_choices)

# Function to generate and save utility data for a specific node
def generate_and_save_utility_data(node, utility_choices):
    electricity_seasons = ['normal', 'normal', 'summer', 'summer', 'summer', 'monsoon', 'monsoon', 'monsoon', 'monsoon', 'normal', 'normal', 'holiday']
    water_seasons = ['normal', 'normal', 'summer', 'summer', 'summer', 'monsoon', 'monsoon', 'monsoon', 'monsoon', 'normal', 'normal', 'holiday']

    for utility_choice in utility_choices:
        if utility_choice == 'E':
            utility_name = 'Electricity'
            data = generate_yearly_data(electricity_seasons=electricity_seasons)
        elif utility_choice == 'W':
            utility_name = 'Water'
            data = generate_yearly_data(water_seasons=water_seasons)
        elif utility_choice == 'G':
            utility_name = 'Gas'
            data = generate_yearly_data()
        else:
            print(f"Invalid utility choice: {utility_choice}")
            continue

        yearly_data = data[0] if utility_choice == 'E' else data[1] if utility_choice == 'W' else data[2]

        # Concatenate daily data to form yearly data
        yearly_data_concatenated = np.concatenate(yearly_data)

        # Create a DataFrame with timestamps dynamically
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()
        index = pd.date_range(start=start_date, end=end_date, freq='H')[:len(yearly_data_concatenated)]

        # Create DataFrame with timestamps and utility usage values
        df = pd.DataFrame({'Datetime': index, f'{utility_name}_Usage': yearly_data_concatenated})

        
        # Save DataFrame to CSV
        save_path = os.path.join('static', f'node{node}_{utility_choice.lower()}.csv')
        df.to_csv(save_path, index=False)

        # Control Actions
        last_30_values = df[f'{utility_name}_Usage'].tail(30*24)
        switch_states = ["No Change"] * len(last_30_values)  # Initialize with default value
        prev_state = None
        for i in range(len(last_30_values) - 1):
            if last_30_values.iloc[i + 1] > last_30_values.iloc[i]:
                if prev_state != "Switch ON":
                    switch_states[i] = "Switch ON"
                prev_state = "Switch ON"
            elif last_30_values.iloc[i + 1] < last_30_values.iloc[i]:
                if prev_state != "Switch OFF":
                    switch_states[i] = "Switch OFF"
                prev_state = "Switch OFF"

        # Create a DataFrame for the last 30 values along with switch states
        last_30_df = pd.DataFrame({'Datetime': last_30_values.index, f'{utility_name}_Usage': last_30_values.values, 'Switch_State': switch_states})
        last_30_df.set_index('Datetime', inplace=True)

        # Save the DataFrame with switch states to CSV
        switch_save_path = os.path.join('static', f'node{node}_{utility_choice.lower()}_last_30_with_switch_states.csv')
        last_30_df.to_csv(switch_save_path)

        print(f"CSV File with Switch States: node{node}_{utility_choice.lower()}_last_30_with_switch_states.csv")

        # Plot the utility usage data
        #plt.figure(figsize=(12, 6))
        #plt.plot(df['Datetime'], df[f'{utility_name}_Usage'], color='blue' if utility_choice == 'E' else 'green' if utility_choice == 'W' else 'red', linewidth=1)
        #plt.title(f'{utility_name} Usage Over Time for Node {node}')
        #plt.xlabel('Date')
        #plt.ylabel(f'{utility_name} Usage')
        #plt.grid(True)
        # plt.show()

@app.route('/submit', methods=['POST'])
def submit():
    data = request.json
    print(data)
    nodes = int(data['numNodes'])
    
    electricity_data = []
    water_data = []
    gas_data = []
    
    for i in range(1, data['numNodes'] + 1):
        electricity_key = f'electricity_node_{i}'
        water_key = f'water_node_{i}'
        gas_key = f'gas_node_{i}'
    
        if electricity_key in data:
            electricity_data.append(int(data[electricity_key]))
        if water_key in data:
            water_data.append(int(data[water_key]))
        if gas_key in data:
            gas_data.append(int(data[gas_key]))
        
    for node in range(1, nodes + 1):
        node_electricity = electricity_data[node - 1]
        node_water = water_data[node - 1]
        node_gas = gas_data[node - 1]
        utility_choices = ""
        if node_electricity == 1:
            utility_choices+="E,"
        if node_water == 1:
            utility_choices+='W,'
        if node_gas == 1:
            utility_choices+='G,'
        generate_and_save_utility_data(node, utility_choices.upper().split(','))
    
    return render_template('dashboard.html')

@app.route('/submit_nodes', methods=['POST'])
def submit_nodes():
    data = request.form
    nodes = int(data['numNodes'])
    print(nodes)
    return render_template('configurations.html', numNodes=nodes)
    
    

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """
    Route to serve static files like CSV files.
    """
    return send_from_directory(os.path.join(app.root_path, 'static'), filename)

if __name__ == '__main__':
    app.run(debug=True)
