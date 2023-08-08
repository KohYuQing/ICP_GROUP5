import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import joblib
from joblib import load
import pickle
from xgboost import XGBRegressor 
import requests
import zipfile
import io
import random

st.set_page_config(page_title='INVEMP Tasty Bytes Group 5', page_icon='🍖🍕🍜')

st.sidebar.title("INVEMP: Inventory/Warehouse Management & Prediction on Sales per Menu Item")
st.sidebar.markdown("This web app allows you to explore the internal inventory of Tasty Bytes. You can explore these functions in the web app (Description of Page)")

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Prediction A', 'Prediction B', 'Bundled Items Sales Analysis', 'Prediction D', 'Prediction E'])

with tab1:
  st.write('hello')
with tab2:
  #Tab 2 code here
  #Hector/Shahid
  st.write('hello')
with tab3:
    st.title('🌭🥤Bundling of Items 🍔🍦')
    st.markdown('________________________________________________')
    st.markdown('This web page displays predictions for the total sales of bundled items. Bundles consist of the top-selling  item from main item category paired with the lowest-selling item from another item category. The aim is to compare 2022 bundled item sales with those of 2021 to determine if there is a sales increase due to item bundling.')
    st.markdown("&nbsp;")
    @st.cache_data
    def load_data():
    # First load the original airbnb listtings dataset
        data = pd.read_csv("final_data_noscaler.csv") #use this for the original dataset, before transformations and cleaning
        return data


    def read_csv_from_zipped_github(url):
    # Send a GET request to the GitHub URL
        response = requests.get(url)

    # Check if the request was successful
        if response.status_code == 200:
            # Create a BytesIO object from the response content
            zip_file = io.BytesIO(response.content)

            # Extract the contents of the zip file
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Assume there is only one CSV file in the zip archive (you can modify this if needed)
                csv_file_name = zip_ref.namelist()[0]
                with zip_ref.open(csv_file_name) as csv_file:
                    # Read the CSV data into a Pandas DataFrame
                    df = pd.read_csv(csv_file)

            return df
        else:
            st.error(f"Failed to retrieve data from {url}. Status code: {response.status_code}")
            return None

    def main():

        # Replace the 'github_url' variable with the actual URL of the zipped CSV file on GitHub
        github_url = "https://github.com/KohYuQing/ICP_INDV_STREAMLIT/raw/main/y2022_data_withqty.zip"
        df = read_csv_from_zipped_github(github_url)


    data = load_data()
    github_url = "https://github.com/KohYuQing/ICP_INDV_STREAMLIT/raw/main/y2022_data_withqty.zip"
    maintable = read_csv_from_zipped_github(github_url)
    github_url_woy2022 = "https://github.com/KohYuQing/ICP_INDV_STREAMLIT/raw/main/woy2022_data.zip"
    woy2022_df = read_csv_from_zipped_github(github_url_woy2022)

    with open('xgbr_gs.pkl', 'rb') as file:
        xgbr_gs = joblib.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = joblib.load(file)

    df = pd.read_csv('final_data_noscaler.csv')
    total_sales = df[['TOTAL_SALES_PER_ITEM']]


    season_mapping = {'WINTER': 0, 'SPRING': 1, 'SUMMER': 2, 'AUTUMN': 3}
    season_reverse_mapping = {v: k for k, v in season_mapping.items()}
    season_labels = list(season_mapping.keys())
    season_values = list(season_mapping.values())

    city_mapping = {'San Mateo': 0, 'Denver': 1, 'Seattle': 2, 'New York City': 3, 'Boston': 4}
    city_reverse_mapping = {v: k for k, v in city_mapping.items()}
    city_labels = list(city_mapping.keys())
    city_values = list(city_mapping.values())

    itemcat_mapping = {'Dessert': 0, 'Beverage': 1, 'Main': 2, 'Snack': 3}
    itemcat_reverse_mapping = {v: k for k, v in itemcat_mapping.items()}
    itemcat_labels = list(itemcat_mapping.keys())

    menut_mapping = {'Ice Cream': 0, 'Grilled Cheese': 1, 'BBQ': 2, 'Tacos': 3, 'Chinese': 4, 'Poutine': 5, 'Hot Dogs': 6, 'Vegetarian': 7, 'Crepes': 8, 'Sandwiches': 9, 'Ramen': 10, 'Ethiopian': 11, 'Gyros': 12, 'Indian': 13, 'Mac & Cheese': 14}
    menut_reverse_mapping = {v: k for k, v in menut_mapping.items()}
    menut_labels = list(menut_mapping.keys())

    truckb_mapping = {'The Mega Melt': 1, 'Smoky BBQ': 2, "Guac n' Roll": 3, 'Peking Truck': 4, 'Revenge of the Curds': 5, 'Not the Wurst Hot Dogs': 6, 'Plant Palace': 7, 'Le Coin des Crêpes': 8, 'Better Off Bread': 9, 'Kitakata Ramen Bar': 10, 'Tasty Tibs': 11, 'Cheeky Greek': 12, "Nani's Kitchen": 13, 'The Mac Shack': 14}
    truckb_reverse_mapping = {v: k for k, v in truckb_mapping.items()}
    truckb_labels = list(truckb_mapping.keys())
    truckb_values = list(truckb_mapping.values())

    menuitem_mapping = {'Mango Sticky Rice': 0, 'Popsicle': 1, 'Waffle Cone': 2, 'Sugar Cone': 3, 'Two Scoop Bowl': 4, 'Lemonade': 5, 'Bottled Water': 6, 'Ice Tea': 7, 'Bottled Soda': 8, 'Ice Cream Sandwich': 9, 'The Ranch': 10, 'Miss Piggie': 11, 
                        'The Original': 12, 'Three Meat Plate': 13, 'Fried Pickles': 14, 'Two Meat Plate': 15, 'Spring Mix Salad': 16, 'Rack of Pork Ribs': 17, 'Pulled Pork Sandwich': 18, 'Fish Burrito': 19, 'Veggie Taco Bowl': 20, 'Chicken Burrito': 21, 'Three Taco Combo Plate': 22,
                        'Two Taco Combo Plate': 23, 'Lean Burrito Bowl': 24, 'Combo Lo Mein': 25, 'Wonton Soup': 26, 'Combo Fried Rice': 27, 'The Classic': 28, 'The Kitchen Sink': 29, 'Mothers Favorite': 30, 'New York Dog': 31, 'Chicago Dog': 32, 'Coney Dog': 33, 'Veggie Burger': 34,
                        'Seitan Buffalo Wings': 35, 'The Salad of All Salads': 36, 'Breakfast Crepe': 37, 'Chicken Pot Pie Crepe': 38, 'Crepe Suzette': 39, 'Hot Ham & Cheese': 40, 'Pastrami': 41, 'Italian': 42, 'Creamy Chicken Ramen': 43, 'Spicy Miso Vegetable Ramen': 44, 'Tonkotsu Ramen': 45,
                        'Veggie Combo': 46, 'Lean Beef Tibs': 47, 'Lean Chicken Tibs': 48, 'Gyro Plate': 49, 'Greek Salad': 50, 'The King Combo': 51, 'Tandoori Mixed Grill': 52, 'Lean Chicken Tikka Masala': 53, 'Combination Curry': 54, 'Lobster Mac & Cheese': 55, 'Standard Mac & Cheese': 56, 
                        'Buffalo Mac & Cheese': 57}
    menuitem_reverse_mapping = {v: k for k, v in menuitem_mapping.items()}
    menuitem_labels = list(menuitem_mapping.keys())

    month_mapping = {'Janurary': 1, 'Feburary': 2, "March": 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
    month_reverse_mapping = {v: k for k, v in month_mapping.items()}
    month_labels = list(month_mapping.keys())
    month_values = list(month_mapping.values())

    value_mapping = {'01': 1, '02': 2, '03': 3, '04': 4, '05': 5, '06': 6, '07': 7, '08': 8, '09': 9, '10': 10, '11': 11, '12': 12}
    


    def get_CITY():
        city = st.selectbox('Select a City', city_labels)
        return city
    city_input = get_CITY()
    city_int = city_mapping[city_input]



    def get_truckb():
        truckb = st.selectbox('Select a Truck Brand Name', truckb_labels)
        return truckb
    truckb_input = get_truckb()
    truckb_int = truckb_mapping[truckb_input]

    
    filtered_rows = []
    filteredw2022_rows = []
    for index, row in maintable.iterrows():
        if (truckb_input in row['TRUCK_BRAND_NAME']) & (city_input in row['CITY']):
            filtered_rows.append(row)
            
    for index, row in woy2022_df.iterrows():
        if (truckb_input in row['TRUCK_BRAND_NAME']) & (city_input in row['CITY']):
            filteredw2022_rows.append(row)

    

    
    filtered_df = pd.DataFrame(filtered_rows, columns= maintable.columns)
    filtered_df_another = pd.DataFrame(filtered_rows, columns= maintable.columns)
    filteredw2022_rows = pd.DataFrame(filteredw2022_rows, columns= woy2022_df.columns)
    filteredw2022_rows['DATE'] = pd.to_datetime(filteredw2022_rows['DATE'])
    filteredw2022_rows['DATE_MONTH'] = filteredw2022_rows['DATE'].dt.strftime('%m')
    filteredw2022_rows['DATE_MONTH'] = filteredw2022_rows['DATE_MONTH'].astype(str)
    filteredw2022_rows['DATE_MONTH'] = filteredw2022_rows['DATE_MONTH'].map(value_mapping)
    filteredw2022_df_list = filteredw2022_rows['DATE_MONTH'].unique().tolist()
    

    # find unique months if month number not in dictionary then drop that value 
    filtered_df_another['DATE'] = pd.to_datetime(filtered_df['DATE'])
    filtered_df_another['DATE_MONTH'] = filtered_df_another['DATE'].dt.strftime('%m')
    filtered_df_another['DATE_MONTH'] = filtered_df_another['DATE_MONTH'].astype(str)
    filtered_df_another['DATE_MONTH'] = filtered_df_another['DATE_MONTH'].map(value_mapping)
    filtered_df_list = filtered_df_another['DATE_MONTH'].unique().tolist()
    

    concat_list = [value for value in filtered_df_list if value in filteredw2022_df_list]
    
    month_list = [1,2,3,4,5,6,7,8,9,10,11,12]
    new_list = [m for m in month_list if m not in concat_list]
    
    month_mapping = {key: value for key, value in month_mapping.items() if value not in new_list}
    
    if not month_mapping:
        st.write('NO RECORDS! CHOOSE AGAIN')
    else:
        month_reverse_mapping = {v: k for k, v in month_mapping.items()}
        month_labels = list(month_mapping.keys())
        month_values = list(month_mapping.values())
        bundle_df = filtered_df[filtered_df['VALUE'] != 0]
        bundle_df = pd.DataFrame(bundle_df)

        def get_month():
            month_chosen = st.selectbox('Select Month', month_labels)
            return month_chosen
        month_input = get_month()
        month_int = month_mapping[month_input]

        

        filterednot2022_rows = []
        filterednot2022_df = woy2022_df.loc[
        (woy2022_df['TRUCK_BRAND_NAME'] == truckb_input) &
        (woy2022_df['CITY'] == city_input)]
        filterednot2022_df['DATE'] = pd.to_datetime(filterednot2022_df['DATE'])
        filterednot2022_df['DATE_MONTH'] = filterednot2022_df['DATE'].dt.strftime('%m')
        filterednot2022_df['DATE_MONTH'] = filterednot2022_df['DATE_MONTH'].astype(str)
        filterednot2022_df['DATE_MONTH'] = filterednot2022_df['DATE_MONTH'].map(value_mapping)
        filterednot2022_df['DATE_MONTH'] = filterednot2022_df['DATE_MONTH'].astype(object)
        filterednot2022_df = filterednot2022_df.loc[filterednot2022_df['DATE_MONTH'] == month_int]

        filterednot2022_df = filterednot2022_df[filterednot2022_df['VALUE'] != 0]
        filterednot2022_df= pd.DataFrame(filterednot2022_df)
        filterednot2022_df = filterednot2022_df[filterednot2022_df['DATE'].dt.year == 2021]
        filter2021 = filterednot2022_df
        filter2021.index = range(len(filter2021))
        qty_df = bundle_df['TOTAL_QTY_SOLD']
        date_df = bundle_df['DATE']
        bundle_df = bundle_df.drop(['TOTAL_SALES_PER_ITEM', 'TOTAL_QTY_SOLD', 'DATE'], axis = 1)
        ## map values to put in dataframe
        bundle_df['SEASON'] = bundle_df['SEASON'].map(season_mapping)
        bundle_df['CITY'] = bundle_df['CITY'].map(city_mapping)
        bundle_df['ITEM_CATEGORY'] = bundle_df['ITEM_CATEGORY'].map(itemcat_mapping)
        bundle_df['MENU_TYPE'] = bundle_df['MENU_TYPE'].map(menut_mapping)
        bundle_df['TRUCK_BRAND_NAME'] = bundle_df['TRUCK_BRAND_NAME'].map(truckb_mapping)
        bundle_df['MENU_ITEM_NAME'] = bundle_df['MENU_ITEM_NAME'].map(menuitem_mapping)
        column_names = []
        column_names = bundle_df.columns.tolist()
        if st.button('Predict Price'):
            input_data = column_names
            input_df = bundle_df
            prediction = xgbr_gs.predict(input_df)
            output_data = pd.DataFrame(input_df, columns = input_df.columns)
            output_data = pd.concat([qty_df, output_data], axis=1)
            output_data = pd.concat([date_df, output_data], axis=1)
            output_data['PREDICTED_PRICE'] = prediction 
            output_data['SEASON'] = output_data['SEASON'].replace({v: k for k, v in season_mapping.items()})
            output_data['CITY'] = output_data['CITY'].replace({v: k for k, v in city_mapping.items()})
            output_data['ITEM_CATEGORY'] = output_data['ITEM_CATEGORY'].replace({v: k for k, v in itemcat_mapping.items()})
            output_data['MENU_TYPE'] = output_data['MENU_TYPE'].replace({v: k for k, v in menut_mapping.items()})
            output_data['TRUCK_BRAND_NAME'] = output_data['TRUCK_BRAND_NAME'].replace({v: k for k, v in truckb_mapping.items()})
            output_data['MENU_ITEM_NAME'] = output_data['MENU_ITEM_NAME'].replace({v: k for k, v in menuitem_mapping.items()})
            output_data['DATE'] = pd.to_datetime(output_data['DATE'])
            output_data['DATE_MONTH'] = output_data['DATE'].dt.strftime('%m')
            output_data['DATE_MONTH'] = output_data['DATE_MONTH'].astype(str)
            output_data['DATE_MONTH'] = output_data['DATE_MONTH'].map(value_mapping)
            output_data['DATE_MONTH'] = output_data['DATE_MONTH'].astype(object)
            output_data = output_data.loc[output_data['DATE_MONTH'] == month_int]

            unique_count = filter2021['DATE'].nunique()
            unique_output_date_list = output_data['DATE'].unique().tolist()
            grouped_data = output_data.groupby('DATE')['PREDICTED_PRICE'].sum()
            grouped_data = pd.DataFrame(grouped_data)
            grouped_data = grouped_data.sort_values(by='PREDICTED_PRICE', ascending=False)
            date_list = []
            date_list = grouped_data.index.tolist()
            unique_dates = date_list[:unique_count]
            final_df = output_data[output_data['DATE'].isin(unique_dates)]
            final_df = final_df.drop(columns=['discount_10%','DATE_MONTH'])
            final_df = final_df.reset_index(drop = True)
            
            filter2021 = filter2021.drop(columns=['discount_10%','DATE_MONTH','TRUCK_ID'])
            filter2021.rename(columns={'TOTAL_SALES_PER_ITEM': 'PREDICTED_PRICE'}, inplace=True)
            filter2021 = filter2021.reindex(columns=final_df.columns, fill_value=None)
            filter2021.rename(columns={'PREDICTED_PRICE': 'TOTAL_SALES_PER_ITEM'}, inplace=True)



            
            st.write('Sales of 2021 (Pre-Bundle)')
            st.write(filter2021)
            st.write('Prediction of 2022 Bundle Sales')
            st.write(final_df)
            


            final_df['PREDICTED_PRICE'] = final_df['PREDICTED_PRICE'].astype(float)
            filter2021['TOTAL_SALES_PER_ITEM'] = filter2021['TOTAL_SALES_PER_ITEM'].astype(float)

            column_sum_2021 = filter2021['TOTAL_SALES_PER_ITEM'].sum()
            column_sum_2022 = final_df['PREDICTED_PRICE'].sum()


            st.success('The total sales for 2021: ${:.2f}.'.format(column_sum_2021))
            st.success('The predicted sales with bundle pricing for 2022: ${:.2f}.'.format(column_sum_2022))

  #Tab 3 code here

with tab4:
  #Tab 4 code here
  st.write('hello')

with tab5:
  #Tab 5 code here
  st.write('hello')