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
from streamlit-card import card

st.set_page_config(page_title='INVEMP Tasty Bytes Group 5', page_icon='🍖🍕🍜')

st.sidebar.title("INVEMP: Inventory/Warehouse Management & Prediction on Sales per Menu Item")
st.sidebar.markdown("This web app allows you to explore the internal inventory of Tasty Bytes. You can explore these functions in the web app (Description of Page)")

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Churn Prediction', 'Prediction B', 'Bundled Items Sales Analysis', 'Prediction D', 'Prediction E'])


def tab1_predict(city,sales_level,frequency_level,history_level):

     required = ['TOTAL_PRODUCTS_SOLD', 'ORDER_AMOUNT', 'TOTAL_ORDERS',
       'MIN_DAYS_BETWEEN_ORDERS', 'MAX_DAYS_BETWEEN_ORDERS',
       'frequency_cluster', 'Customer_age_cluster', 'sale_cluster',
       'CITY_Boston', 'CITY_Denver', 'CITY_New York City', 'CITY_San Mateo',
       'CITY_Seattle']
     required = [i.lower() for i in required]
     
     x_test = [3.0, 30.0, 57.0, 1.0, 103.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

     x_test[required.index(f'city_{city.lower()}')] = 1.0
     if frequency_level=="Low-Frequency":
          x_test[required.index('frequency_cluster')] = 0.0
     elif frequency_level=="Medium-Frequency":
          x_test[required.index('frequency_cluster')] = 2.0
     elif frequency_level=="High-Frequency":
          x_test[required.index('frequency_cluster')] = 1.0

     if sales_level=="Low-Spending":
          x_test[required.index('sale_cluster')] = 1.0
     elif sales_level=="Average-Spending":
          x_test[required.index('sale_cluster')] = 2.0
     elif sales_level=="High-Spending":
          x_test[required.index('sale_cluster')] = 0.0

     if history_level=="New Customer":
          x_test[required.index('customer_age_cluster')] = 1.0
     elif history_level=="Standard Customer":
          x_test[required.index('customer_age_cluster')] = 0.0
     elif history_level=="Long-Standing Customer":
          x_test[required.index('customer_age_cluster')] = 2.0

     # x_test[required.index('sale_cluster')] = 2
     # #x_test[required.index('frequency_cluster')] = frequency_level
     # x_test[required.index('customer_age_cluster')] = 1

     with open('final_model.sav','rb') as f:
          model = pickle.load(f)

     print(x_test)
     y = model.predict(np.array([x_test]))[0]          # 0 -> not churn, 1 -> churn
     print('prediction:', y)
     if y == 0:
          result = 'The selected cluster of customers are predicted to not churn'
     else:
          result = 'The selected cluster of customers are predicted to churn'
     st.session_state['tab1_result'] = result
     st.session_state['tab1_churn_prediction'] = y

     df = pd.read_csv('2021.csv')

def get_bar_chart_df(city, frequency_level, sales_level, history_level):
    df = pd.read_csv('2021.csv')
                     
    df = df[df['CITY']==city]
    df = df[df['frequency_cluster']==frequency_level]
    df = df[df['sale_cluster']==sales_level]
    df = df[df['Customer_age_cluster']==history_level]

    menu = df.loc[:,['MENU_TYPE', 'CITY']].groupby('MENU_TYPE').count().sort_values('CITY').reset_index()
    menu = menu.rename(columns={'CITY': 'QTY'}, errors='ignore')
    st.session_state['Menu_Whole'] = menu
    return menu.iloc[[0,1,2], :], menu.iloc[[-3,-2,-1],:]

def get_total_revenue_of_cluster(city, frequency_level, sales_level, history_level):
    df = pd.read_csv('2021.csv')
    df = df[df['CITY']==city]
    df = df[df['frequency_cluster']==frequency_level]
    df = df[df['sale_cluster']==sales_level]
    df = df[df['Customer_age_cluster']==history_level]

    return df['ORDER_AMOUNT'].sum()


with tab1:

     st.title('Churn Prediction And Measures')
     st.markdown('________________________________________________')

#      df_cleaned = dataset.loc[:, ['CITY', 'REGION', 'MENU_TYPE',
#        'TOTAL_PRODUCTS_SOLD', 'ORDER_AMOUNT', 'TOTAL_ORDERS',
#         'MIN_DAYS_BETWEEN_ORDERS', 'MAX_DAYS_BETWEEN_ORDERS',
#        'DAYS_TO_NEXT_ORDER','frequency_cluster','Customer_age_cluster','sale_cluster']]

     # Create three columns for the dropdown lists
     col1_t1, col2_t1, col3_t1,col4_t1 = st.columns(4)

    # Define the fixed choices for the dropdown lists
     cities = ['San Mateo', 'New York City', 'Boston', 'Denver', 'Seattle']
     spending_choices = ["Low-Spending", "Average-Spending", "High-Spending"]
     frequency_choices = ["Low-Frequency","Medium-Frequency", "High-Frequency"]
     cust_history = ["New Customer", "Standard Customer", "Long-Standing Customer"]

    # First dropdown list - Spending Level
     with col1_t1:
          city = st.selectbox("City", options=cities)

    # Second dropdown list - Frequency Level
     with col2_t1:
          sales_level = st.selectbox("Spending Frequency", options=spending_choices)

    # Third dropdown list - Age Level
     with col3_t1:
          frequency_level = st.selectbox("Frequency History", options=frequency_choices)

     with col4_t1:
          history_level = st.selectbox("Customer History", options=cust_history)

     if 'tab1_result' not in st.session_state:
          st.session_state['tab1_result'] = ''
     
     if 'tab1_insights' not in st.session_state:
          st.session_state['tab1_insights'] = ''

     button_return_value = st.button("Predict", on_click=tab1_predict, args=(city,sales_level,frequency_level,history_level))

    # Prediction section
     st.header("Prediction")
     result = st.session_state['tab1_result']
     print(result)
     st.write(st.session_state.get('tab1_result'))
     # st.text_area('', value=st.session_state.get('tab1_result'))
     # Measures

     if button_return_value:

          

          st.header("Insights & Measures")
          args = [city, 0, 0, 0]

          if frequency_level=="Low-Frequency": args[1] = 0
          elif frequency_level=="Medium-Frequency": args[1] = 2
          elif frequency_level=="High-Frequency": args[1] = 1

          if sales_level=="Low-Spending": args[2] = 1
          elif sales_level=="Average-Spending":  args[2] = 2
          elif sales_level=="High-Spending":  args[2] = 0

          if history_level=="New Customer":  args[3] = 1
          elif history_level=="Standard Customer": args[3] = 0
          elif history_level=="Long-Standing Customer": args[3] = 2

          total_revenue = get_total_revenue_of_cluster(*args)
          card(title=str(total_revenue), text='Total Sales Revenue Generated')
          # st.metric("Total Sales Revenue",total_revenue)

          print(get_bar_chart_df(*args), args)
          st.subheader("Bottom 3 Popular Menu")
          st.bar_chart(get_bar_chart_df(*args)[0], x='MENU_TYPE', y='QTY')
          st.subheader("Top 3 Popular Menu")
          st.bar_chart(get_bar_chart_df(*args)[1], x='MENU_TYPE', y='QTY')

          top_menu, bottom_menu = get_bar_chart_df(*args)
          full_menu_data = get_bar_chart_df(*args)[0]

    # Extract the top and bottom 3 menu types
          top_menu_types = top_menu['MENU_TYPE'].tolist()
          bottom_menu_types = bottom_menu['MENU_TYPE'].tolist()

    # Convert the menu types to formatted strings
          top_menu_types_str = ", ".join(top_menu_types)
          bottom_menu_types_str = ", ".join(bottom_menu_types)

          churn_prediction = st.session_state.get('tab1_churn_prediction', None)
          full_menu_data = st.session_state['Menu_Whole']
          ascending_menu_data = full_menu_data.sort_values(by='QTY', ascending=False)
          st.subheader("Menu Order Types")
          st.dataframe(ascending_menu_data)
          
          if churn_prediction == 0:  # Not churned
               st.write("##### Since the customers in this cluster are predicted to stay,\n ##### Here are more strategies to continue to entice these customers to buy more!")
               st.write(f"1. For the bottom 3 popular items, customers' favorite food menus are **{top_menu_types_str}**.")
               st.write(f"2. For the top 3 popular items, customers' favorite food menus are **{bottom_menu_types_str}**. "
             "Promotional strategies such as giving discounts and vouchers could incentivize them to buy more of these items.")
          else:  # Churned
               st.write("##### Since Customers in this cluster are predicted to churn,\n ##### Here are some strategies to retain them.")
               st.write(f"1. For the bottom 3 popular items, customers' favorite food menus are **{top_menu_types_str}**.")
               st.write(f"2. For the top 3 popular items, customers' favorite food menus are **{bottom_menu_types_str}**. "
             "Promotional strategies such as giving discounts and vouchers could incentivize them to buy more of these items.")











with tab2:
     st.title('Calculation of Revenue ')
     st.markdown("This tab predicts whether or not the customers in a selected cluster is likely to churn. It also includes insights on the selected cluster, such as their total revenue by year \
                 as well as the number of orders made by this cluster for each menu type. \nAt the bottom, there is a revenue calculation to estimate the revenue by this cluster \
                 in the following year if they do not churn. This calculation is based on the cluster's revenue generated in the previous years.")
     st.markdown('________________________________________________')

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
    
     # gu_acd  =  "https://github.com/ShahidR-np/testzipfiles/raw/main/allcustdata.zip"
     # custdata = read_csv_from_zipped_github(gu_acd)
     gu_od = "https://github.com/ShahidR-np/testzipfiles/raw/main/orderdatav2.zip"
     orderdata = read_csv_from_zipped_github(gu_od)

     col1_t2, col2_t2, col3_t2 = st.columns(3)
    
    # First dropdown list - Spending Level
     with col1_t2:
          spending_level_t2 = st.selectbox("Customer Spending", ("High-Spending", "Average-Spending", "Low-Spending"))

    # Second dropdown list - Frequency Level
     with col2_t2:
          frequency_level_t2 = st.selectbox("Customer Frequency", ("High Frequency", "Average Frequency", "Low Frequency"))

    # Third dropdown list - Age Level
     with col3_t2:
          history_level_t2 = st.selectbox("Customer's History", ("Long-Standing", "Regular", "New"))

     #Variables
     generatedsales = 0 #The generate revenue for the cluster
     increasesales = 0 #The increase of revenue
     increaseperc = 0 #The increase of percentage in sales

     #Cluster vals
     freq_dict= {'High Frequency':0, 'Average Frequency':2, 'Low Frequency':1}
     spend_dict= {'High-Spending':0, 'Average-Spending':2, 'Low-Spending':1}
     hist_dict= {"Long-Standing":2, "Regular":0, "New":1}

     freq_val = freq_dict[frequency_level_t2]
     spend_val = spend_dict[spending_level_t2]
     hist_val = hist_dict[history_level_t2]
     if hist_val == 0:
        od = pd.read_csv("./custdatav0.csv")
     elif hist_val == 1:
        od = pd.read_csv("./custdatav1.csv")
     elif hist_val == 2:
        od = pd.read_csv("./custdatav2.csv")
     #Filtering data based on clusters
     #v1filtered = custdatav1[(custdatav1['sale_cluster'] == spend_val) & (custdatav1['Customer_age_cluster'] == hist_val) & (custdatav1['frequency_cluster'] == freq_val )]
     #v2filtered = custdatav2[(custdatav2['sale_cluster'] == spend_val) & (custdatav2['Customer_age_cluster'] == hist_val) & (custdatav2['frequency_cluster'] == freq_val )]
     filteredod = orderdata[(orderdata['sale_cluster'] == spend_val) & (orderdata['Customer_age_cluster'] == hist_val) & (orderdata['frequency_cluster'] == freq_val )]
     odgb = filteredod.groupby(['YEAR_OF_ORDER'])['ORDER_AMOUNT'].sum()
     #filteredcd = pd.concat([v1filtered, v2filtered])
     filteredcd = od[(od['sale_cluster'] == spend_val) & (od['frequency_cluster'] == freq_val )]
     clustermode = filteredcd.mode()
     gbmt = filteredod.groupby(['MENU_TYPE'])['MENU_TYPE'].count()

     st.header("Insights")
     st.write("Total Revenue by Year")
     st.bar_chart(odgb)
     st.write("Number of orders by menu type")
     st.table(gbmt)

     # Model and Prediction
     with open('cdc_xgb.pkl', 'rb') as file:
         cdcxgb = pickle.load(file)
     
     clustermode['frequency_cluster'] = freq_val
     clustermode['Customer_age_cluster'] = hist_val
     clustermode['sale_cluster'] = spend_val


     predictedchurn=cdcxgb.predict(clustermode[['TOTAL_PRODUCTS_SOLD', 'ORDER_AMOUNT', 'TOTAL_ORDERS',
       'MIN_DAYS_BETWEEN_ORDERS', 'MAX_DAYS_BETWEEN_ORDERS',
       'frequency_cluster', 'Customer_age_cluster', 'sale_cluster',
       'CITY_Boston', 'CITY_Denver', 'CITY_New York City', 'CITY_San Mateo',
       'CITY_Seattle', 'REGION_California', 'REGION_Colorado',
       'REGION_Massachusetts', 'REGION_New York', 'REGION_Washington',
       'MENU_TYPE_BBQ', 'MENU_TYPE_Chinese', 'MENU_TYPE_Crepes',
       'MENU_TYPE_Ethiopian', 'MENU_TYPE_Grilled Cheese', 'MENU_TYPE_Gyros',
       'MENU_TYPE_Hot Dogs', 'MENU_TYPE_Ice Cream', 'MENU_TYPE_Indian',
       'MENU_TYPE_Mac & Cheese', 'MENU_TYPE_Poutine', 'MENU_TYPE_Ramen',
       'MENU_TYPE_Sandwiches', 'MENU_TYPE_Tacos', 'MENU_TYPE_Vegetarian']])
  
     
     #predictedchurn = 1
     churntext = ""
     if (predictedchurn == 1):
          churntext = "LESS"
     else: 
          churntext = "MORE"

     odgb2022 = filteredod[filteredod['YEAR_OF_ORDER'] == 2022]
     odgb2021 = filteredod[filteredod['YEAR_OF_ORDER'] == 2021]
     odgb2020 = filteredod[filteredod['YEAR_OF_ORDER'] == 2020]
     odgb2019 = filteredod[filteredod['YEAR_OF_ORDER'] == 2019]
     avemth2022 = odgb[2022] / odgb2022['MONTH_OF_ORDER'].nunique()
     avemth2021 = odgb[2021] / odgb2021['MONTH_OF_ORDER'].nunique()
     avemth2020 = odgb[2020] / odgb2020['MONTH_OF_ORDER'].nunique()
     avemth2019 = odgb[2019] / odgb2019['MONTH_OF_ORDER'].nunique()

     percinc2020 = ((avemth2020-avemth2019)/avemth2019) * 100
     percinc2021 = ((avemth2021-avemth2020)/avemth2020) * 100
     percinc2022 = ((avemth2022-avemth2021)/avemth2021) * 100

     roi2021 = ((percinc2021 - percinc2020)/percinc2020) * 100
     roi2022 = ((percinc2022 - percinc2021)/percinc2021) * 100
     percinc2023 = ((100 + ((roi2021+roi2022)/2)) /100) * percinc2022
     avemth2023 = avemth2022 * ((100 + percinc2023)/100)
     odgb2023 = avemth2023 * 12



     generatedsales = odgb[2022]
     increasesales = odgb[2022] - odgb[2021]
     increaseperc = increasesales / generatedsales * 1002

     st.header("Prediction")
     st.write ("This cluster of customers is " + churntext + " likely to churn as compared to other clusters")
     st.write("After the implementation of discount coupon vouchers to these group of customer,")
     st.write("- The group of customer is less likely to churn the following year")

     st.header("Revenue Calculation")
     st.write("- In the following year, this group of customer is predicted to have an increase of {1:.2f}% in revenue sales, which is a total revenue of {0:.2f}".format(odgb2023, percinc2023 ))
  
with tab3:
    st.title('🌭🥤Bundling of Items 🍔🍦')
    st.markdown('________________________________________________')
    st.markdown('This web page displays predictions for the total sales of bundled items. Bundles consist of the top-selling  item from main item category paired with the lowest-selling item from another item category. The aim is to compare 2022 bundled item sales with those of 2021 to determine if there is a sales increase due to item bundling.')
    st.markdown("&nbsp;")
    @st.cache_data
    


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


    
    github_url = "https://github.com/KohYuQing/ICP_INDV_STREAMLIT/raw/main/y2022_data_withqty.zip"
    maintable = read_csv_from_zipped_github(github_url)
    github_url_woy2022 = "https://github.com/KohYuQing/ICP_INDV_STREAMLIT/raw/main/woy2022_data.zip"
    woy2022_df = read_csv_from_zipped_github(github_url_woy2022)

    with open('xgbr_gs.pkl', 'rb') as file:
        xgbr_gs = joblib.load(file)

    


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
      
      github_url_WO2022 = "https://github.com/kaylaong883/ASG3_streamlit_tab/raw/main/woy2022_data.zip"
      df_WO2022 = read_csv_from_zipped_github(github_url_WO2022)
      github_url = "https://github.com/kaylaong883/ASG3_streamlit_tab/raw/main/y2022_qty_data.zip"
      maintable = read_csv_from_zipped_github(github_url)

      with open('xgbr_gs.pkl', 'rb') as file:
        xgbr_gs = joblib.load(file)
        
      # Define the app title and favicon
      st.title('🚚🍝 Seasonal Menu Variations 🚛🥗') 
      st.subheader('Predict')
      st.markdown("This interactive tab serves as a strategic tool for evaluating the potential benefits of introducing additional trucks during specific seasons. It empowers you to make informed predictions regarding total truck sales based on varying fleet sizes. By inputting the desired number of trucks for implementation, you can project the expected total sales. Furthermore, this tool calculates the average sales for each truck, facilitating a meaningful comparison with historical sales data from previous years. This insightful analysis aids in determining the viability and profitability of expanding your fleet during specific seasons, enabling you to make well-informed decisions for your business growth strategy. ")
      
      st.header('Truck Menu Available')
      
      data = {
          'Truck Name': ["Guac n' Roll", "Tasty Tibs", "The Mac Shack", "Peking Truck", "Le Coin des Crêpes", "Freezing Point", "Nani's Kitchen", "The Mega Melt", "Better Off Bread", "Not the Wurst Hot Dogs", "Plant Palace", "Cheeky Greek", "Revenge of the Curds", "Kitakata Ramen Bar", "Smoky BBQ"],
          'Menu Name': ['Tacos', 'Ethiopian', 'Mac & Cheese', 'Chinese', 'Crepes', 'Ice Cream', 'Indian', 'Grilled Cheese', 'Sandwiches', 'Hot Dogs', 'Vegetarian', 'Gyros', 'Poutine', 'Ramen', 'BBQ']
      }
      truck_menu_table = pd.DataFrame(data)
      
      # Display the DataFrame as a table
      st.table(truck_menu_table)

      st.write("Select a season, the brand name of the truck you're interested in, and specify the desired number of trucks for implementation. This will provide you with the projected average sales per truck following the implementation.")
            
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
      
      truckb_mapping = {'Freezing Point': 0, 'The Mega Melt': 1, 'Smoky BBQ': 2, "Guac n' Roll": 3, 'Peking Truck': 4, 'Revenge of the Curds': 5, 'Not the Wurst Hot Dogs': 6, 'Plant Palace': 7, 'Le Coin des Crêpes': 8, 'Better Off Bread': 9, 'Kitakata Ramen Bar': 10, 'Tasty Tibs': 11, 'Cheeky Greek': 12, "Nani's Kitchen": 13, 'The Mac Shack': 14}
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
  
      def get_season():
        season = st.selectbox('Select a season', season_labels)
        return season

      season_input = get_season()
      season_int = season_mapping[season_input]
      
      def get_truckb():
          truckb = st.selectbox('Select a Truck Brand Name', truckb_labels)
          return truckb
          
      truckb_input = get_truckb()
      truckb_int = truckb_mapping[truckb_input]
  
      filter_rows = []
      for index, row in maintable.iterrows():
        if (season_input in row['SEASON']) & (truckb_input in row['TRUCK_BRAND_NAME']):
          filter_rows.append(row)
          
      filter_df = pd.DataFrame(filter_rows, columns=maintable.columns)

      filter_df = filter_df.drop(columns=['TOTAL_SALES_PER_ITEM','DATE'])

      # user input for number of trucks
      user_truck_input = st.number_input("Enter the number of trucks you want to implement", min_value=0, max_value=100)
      st.write("No. of trucks:", user_truck_input)

      # GENERATE RECORD FOR NEW TRUCKS
      # Initialize an empty list to store generated data
      data = []
      
      # List of possible options
      location = filter_df['LOCATION_ID'].unique()
      
      min_quantity = filter_df['TOTAL_QTY_SOLD'].min()
      max_quantity = filter_df['TOTAL_QTY_SOLD'].max()
      
      shift_no = filter_df['SHIFT_NUMBER'].unique()
      
      city = filter_df['CITY'].unique()
      
      subcat = filter_df['SUBCATEGORY'].unique()
      
      menu_type = filter_df['MENU_TYPE'].unique()
      
      min_air = filter_df['AVG_TEMPERATURE_AIR_2M_F'].min()
      max_air = filter_df['AVG_TEMPERATURE_AIR_2M_F'].max()
      
      min_wb = filter_df['AVG_TEMPERATURE_WETBULB_2M_F'].min()
      max_wb = filter_df['AVG_TEMPERATURE_WETBULB_2M_F'].max()
      
      min_dp = filter_df['AVG_TEMPERATURE_DEWPOINT_2M_F'].min()
      max_dp = filter_df['AVG_TEMPERATURE_DEWPOINT_2M_F'].max()
      
      min_wc = filter_df['AVG_TEMPERATURE_WINDCHILL_2M_F'].min()
      max_wc = filter_df['AVG_TEMPERATURE_WINDCHILL_2M_F'].max()
      
      min_ws = filter_df['AVG_WIND_SPEED_100M_MPH'].min()
      max_ws = filter_df['AVG_WIND_SPEED_100M_MPH'].max()
      
      
      # Initialize an empty dictionary to store item details
      item_details = {}
      
      for index, row in filter_df.iterrows():
          menu_item_name = row['MENU_ITEM_NAME']
          item_category = row['ITEM_CATEGORY']
          cost_of_goods = row['COG_PER_ITEM_USD']
          item_price = row['ITEM_PRICE']
      
          item_details[menu_item_name] = {
              'ITEM_CATEGORY': item_category,
              'COG_PER_ITEM_USD': cost_of_goods,
              'ITEM_PRICE': item_price
          }
          
      
      for user in range(user_truck_input):
          TRUCK_ID = user + 101  # Starting truck ID for each user
      
          # Generate 700 rows of data
          for i in range(600):
      
              LOCATION_ID = np.random.choice(location)
      
              TOTAL_QTY_SOLD = np.random.randint(min_quantity, max_quantity + 1)
              
              SHIFT_NUMBER = np.random.choice(shift_no)
              
              CITY = np.random.choice(city)
              
              SUBCATEGORY = np.random.choice(subcat)
              
              MENU_TYPE = np.random.choice(menu_type)
              
              TRUCK_BRAND_NAME = truckb_input
              
              AVG_TEMPERATURE_AIR_2M_F = np.random.randint(min_air, max_air + 1)
              
              AVG_TEMPERATURE_WETBULB_2M_F = np.random.randint(min_wb , max_wb + 1)
              
              AVG_TEMPERATURE_DEWPOINT_2M_F = np.random.randint(min_dp, max_dp + 1)
              
              AVG_TEMPERATURE_WINDCHILL_2M_F = np.random.randint(min_wc, max_wc + 1)
              
              AVG_WIND_SPEED_100M_MPH = np.random.randint(min_ws, max_ws + 1)
              
              SEASON = season_input
      
              MENU_ITEM_NAME = np.random.choice(filter_df['MENU_ITEM_NAME'])
              ITEM_CATEGORY = item_details[MENU_ITEM_NAME]['ITEM_CATEGORY']
              COG_PER_ITEM_USD = item_details[MENU_ITEM_NAME]['COG_PER_ITEM_USD']
              ITEM_PRICE = item_details[MENU_ITEM_NAME]['ITEM_PRICE']
              
              VALUE = 0
              
              DISCOUNT = ITEM_PRICE
      
              data.append({
                  'LOCATION_ID':LOCATION_ID,
                  'TRUCK_ID':TRUCK_ID,
                  'TOTAL_QTY_SOLD':TOTAL_QTY_SOLD,
                  'SHIFT_NUMBER':SHIFT_NUMBER,
                  'CITY':CITY,
                  'ITEM_CATEGORY': ITEM_CATEGORY,
                  'SUBCATEGORY':SUBCATEGORY,
                  'MENU_TYPE':MENU_TYPE,
                  'TRUCK_BRAND_NAME':TRUCK_BRAND_NAME,
                  'MENU_ITEM_NAME': MENU_ITEM_NAME,
                  'AVG_TEMPERATURE_AIR_2M_F':AVG_TEMPERATURE_AIR_2M_F,
                  'AVG_TEMPERATURE_WETBULB_2M_F':AVG_TEMPERATURE_WETBULB_2M_F,
                  'AVG_TEMPERATURE_DEWPOINT_2M_F':AVG_TEMPERATURE_DEWPOINT_2M_F,
                  'AVG_TEMPERATURE_WINDCHILL_2M_F':AVG_TEMPERATURE_WINDCHILL_2M_F,
                  'AVG_WIND_SPEED_100M_MPH':AVG_WIND_SPEED_100M_MPH,
                  'SEASON':SEASON,
                  'COG_PER_ITEM_USD':COG_PER_ITEM_USD,
                  'ITEM_PRICE': ITEM_PRICE,
                  'VALUE':VALUE,
                  'discount_10%':DISCOUNT
              })
      
      # Create a DataFrame from the generated data
      df_generated = pd.DataFrame(data)

      # JOIN filter_df and df_generated
      frames = [filter_df, df_generated]
      prediction_table = pd.concat(frames)

      if st.button('Predict Sales'):
        prediction_table['VALUE'] = 0
        prediction_table['discount_10%'] = prediction_table['ITEM_PRICE']
        truck_list = prediction_table['TRUCK_ID'] 
        qty_list = prediction_table['TOTAL_QTY_SOLD']
        prediction_table = prediction_table.drop(columns=['TRUCK_ID','TOTAL_QTY_SOLD'])

        # Change values to numeric for model to predict
        ## map values to put in dataframe
        prediction_table['SEASON'] = prediction_table['SEASON'].map(season_mapping)
        prediction_table['CITY'] = prediction_table['CITY'].map(city_mapping)
        prediction_table['ITEM_CATEGORY'] = prediction_table['ITEM_CATEGORY'].map(itemcat_mapping)
        prediction_table['MENU_TYPE'] = prediction_table['MENU_TYPE'].map(menut_mapping)
        prediction_table['TRUCK_BRAND_NAME'] = prediction_table['TRUCK_BRAND_NAME'].map(truckb_mapping)
        prediction_table['MENU_ITEM_NAME'] = prediction_table['MENU_ITEM_NAME'].map(menuitem_mapping)
        column_names = []
        column_names = prediction_table.columns.tolist()

        input_data = column_names
        input_df = prediction_table
        prediction = xgbr_gs.predict(input_df)
        output_data = pd.DataFrame(input_df, columns = input_df.columns)
        output_data['PREDICTED_PRICE'] = prediction 
        
        output_data = pd.concat([truck_list, qty_list, output_data], axis=1)

        output_data['SEASON'] = output_data['SEASON'].map(season_reverse_mapping)
        output_data['CITY'] = output_data['CITY'].map(city_reverse_mapping)
        output_data['ITEM_CATEGORY'] = output_data['ITEM_CATEGORY'].map(itemcat_reverse_mapping)
        output_data['MENU_TYPE'] = output_data['MENU_TYPE'].map(menut_reverse_mapping)
        output_data['TRUCK_BRAND_NAME'] = output_data['TRUCK_BRAND_NAME'].map(truckb_reverse_mapping)
        output_data['MENU_ITEM_NAME'] = output_data['MENU_ITEM_NAME'].map(menuitem_reverse_mapping)

        # filter output data to only contain predicted trucks
        output_data = output_data[output_data['TRUCK_ID'] > 100]
        st.write(output_data)

        # 2022 DATA
        filter_rows_2022 = []
        for index, row in maintable.iterrows():
          if (season_input in row['SEASON']) & (truckb_input in row['TRUCK_BRAND_NAME']):
            filter_rows_2022.append(row)
          
        filter_rows_2022 = pd.DataFrame(filter_rows_2022, columns=maintable.columns)
      
        total_sales_of_trucks_2022 = 0
        truck_avail_2022 = filter_rows_2022['TRUCK_ID'].unique()
        
        # Create a list to hold truck information
        truck_info_2022 = []
        
        for truck in truck_avail_2022:
            total_sales_2022 = filter_rows_2022[filter_rows_2022['TRUCK_ID'] == truck]['TOTAL_SALES_PER_ITEM'].sum()
            truck_info_2022.append({'Truck': truck, 'Total Sales': total_sales_2022})

        truck_info_2022_display = truck_info_2022.copy()
        
        # truck sales with predicted truck
        total_sales_of_trucks = 0
        trucks_available = output_data['TRUCK_ID'].unique()

        # Create a list to hold truck information
        truck_info = []

        for truck in trucks_available:
            total_sales = output_data[output_data['TRUCK_ID'] == truck]['PREDICTED_PRICE'].sum()
            truck_info.append({'Truck': truck, 'Total Sales': total_sales})

        truck_info_2022.extend(truck_info)
        truck_info = truck_info_2022

        # Display truck information in a table
        st.table(pd.DataFrame(truck_info))
        
        # Calculate total and average sales
        total_sales_of_trucks = sum(info['Total Sales'] for info in truck_info)
        average_sales = total_sales_of_trucks / len(trucks_available)
        
        # Print total sales for all trucks combined
        st.write(f"Total sales for all {len(trucks_available) + 5} trucks: ${total_sales_of_trucks:.2f}")
        
        # Display average sales
        st.subheader(f"Average sales for each truck: ${average_sales:.2f}")


        # FOR COMPARISON WITH 2022 DATA
        st.header(f"Comparison with 2022 data")
        st.write(filter_rows_2022)
        
        # PRINTING OF 2022 TRUCK INFO
        # Calculate total and average sales
        total_sales_of_trucks_2022 = sum(info['Total Sales'] for info in truck_info_2022_display)
        average_sales_2022 = total_sales_of_trucks_2022 / len(truck_avail_2022)
      
        # Display truck information in a table
        st.table(pd.DataFrame(truck_info_2022_display))
        
        # Print total sales for all trucks combined
        st.write(f"Total sales for all {len(truck_avail_2022)} trucks: ${total_sales_of_trucks_2022:.2f}")
        
        # Display average sales
        st.subheader(f"Average sales for each truck: ${average_sales_2022:.2f}")


        # FOR COMPARISON WITH 2021 DATA
        st.header(f"Comparison with 2021 data")
        # Convert the 'Date' column to datetime format
        df_WO2022['DATE'] = pd.to_datetime(df_WO2022['DATE'])
        df_2021 = df_WO2022[df_WO2022['DATE'].dt.year == 2021]

        filter_rows_2021 = []
        if truckb_input not in df_2021['TRUCK_BRAND_NAME'].values:
            st.error(f"Truck brand name '{truckb_input}' not found in the DataFrame.")
        else:
            for index, row in df_2021.iterrows():
                if (season_input in row['SEASON']) and (truckb_input in row['TRUCK_BRAND_NAME']):
                    filter_rows_2021.append(row)
          
        filter_rows_2021 = pd.DataFrame(filter_rows_2021, columns=df_2021.columns)
        
        st.write(filter_rows_2021)
        total_sales_of_trucks_2021 = 0
        truck_avail_2021 = filter_rows_2021['TRUCK_ID'].unique()

        # Create a list to hold truck information
        truck_info_2021 = []
        
        for truck in truck_avail_2021:
            total_sales_2021 = filter_rows_2021[filter_rows_2021['TRUCK_ID'] == truck]['TOTAL_SALES_PER_ITEM'].sum()
            truck_info_2021.append({'Truck': truck, 'Total Sales': total_sales_2021})
        
        # Display truck information in a table
        st.table(pd.DataFrame(truck_info_2021))
        
        # Calculate total and average sales
        total_sales_of_trucks_2021 = sum(info['Total Sales'] for info in truck_info_2021)
        average_sales_2021 = total_sales_of_trucks_2021 / len(truck_avail_2021)
        
        # Print total sales for all trucks combined
        st.write(f"Total sales for all {len(truck_avail_2021)} trucks: ${total_sales_of_trucks_2021:.2f}")
        
        # Display average sales
        st.subheader(f"Average sales for each truck: ${average_sales_2021:.2f}")

        st.header("Breakdown of Cost for Buying a Food Truck 💸🚚")
        truck_cost = 50000
        operating_costs = 1500
        equipment_costs = 10000
        liscenses_permit = 28000
        other_costs = 2000
        output_data['cog'] = output_data['TOTAL_QTY_SOLD'] * output_data['COG_PER_ITEM_USD']
        cog = output_data['cog'].sum()
        total_cost = truck_cost + operating_costs + equipment_costs + liscenses_permit + other_costs + cog

        st.write(f"Food Truck Cost: ${truck_cost}")
        st.write(f"Operating Costs: ${operating_costs} per month")
        st.write(f"Equipment Costs: ${equipment_costs}")
        st.write(f"Equipment Costs: ${equipment_costs}")
        st.write(f"Licenses and Permit Costs: ${liscenses_permit}")
        st.write(f"Costs of Goods: ${cog}")
        st.write(f"Other Costs: ${other_costs}")
    
        st.subheader("Total Cost: ${:.2f}".format(total_cost))
with tab5:
  #Tab 5 code here
  st.write('hello')
