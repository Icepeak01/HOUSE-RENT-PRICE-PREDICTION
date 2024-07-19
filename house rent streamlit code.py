import streamlit as st
import time
import numpy as np
import pandas as pd
import joblib


@st.cache_resource(show_spinner = "Loading model")
def load_model():
	model = joblib.load(r"C:\Users\HP\Desktop\streanlit\house_model.pkl")
	return model


@st.cache_resource(show_spinner = "Loading col")
def load_col():
	col_name  = joblib.load(r"C:\Users\HP\Desktop\streanlit\house_col_name.pkl")
	return col_name

	

@st.cache_data(show_spinner = "Predicting...")
def make_prediction(_model, _col_name, house_type, location, bedroom, bathroom, toilet, parking_space):

    cat_df = pd.DataFrame([[house_type, location]], columns=['House_type', 'Location'])
    cat_encoded = pd.get_dummies(cat_df, drop_first=True)

    num_df = pd.DataFrame([[bedroom, bathroom, toilet, parking_space]], columns=['Bedrooms', 'Bathrooms', 'Toilets', 'Parking Spaces'])
    num_df['Rooms'] = num_df['Bedrooms'] * num_df['Bathrooms']

    comb_df = pd.concat([num_df, cat_encoded], axis=1)
    comb_df = comb_df.reindex(columns=col_name, fill_value=0)
   
    # Extract values from combined DataFrame
    price = comb_df.values
    prediction = model.predict(price)
    rounded_prediction = round(prediction[0])
    
    return f'The house rent price for the selected Location will be around {rounded_prediction} Naira'


if __name__ == '__main__':
	st.title('CHECK HOUSE RENT PRICE')

	st.divider()
	col1, col2 = st.columns(2)

	with col1:
		bedroom = st.number_input('How Many Bedroom?', min_value = 0, max_value = 100, value = 1, step = 1)
		bathroom = st.slider('How Many Bathroom?', min_value = 0, max_value = 100, value = 1, step = 1)
		house_type = st.selectbox('Houste Type', ("DD - Detached Duplex", 'TD - Terraced Duplex', "SDD - Semi Detached Duplex", 'H - House'))
		

	with col2:
		parking_space = st.number_input('How Many Parking Space ?', min_value = 0.0, max_value = 100.0, value = 1.0, step = 1.0)
		toilet = st.slider('How Many Toilets ?', min_value = 0, max_value = 100, value = 1, step = 1)
		location = st.selectbox('Where in Lekki Area ?', ('Chevron', 'Ikate', 'Ikota', 'Orchid', 'Osapa', 'Lekki'))





	pred_btn = st.button('APPLY', type = 'primary')
	if pred_btn:
		model = load_model()
		col_name = load_col()
		pred = make_prediction(model, col_name, house_type, location, bedroom, bathroom, toilet, parking_space)
		st.write(pred)