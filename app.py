import joblib
import streamlit as st 

st.title("Jabalpur Housing Prices Prediction")
st.subheader("A web application for predicting house prices in Jabalpur, Madhya Pradesh")
st.markdown("This app uses Machine Learning to predict the prices of houses in Jabalpur.It loads a pre-trained linear regression model which takes as input various features of the house, such as the area,number of bedrooms, number of bathrooms, stories, connection to mainroad, number of guestrooms, basement, hotwaterheating, airconditioning, parking, preferred area, furnishingstatus.")
st.markdown("📖 [GitHub repository](https://www.instagram.com/ansh_vish27?stkn=MWZ1bGd1NWFhZzhnMA==) | ❤️ *My profile:* [@ansh_vish27](https://www.instagram.com/ansh_vish27?stkn=MWZ1bGd1NWFhZzhnMA==)")

st.divider()

st.subheader("Enter attributes of the houses")

with st.form(key="Attribute form"):
    st.subheader("House Attributes")
    area = st.number_input("Enter the house area(in sq.ft.)", min_value = 500)
    number_of_bedrooms = st.select_slider("Number of bedrooms: ",options = [1,2,3,4,5])
    number_of_bathrooms = st.select_slider("Number of bathrooms: ",options = [1,2,3,4,5])
    number_of_stories = st.select_slider("Number of stories: ",options = [1,2,3])
    parking = st.select_slider("Parking area", options = [0,1,2,3])
    mainroad_connection = st.checkbox("Mainroad connection")
    guestroom = st.checkbox("have a guestroom")
    basement = st.checkbox("have a basement")
    heater = st.checkbox("equipped with hot water heater")
    ac = st.checkbox("equipped with well airconditioning")
    preferredarea = st.checkbox("located in a preferred area ?")
    furnished = st.selectbox("select furnishing status: ", options = ['unfurnished','semi-furnished','furnished'])

    button = st.form_submit_button("Predict")

model = joblib.load("model.pkl")

if button:

    if furnished == "unfurnished":
        furnished = 0
    elif furnished == "semi-furnished":
        furnished = 1
    else:
        furnished = 2

    input_data = [[

        area,
        number_of_bedrooms,
        number_of_bathrooms,
        number_of_stories,
        mainroad_connection,
        guestroom,
        basement,
        heater,
        ac,
        parking,
        preferredarea,
        furnished
    ]]
    # st.write(input_data)   -> to see the filled data
    st.subheader("Predicted house price:")
    prediction = model.predict(input_data)
    st.success(f"₹{prediction[0]:,.0f}" )





