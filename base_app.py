"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Climate Change Classifier :earth_africa:")
	st.header("_Where do we stand on climate change?_")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Information", "Prediction", "Predict data", "About Model"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == 'Predict data':
		st.markdown("Below you can have a view of the data we used to predict our model")
		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

#st.write(center_info_data)
	# Building out the "About model" page
	if selection == 'About Model':
		st.info('Model used: Linear Support Vector Classifier(SVC) model')
		st.subheader("Why this model?")

		st.markdown("For this classifier we used a Linear Support Vector Classifier model. The goal of this model is to fit to the provided data, returning a 'best fit' hyperplane that categorizes your data. After testing several multiclass predictor models we concluded that the Linear SVC gives us the higest accuracy in term of predicting someone's view on climate change based on their tweets.")
		st.info('More about Linear SVM models: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html')
	#Building out the "Predict Data" page
	if selection == "Information":
		st.subheader("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("With our **Linear Support Vector Classification Model** you will be able to predict whether or not a person believes in climate change with 74% accuracy.")
		st.markdown("It's fairly simple to predict, just head over to the **Prediction** option and follow the instructions")
		st.markdown("")
		st.markdown("")
		st.markdown(':memo: Instructions:')
		st.markdown("1. Enter your preferred text/tweet in the **Enter text** box\n2. Click on the **Classify** button")
		st.markdown("Your answer will show as **Text Categorized as: [X]**. The 'X' defines the number to each sentiment.")
		st.markdown("")
		st.markdown("**💬 Sentiments:**")
		st.markdown("[-1] Anti: the tweet does not believe in man-made climate change\n")
		st.markdown("[0] Neutral: the tweet neither supports nor refutes the belief of man-made climate change")
		st.markdown("[1] Pro: the tweet supports the belief of man-made climate change")
		st.markdown("[2] News: the tweet links to factual news about climate change")
	# Building out the predication page
	if selection == "Prediction":
		st.text("Classify whether or not a person believes in climate change, based on their tweet.")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
