
# Project: Web Scraping

## Introduction

The goal of this project is to consolidate six weather databases using web scraping from [wunderground.com](https://www.wunderground.com/). The scraped data will be used to create an application that predicts the weather for the user. The resulting database will be a time series data.

## Methodology

1. Web scraping of weather data from [wunderground.com](https://www.wunderground.com/).
2. Feature engineering of the scraped data, including creating new variables such as the day of the week, quarter, and day of the year.
3. Exploratory analysis using descriptive statistics.
4. Prediction of today's weather using an XGBoost machine learning model.
5. Preprocessing and feature creation using PySpark and/or pandas.

## Deliverables

* Python notebook
* Streamlit dashboard
  * Presentation page with information about the website and data source
  * Modeling page for weather forecasting
  * Comparison page to measure prediction error against actual temperature
