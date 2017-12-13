# Time-Series-Sales

This project is all about sales forecast (predicting sales). It is based on a real data (monthly sales of a fast food restaurant) and probably the issue here was the size of the serie. I believe with longer serie we can get better results.

Some models were tested and divided into folders as follows:

ARIMA-R
-------

This folder contains the R code file, R Markdown file and the PDF file with complete report of the code , results and comments of the implementation of ARIMA model.

LINEAR REGRESSION -PY
---------------------

This folder cotains diferent approaches, all of them using Python, for Linear Regression models.

In the "Sales Forecast" notebook could be find the code for a One Variable LR using the a time series with previous month sales to predict  the next month sales. Yet, in the same notebook, could be also find the second part of the model, using a window of previous months and choosing the best window to make the prediction.

In the "Sales Forecast Indexes" I gathered the values of some economic indexes during the same period of our time serie, and apply it for a multi variable LR model.






