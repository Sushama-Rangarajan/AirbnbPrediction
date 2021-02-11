# AirbnbPrediction

There are 3 datasets used for analysis. 

  1. Overall listings dataset that contained details regarding individual listings in New York (Jan 2020 listings)

  2. Review dataset that had individual customer reviews for each of the listings (Jan 2020 reviews)

  3. Calendar dataset that consisted of pricing details and 1 year availability for each of the listings (Jan 2020 calendar) 
  
The datasets are larger in size and hence not included here. 

Two analysis were executed. 

1. To understand the effect of different features on price

2. To understand if negative reviews influence the price (Queens neighborhood)
   * 3 machine learning models were deployed - Linear Regression, Gradient boost, Random Forest

The reviews were classified as positive and negative using Qdap's polarity score and the count of which has been included in the model for analysis

