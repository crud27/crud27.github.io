## ExtraaLearn Decision Tree and Random Forest

**Business Description:** 
The EdTech industry has been surging in the past decade immensely, and according to a forecast, the Online Education market will be worth $286.62bn by 2023 with a compound annual growth rate (CAGR) of 10.26% from 2018 to 2023. 
The modern era of online education has enforced a lot in its growth and expansion beyond any limit. 
Due to having many dominant features like ease of information sharing, personalized learning experience, transparency of assessment, etc, it is now preferable to traditional education.

In the present scenario due to the Covid-19, the online education sector has witnessed rapid growth and is attracting a lot of new customers. 
Due to this rapid growth, many new companies have emerged in this industry. 
With the availability and ease of use of digital marketing resources, companies can reach out to a wider audience with their offerings. 
The customers who show interest in these offerings are termed as leads. There are various sources of obtaining leads for Edtech companies, like

- The customer interacts with the marketing front on social media or other online platforms.
- The customer browses the website/app and downloads the brochure
- The customer connects through emails for more information.
- The company then nurtures these leads and tries to convert them to paid customers. For this, the representative from the organization connects with the lead on call or through email to share further details.

### 1. Objective
ExtraaLearn is an initial-stage startup that offers programs on cutting-edge technologies to students and professionals to help them upskill/reskill. 
Identify which leads are more likely to convert so that they can allocate resources accordingly. 
You, as a data scientist at ExtraaLearn, have been provided the leads data to:
- Analyze and build an ML model to help identify which leads are more likely to convert to paid customers,
- Find the factors driving the lead conversion process
- Create a profile of the leads which are likely to convert

### 2. Data Analysis

Utilized Python to analyze the data set

Data Description
*Rows and columns are present in the data* - 
```javascript
# use the shape method to determine the number of rows and columns of the data
df.shape
```
**Observations:**
(1898, 9)

*Datatypes of the different columns in the dataset*
```javascript
# Use info() to print a concise summary of the DataFrame
df.info()
```
**Observations:**
- There are 4 features (columns) which are int64
- There is 1 feature (column), cost_of_the_order, which is float64
- There are 3 features (columns) that are objects; however, the feature 'rating' should be an int64 or float64, so this should be investigated further.
- Some 'ratings' were not given; therefore, the data type is labeled an object.
- The ratings which were not given either have to be discarded, or the average of the rating for that cuisine or restaurant should be given.

*Missing values in the data*
```javascript
# determine if there are missing values
print('Missing Values')
df.isna().sum()
```
**Observations:**
- There are no missing values

*Statistical summary of the data*
```javascript
# use the describe method to determine the statistical summary of the data
df.describe()
```
**Observations:**
- The statistical summary was completed for all of the quantitative variables
- The order_id and the customer_id are not relevant for the statistical summary as they are identifiers
- The food_preparation_time statistical information:
- average: 27.371970 minutes
- minimum: 20 minutes
- maximum: 35 minutes

```javascript
## how many unique restaurant_names, cuisine_types and looking at other qualitative variables
df.describe(include = 'all')
```
**Observations:**
- There are 178 unique restaurant_names, with Shake Shack (219 orders) being the top restaurant_name
- There are 14 unique cuisine_types, with American (584 orders) being the top cuisine_type
- The most orders are made on the Weekend, with 1351 orders
- The rating should be an integer; however, because 736 orders have a rating of 'Not given' it is an object. And it shows that the majority of customers do not leave a rating.

*Number of orders that are rated / not rated*
```javascript
# determine the number of orders which are not rated
df.groupby('rating').count()
```
**Observations:**
- There are 736 orders which are not rated.

### 3. Exploratory Data Analysis (EDA)
**Univariate Analysis**
