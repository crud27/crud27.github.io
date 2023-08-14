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
- Create a profile of the leads who are likely to convert

### 2. Data Analysis

Utilized Python to analyze the data set

Data Overview
- Observations
- Sanity checks
*View the first and last five rows of the data*
```javascript
#check that the data is loaded and look at the data frame
df.head()
```
```javascript
# check the data last five rows
df.tail()
```

*Rows and columns are present in the data* - 
```javascript
# use the shape method to determine the number of rows and columns of the data
df.shape
```
**Observations:**
(4612, 15)

*Datatypes of the different columns in the dataset*
```javascript
# Use info() to print a concise summary of the DataFrame
df.info()
```
```javascript
# find number of unique IDs
df.ID.nunique()
```
**Observations:**
4612
- There are 4612 unique IDs. Each row is a unique ID, therefore, this column doesn't add value and can be dropped
```javascript
# drop the ID column
df = df.drop(['ID'], axis = 1)
```
```javascript
# look at the first five rows again
df.head()
```
```javascript
df.shape
```
**Observations:**
(4612, 14)

```javascript
df.info()
```
**Observations**
- There are 4612 rows and 14 columns
- There is 1 float types (page_views_per_visit)
- There are 4 integer types (age, website_visits, time_spent_on_website,and status(0 not converted or 1 converted)
- There are 9 object types
- There are no null values in the data
- There are 4612 unique IDs. Each row is a unique ID therefore this column did not add value and was dropped

**START HERE WITH EDA**
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

**Summary of all data columns**
```javascript
# get a summary of the data
df.describe(include = 'all').T
```

**Identify the percentage and count of each group within the categorical variables**
**Percentage**

```javascript
# Making a list of all categorical variables
cat_col = list(df.select_dtypes("object").columns)

# Printing percnetage of each unique value in each categorical column
for column in cat_col:
    print(df[column].value_counts(normalize = True))
    print("-" * 50)
```

**Count of each group within the categorical variables**
```javascript
# Making a list of all categorical variables
cat_col = list(df.select_dtypes("object").columns)

# Printing count of each unique value in each categorical column
for column in cat_col:
    print(df[column].value_counts(normalize = False))
    print("-" * 50)

```

### Observations
#### Categorical Data
- <code>current_occupation</code> - there are 3 responses allowed 
-- **Professional is the top with 2616 (57%)**
- <code>first_interaction</code> - there are 2 responses allowed
-- **Website is the top with 2542 (55%)**
- <code>profile_completed</code> - there are 3 possible 
-- **High is the top with 2264 (49%)**
- <code>last_activity</code> -  there are 3 responses
-- **Email Activity is the top with 2278 (49%)**
- <code>print_media_type1</code> - there are 2 responses
-- **No is the top with 4115 (89%) (not seen the Newspaper Ad)**
- <code>print_media_type2</code> - there are 2 responses
-- **No is the top with 4379 (94%) (not seen the Magazine Ad)**
- <code>digital_media</code> - there are 2 responses
-- **No is the top with 4085 (89%) (not seen an ad on the digital platforms)**
- <code>educational_channels</code> - there are 2 responses
-- **No is the top with 3907(85%) (not heard of ExtraaLearn via online forums, educ websites or discussion threads...)**
- <code>referral</code> - there are 2 responses
-- **No is the top with 4519 (98%) (not referred)**
#### Numeric Data
- <code>age</code> The mean age is 46.2 with the range from 18 to 63 years.
- <code>website_visits</code> The mean number of website visits are 3.566782 with a range from 0 to 30 visits.  75% of the visitors visit up to 5 times. An outlier may be present.
- <code>time_spent_on_website</code> The mean time spent was 743.828683 with a range from 0 to 2537. 75% of the visitors spent only up to 1336.75 
- <code>page_views_per_visit</code> The mean page views were 3.026126 pages with a range from 0 to 18.434.  75% of the viewers viewed up to 3.75625 pages. An outlier may be present.


**Status**
**How many of the visitors are converted to customers**

```javascript
#create a bar chart to determine the number of visitors which are converted to customers (1)
plt.figure(figsize = (10, 6))

ax = sns.countplot(x = 'status', data = df)

# Place the exact count on the top of the bar for each category using annotate
for p in ax.patches:
    ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x(), p.get_height()))
```

**Observation Summary**
The majority of the visitors are professionals (56.7%). The website (55%) is the primary first interaction with ExtraaLearn. Only 2% of the visitors have a low completion of the profile. Email (49%) is the highest last activity of the visitor. Very few visitors have had interactions with ExtraaLearn through advertisements or referrals seen the newspaper ads (10%), magazine ads (5%), digital media (11%), educational channels (15%) or have been referred (2%).

Only **1377 (approx 30%**) from a total of 4612 visitors are converted to customers.

**Data Preprocessing**
- Missing value treatment (if needed)
- Feature engineering (if needed)
- Outlier detection and treatment (if needed)
- Preparing data for modeling
- Any other preprocessing steps (if needed)

**Distributions and Outliers**
**Create count plots to identify the distribution of the data.**
**Create box plots to determine if there are any outliers in the data.**

```javascript
#create countplots and box plots to visualize data to identify the distribution and outliers

for col in ['age', 'website_visits', 'time_spent_on_website', 'page_views_per_visit']:
    print(col)
    
    print('The skew is :',round(df[col].skew(), 2))
    
    plt.figure(figsize = (20, 4))
# histogram    
    plt.subplot(1, 2, 1)
    df[col].hist(bins = 10, grid = False)
    plt.ylabel('count')
#box plot    
    plt.subplot(1, 2, 2)
    sns.boxplot(df[col])
   
    plt.show()
```

### Observations 
<code>age</code>
There is a negative skew (-0.72) with most visitors approx 55 years of age.  
<code>website_visits</code>
There is positive skew (2.16) with highest frequency visiting from 0 to 5 times decreasing from there. **The box plot shows outliers.**
<code>time_spent_on_website</code>
There is a positive skew (0.95) with the highest frequency visitors spending between 0 and 250 on the site.  
<code>page_views_per_visit</code>
There is a positive skew (1.27) with the highest frequency of page views between 2.5 and 5.  **The box plot shows outliers.**

### Identifying outliers
- defined a function <code>find_outliers_IQR</code> which takes in a dataframe as an input and returns a dataframe as an output.  The returned data frame contains the outliers as numerical values and others as NaN
- identified the outliers: lower limit < q1 - 1.5 * IQR and higher limit > q3 + 1.5 * IQR

```javascript
# defining the definition to identify outliers 
def find_outliers_IQR(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[((df<(Q1 - 1.5*IQR)) | (df>(Q3 + 1.5*IQR)))]
    return outliers
```

```javascript
#identifying the outliers for the website_visits
outliers = find_outliers_IQR(df['website_visits'])
print('website_visits number of outliers: ' + str(outliers.count()))
print('website_visits min outlier value: ' + str(outliers.min()))
print('website_visits max outlier value: ' + str(outliers.max()))
outliers
```

```javascript
#identifying the outliers for the page_views_per_visit
outliers = find_outliers_IQR(df['page_views_per_visit'])
print('page_views_per_visit number of outliers: ' + str(outliers.count()))
print('page_views_per_visit min outlier value: ' + str(outliers.min()))
print('page_views_per_visit max outlier value: ' + str(outliers.max()))
outliers
```

## Bivariate Data Analysis
- Will continue exploring the data to identify any relationships between the various variables. 

```javascript
# create a pair plot to see if there are any relationships 
# distingueished the paiplot by adding the status as an additional parameter for the pairplot
sns.pairplot(df, hue ='status')
```

### Pairplot Summary
Pairplot visualizes given data to find the relationship between them where the variables can be continuous or categorical.
The color represents if the visitor was converted to a customer or not.  Orange shows those that are now customers.

- Older visitors are more likely to become customers.
- Those visitors that spend more time on the website are more likely to become customers. 
There were no other major trends identified by the pairplot.

### <code>age</code>
