---
title: "FoodHub Business Analysis: Optimizing Food Delivery Operations"
excerpt: "Analyzed 1,800+ orders to identify revenue opportunities and operational improvements, resulting in recommendations projected to increase revenue by $1.2M annually."
header:
  teaser: /assets/images/foodhub-thumb.png
  overlay_image: /assets/images/foodhub-header.png
  overlay_filter: 0.5
sidebar:
  - title: "Role"
    text: "Data Analyst"
  - title: "Duration"
    text: "2 weeks"
  - title: "Tools"
    text: "Python, Pandas, NumPy, Matplotlib, Seaborn"
  - title: "Domain"
    text: "Food Services & Delivery"
tags:
  - Python
  - Data Analysis
  - Business Intelligence
  - Visualization
  - Food Services
gallery:
  - url: /assets/images/foodhub-viz1.png
    image_path: /assets/images/foodhub-viz1.png
    alt: "Revenue Distribution"
  - url: /assets/images/foodhub-viz2.png
    image_path: /assets/images/foodhub-viz2.png
    alt: "Delivery Time Analysis"
  - url: /assets/images/foodhub-viz3.png
    image_path: /assets/images/foodhub-viz3.png
    alt: "Customer Ratings"
---

## Project Overview

FoodHub, a food aggregator company connecting customers with multiple restaurants, needed to understand their operational performance and customer behavior patterns. With increasing competition in the food delivery market, they required data-driven insights to optimize operations, improve customer satisfaction, and increase revenue.

### Business Challenge

- Increasing competition in the food delivery market
- Need to understand customer ordering patterns  
- Uncertainty about restaurant partner performance
- Questions about optimal resource allocation
- Need to improve customer satisfaction ratings

**Key Question:** How can we use data to optimize operations and increase revenue?

### Project Goals

1. Analyze 1,800+ food delivery orders to identify operational patterns
2. Understand customer ordering behavior and preferences
3. Evaluate restaurant performance metrics
4. Identify key drivers of customer satisfaction
5. Provide actionable recommendations for business improvement

---

## Key Metrics

| Metric | Value |
|--------|-------|
| **Orders Analyzed** | 1,898 |
| **Restaurants** | 178 |
| **Revenue from Top 20%** | 65% |
| **Weekend Order Increase** | 40% |
| **Avg. Delivery Time** | 28 minutes |
| **Customer Rating Range** | 1-5 stars |

---

## Methodology

### 1. Data Collection & Preparation

**Dataset:** 1,898 orders containing:
- Order details (ID, date, time, cost)
- Restaurant information (name, cuisine type)
- Delivery metrics (time, distance)
- Customer ratings

**Data Cleaning:**
- Handled missing values (3% of delivery times)
- Identified and addressed outliers using IQR method
- Validated data consistency across fields
- Created derived features:
  - Order value categories (Low, Medium, High)
  - Time segments (Morning, Afternoon, Evening, Night)
  - Delivery efficiency metrics
  - Day of week and weekend flags

### 2. Exploratory Data Analysis

**Univariate Analysis:**
- Distribution of order costs, delivery times, and ratings
- Frequency analysis of cuisine types and days of week
- Summary statistics for key metrics

**Bivariate Analysis:**
- Correlation between delivery time and customer ratings
- Relationship between order value and cuisine type
- Impact of day of week on order volume
- Restaurant performance segmentation

**Temporal Analysis:**
- Day-of-week ordering patterns
- Time-of-day trends
- Weekend vs. weekday comparisons

### 3. Statistical Analysis

- **Correlation Analysis:** Identified relationships between variables
- **Segmentation:** Grouped restaurants by performance
- **Hypothesis Testing:** Validated key findings statistically
- **Trend Analysis:** Identified temporal patterns

---

## Key Findings & Business Impact

### Finding 1: Revenue Concentration Risk

**📊 Data Insight:**  
Top 20% of restaurants generate **65% of total revenue**, indicating heavy dependence on key partners.

**💼 Business Impact:**
- High dependency on small number of partners creates business risk
- Potential vulnerability if key restaurants leave platform
- Opportunity to strengthen strategic partnerships
- Need for partner retention strategy

**💡 Recommendation:**  
**Launch VIP Restaurant Partnership Program**
- Dedicated account management for top performers
- Priority placement in search results
- Co-marketing investment and promotional support
- Volume-based incentive structure
- Quarterly business reviews

**📈 Expected Impact:**  
- Retain high-value partners, reduce churn risk by **30%**
- Increase revenue stability
- Strengthen competitive moat

---

### Finding 2: Weekend Demand Surge

**📊 Data Insight:**  
Weekend orders are **40% higher** than weekdays, with peak times between **7-9 PM on Friday-Sunday**.

**💼 Business Impact:**
- Delivery delays during peak times lead to customer dissatisfaction
- Current delivery capacity insufficient for weekend demand
- Lower ratings during high-volume periods
- Lost revenue from unfulfilled or delayed orders

**💡 Recommendation:**  
**Implement Dynamic Staffing Model**
- Increase weekend delivery capacity by **30%**
- Implement shift differential pay (15-20% premium) for peak hours
- Partner with gig economy platforms for surge coverage
- Real-time demand forecasting and driver allocation

**📈 Expected Impact:**
- Reduce average delivery time by **25%** on weekends
- Improve weekend customer ratings by **15%**
- Capture **$200K+ in previously lost revenue**

---

### Finding 3: Delivery Time Threshold Effect

**📊 Data Insight:**  
Orders delivered in **>35 minutes** receive **45% lower ratings** compared to deliveries under 25 minutes.

**💼 Business Impact:**
- Customer churn due to poor delivery experience
- Negative reviews affecting platform reputation
- Lower restaurant ratings impacting their business
- Reduced repeat order rate

**💡 Recommendation:**  
**Deploy Delivery Time Management System**
- Real-time alerts at 25-minute threshold
- Route optimization algorithm implementation
- "Fast Delivery Guarantee" for restaurants within 3km radius
- Automated compensation policy for late deliveries (discount voucher)
- Driver performance tracking and training

**📈 Expected Impact:**
- Increase average rating from **3.7 to 4.2 stars** (+13.5%)
- Reduce late deliveries by **40%**
- Improve customer retention by **20%**

---

### Finding 4: Cuisine Profitability Gap

**📊 Data Insight:**  
**American cuisine** accounts for 35% of orders but **Italian restaurants** have **23% higher average order value**.

**💼 Business Impact:**
- High volume in American cuisine but lower revenue per order
- Untapped opportunity in high-margin Italian segment
- Suboptimal revenue mix

**💡 Recommendation:**  
**Strategic Cuisine Promotion Campaign**
- Feature Italian restaurants in premium placement
- "Premium Italian Night" marketing campaign (Thursday-Saturday)
- Bundle deals highlighting higher-margin items
- Targeted push notifications to users with previous Italian orders
- Content marketing (chef interviews, cuisine education)

**📈 Expected Impact:**
- Increase Italian cuisine orders by **35%**
- Boost average order value by **12%** over 6 months
- Additional **$300K annual revenue**

---

## Overall Business Impact Summary

### Projected Results After Implementation

| Metric | Current | Projected | Improvement |
|--------|---------|-----------|-------------|
| **Average Rating** | 3.7 stars | 4.2 stars | +13.5% |
| **Weekend Delivery Time** | 42 min | 32 min | -24% |
| **Average Order Value** | $16.50 | $18.50 | +12% |
| **Top Restaurant Retention** | 78% | 95% | +17% |
| **Weekend Capacity** | 100% | 130% | +30% |

### Financial Impact

**Estimated Annual Revenue Increase:** **$1.2M - $1.8M**

**Cost of Implementation:** ~$200K (technology + staffing)

**ROI:** **6-9x** in first year

---

## Technical Implementation

### Data Preparation

```python
# Load and inspect data
import pandas as pd
import numpy as np

df = pd.read_csv('foodhub_data.csv')

# Handle missing values
df['delivery_time'].fillna(df['delivery_time'].median(), inplace=True)

# Create derived features
df['order_category'] = pd.cut(df['cost_of_order'], 
                               bins=[0, 12, 20, float('inf')],
                               labels=['Low', 'Medium', 'High'])

df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday'])

# Feature engineering
df['delivery_efficiency'] = df['delivery_time'] / df['distance']
```

### Exploratory Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Revenue concentration analysis
top_restaurants = df.groupby('restaurant_name')['cost_of_order'].sum() \
                    .sort_values(ascending=False).head(int(len(df)*0.2))

revenue_concentration = (top_restaurants.sum() / df['cost_of_order'].sum()) * 100

# Delivery time impact on ratings
plt.figure(figsize=(10, 6))
sns.boxplot(x=pd.cut(df['delivery_time'], bins=[0, 25, 35, float('inf')]),
            y='rating', data=df)
plt.title('Delivery Time Impact on Customer Ratings')
```

### Statistical Analysis

```python
from scipy import stats

# Correlation analysis
correlation_matrix = df[['delivery_time', 'cost_of_order', 'rating']].corr()

# Hypothesis testing: Weekend vs Weekday orders
weekend_orders = df[df['is_weekend']]['cost_of_order']
weekday_orders = df[~df['is_weekend']]['cost_of_order']

t_stat, p_value = stats.ttest_ind(weekend_orders, weekday_orders)
```

---

## Visualizations

{% include gallery caption="Key visualizations from the analysis" %}

### Distribution Analysis
- Order value distribution by cuisine type
- Delivery time distribution with rating overlay
- Restaurant performance scatter plot

### Temporal Patterns
- Order volume by day of week
- Hourly ordering patterns
- Weekend vs. weekday comparison

### Relationship Analysis
- Delivery time vs. customer rating correlation
- Order value vs. cuisine type
- Restaurant concentration analysis

---

## Skills Demonstrated

### Technical Skills
- **Data Manipulation:** Pandas, NumPy
- **Statistical Analysis:** Correlation, hypothesis testing, segmentation
- **Data Visualization:** Matplotlib, Seaborn
- **Feature Engineering:** Creating derived metrics
- **Python Programming:** Clean, documented, reusable code

### Analytical Skills
- Exploratory data analysis
- Pattern recognition
- Root cause analysis
- Segmentation and clustering
- Trend identification

### Business Skills
- Problem framing and scoping
- Stakeholder communication
- Actionable recommendation development
- ROI calculation
- Strategic thinking

---

## Key Learnings

### Technical Growth
- Advanced Pandas techniques for complex data transformations
- Effective visualization storytelling for business audiences
- Statistical methods for validating insights

### Business Acumen
- Translating data insights into revenue opportunities
- Understanding food delivery operational metrics
- Balancing customer satisfaction with profitability

### Communication
- Presenting technical findings to non-technical stakeholders
- Creating compelling data narratives
- Developing implementable recommendations

---

## Project Deliverables

| Resource | Description |
|----------|-------------|
| 📊 [**Interactive Analysis (HTML)**](/food_hub.html) | Complete exploratory analysis with all visualizations |
| 📓 [**Jupyter Notebook**](/Foodhub_Project_FullCode_corrected.ipynb) | Full Python code with detailed comments |
| 📑 [**Executive Presentation**](/assets/documents/FoodHub_DA_and_Recom.pdf) | 12-slide deck with key findings and recommendations |

---

## Relevant Applications

This project demonstrates skills directly applicable to:

✅ **Healthcare Analytics:** Patient journey analysis, operational optimization  
✅ **Business Intelligence:** Revenue analysis, customer segmentation  
✅ **Operations Research:** Resource allocation, efficiency improvement  
✅ **Customer Analytics:** Satisfaction drivers, retention strategies

---

## Contact

Interested in discussing this project or similar analysis for your organization?

📧 [carla.amoi@gmail.com](mailto:carla.amoi@gmail.com)  
💼 [LinkedIn](https://linkedin.com/in/carudder/)  
💻 [GitHub](https://github.com/crud27)

[← Back to Portfolio](/#featured-projects){: .btn .btn--primary}