
# Pyber Analysis

* In general, most drivers are in urban areas.

* There are also more rides occured in urban areas. Although 77.8% of total drivers only yield 68.4% of total rides. This means residents of the urban areas enjoy more ride share resources.

* The fares in urban areas are the lowest. The fares in rural areas are not necessarily higher, but they fluctuate immensely.


```python
import pandas as pd
import matplotlib.pyplot as plt

city_df = pd.read_csv("raw_data/city_data.csv")
ride_df = pd.read_csv("raw_data/ride_data.csv")
```


```python
# There are 2 entries of the same city.
city_df["city"].value_counts().head()
```




    Port James     2
    Stewartview    1
    West Peter     1
    Jasonfort      1
    Carrollbury    1
    Name: city, dtype: int64




```python
city_df[city_df["city"] == "Port James"]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>driver_count</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>84</th>
      <td>Port James</td>
      <td>15</td>
      <td>Suburban</td>
    </tr>
    <tr>
      <th>100</th>
      <td>Port James</td>
      <td>3</td>
      <td>Suburban</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Combine the 2 entries into 1.
city_df.iloc[84,1] = city_df.iloc[84,1] + city_df.iloc[100,1]
city_df = city_df.drop(100)
city_df[city_df["city"] == "Port James"]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>driver_count</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>84</th>
      <td>Port James</td>
      <td>18</td>
      <td>Suburban</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = pd.merge(city_df, ride_df, on="city")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>driver_count</th>
      <th>type</th>
      <th>date</th>
      <th>fare</th>
      <th>ride_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Kelseyland</td>
      <td>63</td>
      <td>Urban</td>
      <td>2016-08-19 04:27:52</td>
      <td>5.51</td>
      <td>6246006544795</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Kelseyland</td>
      <td>63</td>
      <td>Urban</td>
      <td>2016-04-17 06:59:50</td>
      <td>5.54</td>
      <td>7466473222333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Kelseyland</td>
      <td>63</td>
      <td>Urban</td>
      <td>2016-05-04 15:06:07</td>
      <td>30.54</td>
      <td>2140501382736</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Kelseyland</td>
      <td>63</td>
      <td>Urban</td>
      <td>2016-01-25 20:44:56</td>
      <td>12.08</td>
      <td>1896987891309</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kelseyland</td>
      <td>63</td>
      <td>Urban</td>
      <td>2016-08-09 18:19:47</td>
      <td>17.91</td>
      <td>8784212854829</td>
    </tr>
  </tbody>
</table>
</div>



## Bubble Plot of Ride Sharing Data


```python
groups = df.groupby(["type", "city"])

n_rides = groups["ride_id"].count()
n_rides1 = n_rides["Urban"]
n_rides2 = n_rides["Suburban"]
n_rides3 = n_rides["Rural"]

avg_fare = groups["fare"].mean()
avg_fare1 = avg_fare["Urban"]
avg_fare2 = avg_fare["Suburban"]
avg_fare3 = avg_fare["Rural"]

n_drivers = groups["driver_count"].mean()
n_drivers1 = n_drivers["Urban"]
n_drivers2 = n_drivers["Suburban"]
n_drivers3 = n_drivers["Rural"]
```


```python
plt.style.use("seaborn")
plt.scatter(n_rides1, avg_fare1, s=n_drivers1*5, c="tomato", alpha=.8, lw=.8, edgecolor="k", label="Urban")
plt.scatter(n_rides2, avg_fare2, s=n_drivers2*5, c="skyblue", alpha=.8, lw=.8, edgecolor="k", label="Suburban")
plt.scatter(n_rides3, avg_fare3, s=n_drivers3*5, c="gold", alpha=.8, lw=.8, edgecolor="k", label="Rural")
plt.title("Pyber Ride Sharing Data (2016)")
plt.xlabel("Total Number of Rides (Per City)")
plt.ylabel("Average Fare ($)")
lgnd = plt.legend(title="City Types")
for handle in lgnd.legendHandles:
    handle.set_sizes([20])
plt.text(37.5, 45, "Note:\nCircle size correlates with driver count per city.", va="top")
plt.show()
```


![png](output_8_0.png)


## Total Fares by City Type


```python
fare_type = df.groupby("type")["fare"].sum()
plt.pie(fare_type, explode=[0,0,0.1], labels=fare_type.index, colors=["gold", "skyblue", "tomato"], autopct="%.1f%%", 
        shadow=True, startangle=140)
plt.title("% of Total Fares by City Type")
plt.show()
```


![png](output_10_0.png)


## Total Rides by City Type


```python
ride_type = df.groupby("type")["ride_id"].count()
plt.pie(ride_type, explode=[0,0,0.1], labels=ride_type.index, colors=["gold", "skyblue", "tomato"], autopct="%.1f%%", 
        shadow=True, startangle=140)
plt.title("% of Total Rides by City Type")
plt.show()
```


![png](output_12_0.png)


## Total Drivers by City Type


```python
driver_type = city_df.groupby("type")["driver_count"].sum()
plt.pie(driver_type, explode=[0,0,0.1], labels=driver_type.index, colors=["gold", "skyblue", "tomato"], autopct="%.1f%%", 
        shadow=True, startangle=140)
plt.title("% of Total Drivers by City Type")
plt.show()
```


![png](output_14_0.png)

