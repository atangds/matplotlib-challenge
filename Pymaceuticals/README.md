
# Pymaceuticals Analysis

* Although no drugs could reverse the cancer process, mice taking Capomulin got the least metastatic spread (1.48 vs. 2.11-3.36) by the end of the treatment.

* Mice taking Capomulin also had a significantly higher survival rate (84% vs. 36-44%) by the end of the treatment.

* Capomulin is the only drug that decreased tumor volume (by 19%) by the end of the treatment.

* In conclusion, Capomulin has drastically outperformed the other 3 drugs in the group for this clinical trial.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

trial_df = pd.read_csv("raw_data/clinicaltrial_data.csv")
drug_df = pd.read_csv("raw_data/mouse_drug_data.csv")
```


```python
df = pd.merge(drug_df, trial_df, on="Mouse ID")
df = df[df["Drug"].isin(["Capomulin", "Infubinol", "Ketapril", "Placebo"])]
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
      <th>Mouse ID</th>
      <th>Drug</th>
      <th>Timepoint</th>
      <th>Tumor Volume (mm3)</th>
      <th>Metastatic Sites</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>393</th>
      <td>q119</td>
      <td>Ketapril</td>
      <td>0</td>
      <td>45.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>394</th>
      <td>q119</td>
      <td>Ketapril</td>
      <td>5</td>
      <td>47.864440</td>
      <td>0</td>
    </tr>
    <tr>
      <th>395</th>
      <td>q119</td>
      <td>Ketapril</td>
      <td>10</td>
      <td>51.236606</td>
      <td>0</td>
    </tr>
    <tr>
      <th>396</th>
      <td>n923</td>
      <td>Ketapril</td>
      <td>0</td>
      <td>45.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>397</th>
      <td>n923</td>
      <td>Ketapril</td>
      <td>5</td>
      <td>45.824881</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Tumor Response to Treatment


```python
tumor_avg = df.groupby(["Timepoint", "Drug"])[["Tumor Volume (mm3)"]].mean().unstack()
tumor_avg.columns = ["Capomulin", "Infubinol", "Ketapril", "Placebo"]
tumor_avg
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
      <th>Capomulin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Placebo</th>
    </tr>
    <tr>
      <th>Timepoint</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>44.266086</td>
      <td>47.062001</td>
      <td>47.389175</td>
      <td>47.125589</td>
    </tr>
    <tr>
      <th>10</th>
      <td>43.084291</td>
      <td>49.403909</td>
      <td>49.582269</td>
      <td>49.423329</td>
    </tr>
    <tr>
      <th>15</th>
      <td>42.064317</td>
      <td>51.296397</td>
      <td>52.399974</td>
      <td>51.359742</td>
    </tr>
    <tr>
      <th>20</th>
      <td>40.716325</td>
      <td>53.197691</td>
      <td>54.920935</td>
      <td>54.364417</td>
    </tr>
    <tr>
      <th>25</th>
      <td>39.939528</td>
      <td>55.715252</td>
      <td>57.678982</td>
      <td>57.482574</td>
    </tr>
    <tr>
      <th>30</th>
      <td>38.769339</td>
      <td>58.299397</td>
      <td>60.994507</td>
      <td>59.809063</td>
    </tr>
    <tr>
      <th>35</th>
      <td>37.816839</td>
      <td>60.742461</td>
      <td>63.371686</td>
      <td>62.420615</td>
    </tr>
    <tr>
      <th>40</th>
      <td>36.958001</td>
      <td>63.162824</td>
      <td>66.068580</td>
      <td>65.052675</td>
    </tr>
    <tr>
      <th>45</th>
      <td>36.236114</td>
      <td>65.755562</td>
      <td>70.662958</td>
      <td>68.084082</td>
    </tr>
  </tbody>
</table>
</div>




```python
tumor_sem = df.groupby(["Timepoint", "Drug"])[["Tumor Volume (mm3)"]].sem().unstack()
tumor_sem.columns = ["Capomulin", "Infubinol", "Ketapril", "Placebo"]
tumor_sem
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
      <th>Capomulin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Placebo</th>
    </tr>
    <tr>
      <th>Timepoint</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.448593</td>
      <td>0.235102</td>
      <td>0.264819</td>
      <td>0.218091</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.702684</td>
      <td>0.282346</td>
      <td>0.357421</td>
      <td>0.402064</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.838617</td>
      <td>0.357705</td>
      <td>0.580268</td>
      <td>0.614461</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.909731</td>
      <td>0.476210</td>
      <td>0.726484</td>
      <td>0.839609</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.881642</td>
      <td>0.550315</td>
      <td>0.755413</td>
      <td>1.034872</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.934460</td>
      <td>0.631061</td>
      <td>0.934121</td>
      <td>1.218231</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1.052241</td>
      <td>0.984155</td>
      <td>1.127867</td>
      <td>1.287481</td>
    </tr>
    <tr>
      <th>40</th>
      <td>1.223608</td>
      <td>1.055220</td>
      <td>1.158449</td>
      <td>1.370634</td>
    </tr>
    <tr>
      <th>45</th>
      <td>1.223977</td>
      <td>1.144427</td>
      <td>1.453186</td>
      <td>1.351726</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(10,7.5))
x = range(0,50,5)
ax.errorbar(x, tumor_avg["Capomulin"], tumor_sem["Capomulin"], fmt="ro--", ms=8, dashes=(20,20), elinewidth=.5, capsize=5, 
            lw=.5, mec="k", mew=.5)
ax.errorbar(x, tumor_avg["Infubinol"], tumor_sem["Infubinol"], fmt="b^--", ms=8, dashes=(20,20), elinewidth=.5, capsize=5, 
            lw=.5, mec="k", mew=.5)
ax.errorbar(x, tumor_avg["Ketapril"], tumor_sem["Ketapril"], fmt="gs--", ms=8, dashes=(20,20), elinewidth=.5, capsize=5, 
            lw=.5, mec="k", mew=.5)
ax.errorbar(x, tumor_avg["Placebo"], tumor_sem["Placebo"], fmt="kd--", ms=8, dashes=(20,20), elinewidth=.5, capsize=5, 
            lw=.5, mec="k", mew=.5)
ax.set_title("Tumor Response to Treatment", size=20)
ax.set_xlabel("Time (Days)", size=15)
ax.set_ylabel("Tumor Volume (mm3)", size=15)
ax.get_xaxis().set_tick_params(direction="in", length=8, labelsize=15, top=True)
ax.get_yaxis().set_tick_params(direction="in", length=8, labelsize=15, right=True)
for _ in ["top", "bottom", "left", "right"]:
    ax.spines[_].set_linewidth(2)
ax.set_xlim(0,45)
ax.set_yticks(range(30,90,10))
ax.grid(c="k", ls=":", dashes=(2,5))
ax.legend(fontsize=15, numpoints=2)
plt.show()
```


![png](output_6_0.png)


## Metastatic Response to Treatment


```python
meta_avg = df.groupby(["Timepoint", "Drug"])[["Metastatic Sites"]].mean().unstack()
meta_avg.columns = ["Capomulin", "Infubinol", "Ketapril", "Placebo"]
meta_avg
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
      <th>Capomulin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Placebo</th>
    </tr>
    <tr>
      <th>Timepoint</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.160000</td>
      <td>0.280000</td>
      <td>0.304348</td>
      <td>0.375000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.320000</td>
      <td>0.666667</td>
      <td>0.590909</td>
      <td>0.833333</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.375000</td>
      <td>0.904762</td>
      <td>0.842105</td>
      <td>1.250000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.652174</td>
      <td>1.050000</td>
      <td>1.210526</td>
      <td>1.526316</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.818182</td>
      <td>1.277778</td>
      <td>1.631579</td>
      <td>1.941176</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1.090909</td>
      <td>1.588235</td>
      <td>2.055556</td>
      <td>2.266667</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1.181818</td>
      <td>1.666667</td>
      <td>2.294118</td>
      <td>2.642857</td>
    </tr>
    <tr>
      <th>40</th>
      <td>1.380952</td>
      <td>2.100000</td>
      <td>2.733333</td>
      <td>3.166667</td>
    </tr>
    <tr>
      <th>45</th>
      <td>1.476190</td>
      <td>2.111111</td>
      <td>3.363636</td>
      <td>3.272727</td>
    </tr>
  </tbody>
</table>
</div>




```python
meta_sem = df.groupby(["Timepoint", "Drug"])[["Metastatic Sites"]].sem().unstack()
meta_sem.columns = ["Capomulin", "Infubinol", "Ketapril", "Placebo"]
meta_sem
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
      <th>Capomulin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Placebo</th>
    </tr>
    <tr>
      <th>Timepoint</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.074833</td>
      <td>0.091652</td>
      <td>0.098100</td>
      <td>0.100947</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.125433</td>
      <td>0.159364</td>
      <td>0.142018</td>
      <td>0.115261</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.132048</td>
      <td>0.194015</td>
      <td>0.191381</td>
      <td>0.190221</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.161621</td>
      <td>0.234801</td>
      <td>0.236680</td>
      <td>0.234064</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.181818</td>
      <td>0.265753</td>
      <td>0.288275</td>
      <td>0.263888</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.172944</td>
      <td>0.227823</td>
      <td>0.347467</td>
      <td>0.300264</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0.169496</td>
      <td>0.224733</td>
      <td>0.361418</td>
      <td>0.341412</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0.175610</td>
      <td>0.314466</td>
      <td>0.315725</td>
      <td>0.297294</td>
    </tr>
    <tr>
      <th>45</th>
      <td>0.202591</td>
      <td>0.309320</td>
      <td>0.278722</td>
      <td>0.304240</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(10,7.5))
x = range(0,50,5)
ax.errorbar(x, meta_avg["Capomulin"], meta_sem["Capomulin"], fmt="ro--", ms=8, dashes=(20,20), elinewidth=.5, capsize=5, 
            lw=.5, mec="k", mew=.5)
ax.errorbar(x, meta_avg["Infubinol"], meta_sem["Infubinol"], fmt="b^--", ms=8, dashes=(20,20), elinewidth=.5, capsize=5, 
            lw=.5, mec="k", mew=.5)
ax.errorbar(x, meta_avg["Ketapril"], meta_sem["Ketapril"], fmt="gs--", ms=8, dashes=(20,20), elinewidth=.5, capsize=5, 
            lw=.5, mec="k", mew=.5)
ax.errorbar(x, meta_avg["Placebo"], meta_sem["Placebo"], fmt="kd--", ms=8, dashes=(20,20), elinewidth=.5, capsize=5, 
            lw=.5, mec="k", mew=.5)
ax.set_title("Metastatic Spread During Treatment", size=20)
ax.set_xlabel("Time (Days)", size=15)
ax.set_ylabel("Metastatic Sites", size=15)
ax.get_xaxis().set_tick_params(direction="in", length=8, labelsize=15, top=True)
ax.get_yaxis().set_tick_params(direction="in", length=8, labelsize=15, right=True)
for _ in ["top", "bottom", "left", "right"]:
    ax.spines[_].set_linewidth(2)
ax.set_xlim(0,45)
ax.set_ylim(ymin=0)
ax.set_yticks(np.arange(0,4.5,.5))
ax.grid(c="k", ls=":", dashes=(2,5))
ax.legend(fontsize=15, numpoints=2)
plt.show()
```


![png](output_10_0.png)


## Survival Rates


```python
live_ct = df.groupby(["Timepoint", "Drug"])[["Mouse ID"]].count().unstack()
live_ct.columns = ["Capomulin", "Infubinol", "Ketapril", "Placebo"]
live_rt = pd.DataFrame([live_ct.iloc[:,_]/25*100 for _ in range(4)]).T
live_rt
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
      <th>Capomulin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Placebo</th>
    </tr>
    <tr>
      <th>Timepoint</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>100.0</td>
      <td>100.0</td>
      <td>92.0</td>
      <td>96.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>100.0</td>
      <td>84.0</td>
      <td>88.0</td>
      <td>96.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>96.0</td>
      <td>84.0</td>
      <td>76.0</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>92.0</td>
      <td>80.0</td>
      <td>76.0</td>
      <td>76.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>88.0</td>
      <td>72.0</td>
      <td>76.0</td>
      <td>68.0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>88.0</td>
      <td>68.0</td>
      <td>72.0</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>88.0</td>
      <td>48.0</td>
      <td>68.0</td>
      <td>56.0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>84.0</td>
      <td>40.0</td>
      <td>60.0</td>
      <td>48.0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>84.0</td>
      <td>36.0</td>
      <td>44.0</td>
      <td>44.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(10,7.5))
x = range(0,50,5)
ax.errorbar(x, live_rt["Capomulin"], fmt="ro--", ms=8, dashes=(20,20), elinewidth=.5, capsize=5, lw=.5, mec="k", mew=.5)
ax.errorbar(x, live_rt["Infubinol"], fmt="b^--", ms=8, dashes=(20,20), elinewidth=.5, capsize=5, lw=.5, mec="k", mew=.5)
ax.errorbar(x, live_rt["Ketapril"], fmt="gs--", ms=8, dashes=(20,20), elinewidth=.5, capsize=5, lw=.5, mec="k", mew=.5)
ax.errorbar(x, live_rt["Placebo"], fmt="kd--", ms=8, dashes=(20,20), elinewidth=.5, capsize=5, lw=.5, mec="k", mew=.5)
ax.set_title("Survival During Treatment", size=20)
ax.set_xlabel("Time (Days)", size=15)
ax.set_ylabel("Survival Rate (%)", size=15)
ax.get_xaxis().set_tick_params(direction="in", length=8, labelsize=15, top=True)
ax.get_yaxis().set_tick_params(direction="in", length=8, labelsize=15, right=True)
for _ in ["top", "bottom", "left", "right"]:
    ax.spines[_].set_linewidth(2)
ax.set_xlim(0,45)
ax.set_ylim(30,100)
ax.grid(c="k", ls=":", dashes=(2,5))
ax.legend(fontsize=15, numpoints=2)
plt.show()
```


![png](output_13_0.png)


## Summary Bar Graph


```python
tumor_change = [(tumor_avg.iloc[-1,_]-tumor_avg.iloc[0,_])/tumor_avg.iloc[0,_]*100 for _ in range(4)]
tumor_change
```




    [-19.475302667894173, 46.12347172785187, 57.028794686606076, 51.29796048315153]




```python
fig, ax = plt.subplots(figsize=(10,7.5))
plt.bar(range(4), tumor_change, width=1, color=["g","r","r","r"], ec="k", lw=2)
plt.title("Tumor Change Over 45-Day Treatment", size=20)
plt.ylabel("Tumor Volume Change (%)", size=15)
ax.get_xaxis().set_tick_params(direction="in", length=8, labelsize=15, top=True)
ax.get_yaxis().set_tick_params(direction="in", length=8, labelsize=15, right=True)
for _ in ["top", "bottom", "left", "right"]:
    ax.spines[_].set_linewidth(2)
plt.xlim(-0.5,3.5)
plt.ylim(-20,60)
plt.xticks(range(4), tumor_avg.columns.values)
plt.yticks(range(-20,80,20))
for x, y in zip(range(4), [-5,2,2,2]):
    plt.text(x, y, f"{int(tumor_change[x])}%", color="w", fontsize=15, ha="center")
plt.grid(c="k", ls=":", dashes=(2,5))
plt.show()
```


![png](output_16_0.png)

