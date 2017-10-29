

```python
import numpy as np
import pandas as pd
# from sklearn import 
import matplotlib.pyplot as plt

%matplotlib inline
```


```python
df = pd.DataFrame(np.random.randn(8,4), index=list('abcdefgh'), columns=list('ABCD'))
df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>-0.734848</td>
      <td>1.122709</td>
      <td>1.816192</td>
      <td>1.724869</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-0.396129</td>
      <td>-0.476525</td>
      <td>-0.711366</td>
      <td>0.021546</td>
    </tr>
    <tr>
      <th>c</th>
      <td>1.446226</td>
      <td>1.186130</td>
      <td>0.780496</td>
      <td>0.206887</td>
    </tr>
    <tr>
      <th>d</th>
      <td>0.183712</td>
      <td>1.216086</td>
      <td>1.876283</td>
      <td>-0.438990</td>
    </tr>
    <tr>
      <th>e</th>
      <td>-1.588224</td>
      <td>-0.501879</td>
      <td>0.714124</td>
      <td>-0.570423</td>
    </tr>
    <tr>
      <th>f</th>
      <td>-3.592550</td>
      <td>1.505495</td>
      <td>-0.145490</td>
      <td>-0.035358</td>
    </tr>
    <tr>
      <th>g</th>
      <td>0.896083</td>
      <td>-1.584280</td>
      <td>0.773476</td>
      <td>-0.468209</td>
    </tr>
    <tr>
      <th>h</th>
      <td>1.661386</td>
      <td>-1.917780</td>
      <td>-0.381853</td>
      <td>-0.744607</td>
    </tr>
  </tbody>
</table>
</div>



## Slicing

For DataFrame, [ ] slices the row


```python
df[:2:-1] # df[,] wrong
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>h</th>
      <td>1.661386</td>
      <td>-1.917780</td>
      <td>-0.381853</td>
      <td>-0.744607</td>
    </tr>
    <tr>
      <th>g</th>
      <td>0.896083</td>
      <td>-1.584280</td>
      <td>0.773476</td>
      <td>-0.468209</td>
    </tr>
    <tr>
      <th>f</th>
      <td>-3.592550</td>
      <td>1.505495</td>
      <td>-0.145490</td>
      <td>-0.035358</td>
    </tr>
    <tr>
      <th>e</th>
      <td>-1.588224</td>
      <td>-0.501879</td>
      <td>0.714124</td>
      <td>-0.570423</td>
    </tr>
    <tr>
      <th>d</th>
      <td>0.183712</td>
      <td>1.216086</td>
      <td>1.876283</td>
      <td>-0.438990</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['A'] # df['a'], df['A','C'] wrong
```




    a   -0.734848
    b   -0.396129
    c    1.446226
    d    0.183712
    e   -1.588224
    f   -3.592550
    g    0.896083
    h    1.661386
    Name: A, dtype: float64



## Selection by lable


```python
df.loc['a':'c'] # df.loc['A'] wrong
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>-0.734848</td>
      <td>1.122709</td>
      <td>1.816192</td>
      <td>1.724869</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-0.396129</td>
      <td>-0.476525</td>
      <td>-0.711366</td>
      <td>0.021546</td>
    </tr>
    <tr>
      <th>c</th>
      <td>1.446226</td>
      <td>1.186130</td>
      <td>0.780496</td>
      <td>0.206887</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc[:,'A':'C']
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>-0.734848</td>
      <td>1.122709</td>
      <td>1.816192</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-0.396129</td>
      <td>-0.476525</td>
      <td>-0.711366</td>
    </tr>
    <tr>
      <th>c</th>
      <td>1.446226</td>
      <td>1.186130</td>
      <td>0.780496</td>
    </tr>
    <tr>
      <th>d</th>
      <td>0.183712</td>
      <td>1.216086</td>
      <td>1.876283</td>
    </tr>
    <tr>
      <th>e</th>
      <td>-1.588224</td>
      <td>-0.501879</td>
      <td>0.714124</td>
    </tr>
    <tr>
      <th>f</th>
      <td>-3.592550</td>
      <td>1.505495</td>
      <td>-0.145490</td>
    </tr>
    <tr>
      <th>g</th>
      <td>0.896083</td>
      <td>-1.584280</td>
      <td>0.773476</td>
    </tr>
    <tr>
      <th>h</th>
      <td>1.661386</td>
      <td>-1.917780</td>
      <td>-0.381853</td>
    </tr>
  </tbody>
</table>
</div>



## selection by position


```python
df.iloc[:,1:4:2] #[-1] is to exclude, not reverse order
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>B</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.122709</td>
      <td>1.724869</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-0.476525</td>
      <td>0.021546</td>
    </tr>
    <tr>
      <th>c</th>
      <td>1.186130</td>
      <td>0.206887</td>
    </tr>
    <tr>
      <th>d</th>
      <td>1.216086</td>
      <td>-0.438990</td>
    </tr>
    <tr>
      <th>e</th>
      <td>-0.501879</td>
      <td>-0.570423</td>
    </tr>
    <tr>
      <th>f</th>
      <td>1.505495</td>
      <td>-0.035358</td>
    </tr>
    <tr>
      <th>g</th>
      <td>-1.584280</td>
      <td>-0.468209</td>
    </tr>
    <tr>
      <th>h</th>
      <td>-1.917780</td>
      <td>-0.744607</td>
    </tr>
  </tbody>
</table>
</div>


