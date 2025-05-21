# %%
import numpy as np
import plotly.express as px 

import plotly.graph_objects as go
from function import distans_pythagoras , gaussian
# %%
# درست کردن دیتا ست ۲ بعدی از روی توضیع دو جمله‌ای
data = np.random.rand(10, 2)
data = np.array([[2, 3],
                 [2, 4],
                 [1, 3],
                 [1, 4],
                 [5, 4],
                 [7, 9],
                 [9, 7],
                 [5, 1],
                 [3, 2],
                 [8, 9],
                 ])
data.shape
# %%

# plotting the scatter chart
fig = px.scatter(data) 

# showing the plot
fig.show()

# %%
delta = 0.07

means = np.array([])
for i in range(len(data)):
    print(i)
    print(data[i])
    mean_old = data[i]
    distanse = 1000
    while distanse > delta:
        soorat = 0
        makhraj = 0
        for j in range(len(data[0])):
            soorat += gaussian((data[j]), mean_old, 100) * data[j]
            makhraj += gaussian((data[j]), mean_old, 100)
        mean = soorat / makhraj
        distanse = distans_pythagoras(mean , mean_old)
        mean_old = mean
        print("dis : ",distanse)
    
    print("old : ", mean_old)
    means = np.append(means, np.array(mean_old))

means = means.reshape(10, len(means)//10)

# %%
fig = px.scatter(x = data.T[0], y = data.T[1]) 
fig.show()
# %%
fig = px.scatter(x = means.T[0], y = means.T[1],) 
fig.show()

# %%
radius = 0.1

fig = go.Figure()

# 1) نقاط اصلی داده
fig.add_trace(go.Scatter(
    x=data[:, 0],
    y=data[:, 1],
    mode='markers',
    marker=dict(size=6, color='blue'),
    name='Data'
))

# 2) نقاط مرکزهای نهایی
fig.add_trace(go.Scatter(
    x=means[:, 0],
    y=means[:, 1],
    mode='markers',
    marker=dict(size=8, color='red'),
    name='Means'
))

# 3) دایره‌های شفاف اطرافِ هر مرکز
for (mx, my) in means:
    fig.add_shape(
        type="circle",
        xref="x", yref="y",
        x0=mx - radius, y0=my - radius,
        x1=mx + radius, y1=my + radius,
        line=dict(color="rgba(200,0,0,0.5)"),
        fillcolor="rgba(200,0,0,0.2)",
    )

fig.update_layout(
    width=600, height=600,
    title="Data Points & Mean-Shift Centers",
    xaxis_title="X",
    yaxis_title="Y",
    showlegend=True
)

fig.show()
# %%
