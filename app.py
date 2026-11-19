# Pehle ye chalao
!pip install pandas scikit-learn matplotlib

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset load - Ye link se direct aayega
url = "https://raw.githubusercontent.com/plotly/datasets/master/superstore.csv"
df = pd.read_csv(url)

df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Month'] = df['Order Date'].dt.to_period('M').astype(str)
monthly = df.groupby('Month')['Sales'].sum().reset_index()
monthly['Month_Number'] = range(1, len(monthly)+1)

# Model
X = monthly[['Month_Number']]
y = monthly['Sales']
model = LinearRegression()
model.fit(X, y)

# Agle 3 Mahine ki Prediction
future = [[len(monthly)+1], [len(monthly)+2], [len(monthly)+3]]
print("Agle 3 Mahine ki Predicted Sales:")
print(model.predict(future))

# Graph
plt.figure(figsize=(10,4))
plt.plot(monthly['Month'], monthly['Sales'], marker='o')
plt.xticks(rotation=45)
plt.title("Monthly Sales Trend")
plt.show()