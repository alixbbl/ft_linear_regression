import matplotlib.pyplot as plt

x=dataframe['km']
y=dataframe['price']
plt.xlabel('km')
plt.ylabel('price')
plt.plot(x, label='km')
plt.plot(y, label='price')
plt.show()