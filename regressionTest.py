import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
#set data
t_base = np.arange(0,200,1)
t = np.reshape(t_base, (len(t_base),1))
straight = np.random.normal(8000, 10, len(t_base))
increase = straight * (1 + t_base * 0.0002)

s_reshaped = np.reshape(straight, (len(straight),1))
inc_reshaped = np.reshape(increase, (len(increase),1))

x_train = t[100:-30]
x_test = t[-30:]

straight_y_train = s_reshaped[100:-30]
straight_y_test = s_reshaped[-30:]


increase_y_train = inc_reshaped[100:-30]
increase_y_test = inc_reshaped[-30:]

#create & train model
straight_regr = linear_model.LinearRegression()
straight_regr.fit(x_train, straight_y_train)
straight_y_pred = straight_regr.predict(x_test)

increase_regr = linear_model.LinearRegression()
increase_regr.fit(x_train, increase_y_train)
increase_y_pred = increase_regr.predict(t)



# print('Mean squared error: %.2f'
#       % mean_squared_error(straight_y_test, straight_y_pred))

#plots
fig, (ax1,ax2) = plt.subplots(1,2)
ax1.plot(t, straight, label = 'Actual Usage')
ax2.plot(t, increase, label = 'Actual Usage')

ax1.plot(x_test, straight_y_pred, color='red', label='Regression Model')
ax2.plot(t, increase_y_pred, color='red', label='Regression Model')

ax1.set(xlabel='days', ylabel='usage (RUR)')

handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')

plt.show()
