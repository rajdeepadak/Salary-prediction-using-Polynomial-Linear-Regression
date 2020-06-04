# Import all libraries---------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# -----------------------------


# Construct data set, IDV, DV-----------------
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2]
y = dataset.iloc[:, -1]
# --------------------------------------------

# Visualize how y varies with x--
plt.scatter(x, y, color = 'red')
plt.plot(x, y, color = 'green')
plt.xlabel('Position')
plt.ylabel('Salaries')
plt.title('Position vs Salaries')
plt.show()
# -------------------------------


# Construct SLR regressor for x and find y_pred using just SLR
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)
y_pred = lin_reg.predict(x)
# ------------------------------------------------------------


# Visualise SLR between x vs y ------------------------------------------------------------
# Dissatisfactory results from SLR
# Using the correlation coefficient it can be determined if the data will suit an SLR model
# print(np.corrcoef(x,y)[0, 1]*100, "%") gives 81%, suggesting that an SLR will be unfit

# Calculate correlation coefficient
x1 = dataset['Level'].values.tolist()
x1 = np.array(x1)
y1 = np.array(y)
    
r1 = np.corrcoef(x1, y)[0, 1]     # 81.79 %

"""
    x = dataset['Level'].values.tolist()
    x = np.array(x)
    y = np.array(y)
    
    print(np.corrcoef(x, y[0, 1]))   # Since graph is non linear r != 1
    print(np.corrcoef(x, y_pred)[0, 1]) # As the SLR line is linear r = 0.99999 i.e. 1
"""

plt.scatter(x, y, color = 'red')
plt.plot(x, y, color = 'green')
plt.plot(x, y_pred, color = 'blue')
plt.xlabel('Position')
plt.ylabel('Salaries')
plt.title('Position vs Salaries SLR model')
plt.show()
# ------------------------------------------------------------------------------------------


"""
    Construct a PLR model from SLR analysis. PLR model may include a degree upto n
    General equation:
        y = a0 + a1x + a2x^2 + a3x^3 + ............... + anx^n
        
    For the above we will take upto a degree of 3
"""

# Find 2nd power of x and construct SLR model between x^2 and y ---------------------
x1_2 = x1*x1
r2 = np.corrcoef(x1_2, y)[0, 1]   # 90.85 %

x1_2 = x1_2.reshape(-1, 1)     #Convert to 2D array as fit method demands 2D array
print(x1_2)

# SLR model between x1^2 and y
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x1_2, y)
y_pred2 = lin_reg.predict(x1_2)

# Convert x1_2 back to a 1d Array
n = []
for i in x1_2:
    for m in i:
        n.append(m)
        
x1_2 = np.array(n)

# Visualize SLR model between x1_2 vs y
plt.scatter(x1_2, y, color = 'red')
plt.plot(x1_2, y, color = 'green')
plt.plot(x1_2, y_pred2, color = 'blue', label = 'SLR line')
plt.legend()
plt.xlabel('Position^2')
plt.ylabel('Salaries')
plt.title('P^2 v/s S curve is now relatively more linear')
plt.show()
# -----------------------------------------------------------------------------------


"""
    # Curve is more linear as the correlation coefficient is now at 90.85 %
    print(np.corrcoef(x1_2, y)[0, 1]*100, "%")
"""


# Find 3rd power of x and construct SLR model between x^2 and y ---------------------
x1_3 = x1*x1*x1
r3 = np.corrcoef(x1_3, y)[0, 1] 
x1_3 = x1_3.reshape(-1, 1)     #Convert to 2D array as fit method demands 2D array

# SLR model between x1^3 and y
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x1_3, y)
y_pred3 = lin_reg.predict(x1_3)

# Convert x1_3 back to a 1d Array
n = []
for i in x1_3:
    for m in i:
        n.append(m)
        
x1_3 = np.array(n)

# Visualize SLR model between x1_2 vs y
plt.scatter(x1_3, y, color = 'red')
plt.plot(x1_3, y, color = 'green')
plt.plot(x1_3, y_pred2, color = 'blue', label = 'SLR line')
plt.legend()
plt.xlabel('Position^3')
plt.ylabel('Salaries')
plt.title('P^3 v/s S curve is now relatively more linear')
plt.show()
# ----------------------------------------------------------------------------------

"""
    Drawing regression line. Equation of regression lines:
        y1 = m1*x1 + b1
        y1 = m2*x1_2 + b2
        y1 = m3*x1_3 + b3
"""

# Calculate equation of SLR lines for both x1, x1^2, x1^3. 
# x1^2 is represented by x1_2
# x1^3 is represented by x1^3
def slope_intercept(x, y):
    x1 = np.array(x)
    y1 = np.array(y)
    
    m = ((np.mean(x1)*np.mean(y1)) - np.mean(x1*y1)) / ((np.mean(x1)*np.mean(x1)) - np.mean(x1*x1))
        
    m = round(m, 2)
    b = (np.mean(y1) - np.mean(x1)*m)
    b = round(b, 2)
    
    return m, b        

m1, b1 = slope_intercept(x1, y1)
m2, b2 = slope_intercept(x1_2, y1)
m3, b3 = slope_intercept(x1_3, y1)
# Show both SLR equations
print("y1 = ", m1, "*x1", " + ", b1, sep = '')
print("y1 = ", m2, "*x1_2", " + ", b2, sep = '')
print("y1 = ", m3, "*x1_3", " + ", b3, sep = '')


"""
    Multiply all the equations by their correlation coefficients and add
    
        y1 = m1*x1 + b1     ........ eqn 1 ......... X r1
        y1 = m2*x1_2 + b2   ........ eqn 2 ......... X r2
        y1 = m3*x1_3 + b3   ........ eqn 3 ......... X r3
     +  ______________________________________________________________________
        (r1 + r2 + r3)*y1 = m1*r1*x1 + m2*r2*x1_2 + m3*r3*x1_3 + b1 + b2 + b3
        
     =  y1 = {m1*r1*x1 + m2*r2*x2 +  m3*r3*x1_3 + b1 + b2 + b3}/(r1+r2+r3)
     
     Therefore 
     b1 = (m1*r1)/(r1 + r2 + r3) 
     b2 = (m2*r2)/(r1 + r2 + r3)
     b3 = (m3*r3)/(r1 + r2 + r3)
     b0 = (b1 + b2 + b3)/(r1 + r2 + r3)
     
     Note that: y1 here is predicted value of Salaries 
"""
y1_pred = []

for i in range(0, 10):
    s = (m1*r1*x1[i] + m2*r2*x1_2[i] + m3*r3*x1_3[i] + b1 + b2 + b3)/(r1+r2+r3)
    y1_pred.append(s)
    
y1_pred = np.array(y1_pred)

plt.scatter(x, y, color = 'red')
plt.plot(x, y1_pred, color = 'green')
plt.xlabel('Position')
plt.ylabel('Salaries')
plt.title('P v/s S curve is now non linear')
plt.show()

# From SLR the correlation coefficient has increased from 81.79 % to 91.20% when PLR was performed 
print(np.corrcoef(y, y_pred)[0, 1]*100, "%")
print(np.corrcoef(y, y1_pred)[0, 1]*100, "%") 

# Perform PLR for degree = 2
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y) 
y_PLR_pred = lin_reg_2.predict(x_poly)  
       
plt.scatter(x, y, color = 'red')
plt.plot(x, y_PLR_pred, color = 'green')
plt.xlabel('Position')
plt.ylabel('Salaries')
plt.title('PLR curve of sklearn')
plt.show()     
print(np.corrcoef(y, y_PLR_pred)[0, 1]*100, "%")    # 95.71 %

# Perform PLR for degree = 3 and keep increasing
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y) 
y_PLR_pred = lin_reg_2.predict(x_poly)  
       
plt.scatter(x, y, color = 'red')
plt.plot(x, y_PLR_pred, color = 'green')
plt.xlabel('Position')
plt.ylabel('Salaries')
plt.title('PLR curve of sklearn')
plt.show()     
print(np.corrcoef(y, y_PLR_pred)[0, 1]*100, "%")    # 99.05 %


# degree = 4
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y) 
y_PLR_pred = lin_reg_2.predict(x_poly)  
       
plt.scatter(x, y, color = 'red')
plt.plot(x, y_PLR_pred, color = 'green')
plt.xlabel('Position')
plt.ylabel('Salaries')
plt.title('PLR curve of sklearn')
plt.show()     
print(np.corrcoef(y, y_PLR_pred)[0, 1]*100, "%")    # 99.86%

# degree = 10
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 10)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y) 
y_PLR_pred = lin_reg_2.predict(x_poly)  
       
plt.scatter(x, y, color = 'red')
plt.plot(x, y_PLR_pred, color = 'green')
plt.xlabel('Position')
plt.ylabel('Salaries')
plt.title('PLR curve of sklearn')
plt.show()     
print(np.corrcoef(y, y_PLR_pred)[0, 1]*100, "%")    # 100%

"""
    At this point the PLR curve is no more non-linear between 2 points. This is same 
    as plt.plot(x, y). 
    The PLR of sklearn performs much better and produces nearly 100 % correlation
    One can also chose to use PLR from the above SLR analysis. Accuracy can be increased 
    with more number of degrees.
"""

# PLR salary Prediction algorithm --------------------------------------------
x = dataset['Level'].values.tolist()
x = np.array(x)

P = int(input("Enter Position index of applicant = "))
s = (m1*r1*P + m2*r2*P*P + m3*r3*P*P*P + b1 + b2 + b3)/(r1+r2+r3)

X = [P]
S =[s]

plt.axvline(x = Position, label = 'Position', color = 'green')
plt.plot(x, y1_pred, color = 'red', label = 'PLR curve')
plt.scatter(X, S, color = 'blue', label = 'Predicted Salary = ' + str(s))
plt.legend()
plt.title('Predict Salary')
plt.show()

print("Predicted salary for Position level ", Position, " is $", s, sep = '')
# ----------------------------------------------------------------------------
