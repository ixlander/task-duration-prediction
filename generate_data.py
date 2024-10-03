import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

def generate_task_data(num_samples):
    tasks = []
    priorities = ['Low', 'Medium', 'High']
    
    for _ in range(num_samples):
        description = f'Task {_ + 1}'
        complexity = np.random.randint(1, 6)  
        priority = np.random.choice(priorities)
        due_date = datetime.now() + timedelta(days=np.random.randint(1, 30))  
        actual_duration = complexity * 2 + np.random.normal(0, 1)  
        
        tasks.append({
            'Task Description': description,
            'Estimated Complexity': complexity,
            'Priority Level': priority,
            'Due Date': due_date,
            'Actual Duration': actual_duration
        })
    
    return pd.DataFrame(tasks)

task_data = generate_task_data(100)
print("Generated Task Data:")
print(task_data.head())

label_encoder = LabelEncoder()
task_data['Priority Level'] = label_encoder.fit_transform(task_data['Priority Level'])

task_data = task_data.drop(columns=['Task Description', 'Due Date'])

X = task_data.drop(columns=['Actual Duration']) 
y = task_data['Actual Duration'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

results = pd.DataFrame({'Actual Duration': y_test, 'Predicted Duration': y_pred})
print("\nPredictions vs Actual:")
print(results.head())
task_data.to_csv('generated_task_data.csv', index=False)
print("\nTask data saved to 'generated_task_data.csv'")
