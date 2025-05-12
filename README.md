# 🎬 Storyboard: CLV Prediction Model
🧩 1. Introduction Screen
📌 Title: Customer Lifetime Value Prediction

📝 Description: Predict how valuable your customers are over time using machine learning.

📂 User Action: Upload your customer data as a .csv file.

🔍 2. Data Upload & Preview
📊 Components:

📁 File uploader

👀 Data preview (first 100 rows)

🏷️ List of column names

📈 Summary statistics (mean, std, etc.)

🧠 3. Model Information
🤖 Status Display:

✅ Pre-trained model loaded

🔄 New model trained if no saved version exists

📉 Metrics: MSE and R² score shown (if model is trained)

🎯 4. Feature Selection
🔽 Dropdown Menus:

🕒 Recency

🔁 Frequency

💰 Monetary Value

🔒 Ensures correct feature mapping by user

🧾 5. Model Description Section
📚 Auto-generated explanation:

📦 Recency: Time since last purchase

🔁 Frequency: Number of purchases

💸 Monetary: Total money spent

🌲 Model Used: Random Forest Regressor

🛠️ 6. Report Generation
🖱️ Button: "Generate Report"

⏳ Progress Bar: Shows real-time progress

💬 Status Text: Time elapsed while generating visuals

❗Error Feedback: Alerts for missing or misformatted columns

📈 7. Prediction Phase
🔍 Preprocessing:

🔄 Converts range strings (e.g., '5L-10L') to numbers

🧼 Handles nulls and type mismatches

📥 Prediction Output:

Adds predicted_clv to dataset

Displays first few results

📊 8. Visualizations
📉 CLV Line Chart:

Sorted predictions shown via interactive Plotly chart

📋 Feature Importance:

Horizontal bar chart displaying each feature's contribution to the model

🎉 9. Completion Message
🕓 Final Time Taken: Displays elapsed time

✅ Progress Bar: Reaches 100% with success message


python code :
 - <a href = "https://github.com/akshya408/Customer-Lifetime-Value-Prediction-Model/blob/main/Customer%20Lifetime%20Value%20Prediction%20Model.py">python code</a>

sample file :
 - <a href = "https://github.com/akshya408/Customer-Lifetime-Value-Prediction-Model/blob/main/test_koRSKBP.csv">sample file</a>
 
 
local URL:
- http://localhost:8501

![Predict Customer Value with AI (2)](https://github.com/user-attachments/assets/686d2a94-b5c2-4465-899d-1d531f6e143f)
