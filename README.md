Student Performance Risk Prediction Project
This project aims to predict student performance risk based on various factors using machine learning and deep learning models.

Project Goal
The main goal is to build a classification model that can categorize students into different risk levels (High, Medium, Low) based on their academic and personal attributes. This can help educators identify students who might need additional support.

Dataset
The project uses two datasets: student-mat.csv and student-por.csv, containing data about students from two schools, focusing on their math and Portuguese course performance, respectively. The datasets were merged and preprocessed to create a single dataset of unique students.

Methodology
The project follows a standard machine learning workflow:

Data Loading and Merging: Load the two datasets and merge them based on common student attributes, handling potential duplicates.
Data Preprocessing:
Handle missing values (none were found in this dataset).
Encode categorical features using one-hot encoding.
Engineer new features such as 'Attendance Ratio' and 'Average Grade'.
Target Variable Definition: Create a 'Risk Category' target variable based on the final grade (G3), categorizing students into High, Medium, and Low risk.
Data Splitting and Normalization: Split the preprocessed data into training and testing sets and normalize numerical features using Min-Max scaling.
Model Training: Train several traditional machine learning models (Logistic Regression, Decision Tree, Random Forest, SVM) and a deep learning model (Neural Network) for multi-class classification.
Model Evaluation: Evaluate the performance of each model using metrics like Accuracy, Precision, Recall, and F1-score.
Model Selection: Choose the best performing model based on the evaluation metrics, considering potential overfitting.
Model Saving: Save the selected best model for future use.
Sentiment Analysis (Optional): A simulated dataset of student feedback was created and sentiment analysis was performed using VADER. This was explored as a potential additional feature but not integrated into the main model due to the simulated nature of the data.
Models Used
Logistic Regression
Decision Tree
Random Forest
Support Vector Machine (SVM)
Deep Learning (Sequential Neural Network)
Results
The evaluation metrics for each model are available in the notebook. The Random Forest model was selected as the best performing model for this multi-class classification task, demonstrating high accuracy and other relevant metrics on the test set.

Files
student-mat.csv: Dataset for math course.
student-por.csv: Dataset for Portuguese course.
best_student_performance_model.pkl: The saved best performing model (Random Forest).
app.py: A basic Flask application demonstrating how to load the saved model and make predictions.
columns.txt: A file containing the names of the columns used for training the model, in the correct order. This is useful for ensuring new input data has the same structure.
requirements.txt: A file listing the Python dependencies required to run the project.
How to Use
Run the Jupyter Notebook (.ipynb file) to execute the data preprocessing, model training, and evaluation steps.
The best performing model will be saved as best_student_performance_model.pkl.
The app.py file provides a basic example of how to load the saved model and use it for predictions within a web framework (Flask). You would need to install the required libraries (pip install -r requirements.txt) and potentially adapt the app.py for your specific deployment environment.
Future Work
Explore more advanced feature engineering techniques.
Experiment with different machine learning algorithms and deep learning architectures.
Perform hyperparameter tuning to optimize model performance.
Integrate the sentiment analysis results as a feature if real student feedback data is available.
Build a more robust and scalable deployment solution.
