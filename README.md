<h1>HDB Resale Price Predictor</h1>
<h2>Project Overview</h2>
<p>The HDB Resale Price System is a python-based application designed to predict real-time resale price for a specified location. By utilising advanced machine learning models such as the Linear Regression and Random Forest, this system provides accurate forecasts for future public home prices. With comprehensive data integration that incorporates essential macroeconomic factors like the Consumer Price Index(CPI), the application offers users a holistic understanding of property price fluctuations. Hyperparameter tuning further enhances the performance and generalisation of the machine learning models, ensuring reliable predictions validated through extensive testing. This empowers individuals, real estate professionals, and stakeholders to make well-informed decisions in the dynamic realm of property prices, supported by transparent and dependable performance evaluation metrics. The application is hosted on Streamlit, offering an interactive web interface.</p>
<div>
    <h2>Technologies Used</h2>
    <h3>Data Manipulation and Analysis</h3>
    <ul>
        <li><code>pandas</code>: For data manipulation and analysis.</li>
        <li><code>numpy</code>: For numerical computations.</li>
    </ul>
    <h3>Data Visualization</h3>
    <ul>
        <li><code>matplotlib.pyplot</code>: For creating static, animated, and interactive visualizations.</li>
        <li><code>seaborn</code>: For data visualization based on matplotlib.</li>
        <li><code>pywaffle</code>: For creating waffle charts.</li>
        <li><code>joypy</code>: For visualizing distributions of variables using Joy plots.</li>
    </ul>
    <h3>Statistical Analysis</h3>
    <ul>
        <li><code>statsmodels</code>: For estimating and interpreting models for statistical analysis.</li>
        <li><code>scipy.stats</code>: For statistical functions including spearmanr and pearsonr.</li>
    </ul>
    <h3>Machine Learning</h3>
    <ul>
        <li><code>scikit-learn</code>: For implementing machine learning algorithms such as Linear Regression and Random Forest Regressor.</li>
        <li><code>GridSearchCV</code>: For hyperparameter tuning of machine learning models.</li>
    </ul>
    <h3>Model Evaluation and Validation</h3>
    <ul>
        <li><code>sklearn.metrics</code>: For model evaluation metrics such as R² score and mean absolute error.</li>
        <li><code>yellowbrick.regressor</code>: For visualization of model diagnostics.</li>
        <li><code>CooksDistance</code>, <code>ResidualsPlot</code>: For identifying influential observations and plotting residuals of models.</li>
    </ul>
    <h3>Preprocessing</h3>
    <ul>
        <li><code>StandardScaler</code>: For feature scaling.</li>
        <li><code>train_test_split</code>: For splitting the data into training and test sets.</li>
    </ul>
    <h3>Model Persistence</h3>
    <ul>
        <li><code>joblib</code>: For saving and loading machine learning models.</li>
    </ul>
</div>
<div>
    <h2>Performance Measurement</h2>
    <p>In the development of our HDB Resale Price Predictor, various evaluation metrics were employed to assess the performance of the house pricing prediction models:</p>
    <ul>
        <li><strong>R² Score</strong>: Used to measure the proportion of variance in the target variable explained by the predictors. This allowed comparison of the predictive power of different models:
            <ul>
                <li>Linear Regression (with outliers): R² Score = 0.90</li>
                <li>Linear Regression (without outliers): R² Score = 0.87</li>
                <li>Random Forest (Out-of-bag): R² Score = 0.966</li>
                <li>Random Forest (K-fold Cross Validation): R² Score = 0.967</li>
            </ul>
        </li>
        <li><strong>Mean Absolute Error (MAE)</strong>: Calculated for the Random Forest models to quantify the average magnitude of errors, providing a straightforward interpretation of the average prediction error.</li>
        <li><strong>Correlation Coefficients (Spearman and Pearson)</strong>: Employed to assess the relationship between predicted and actual resale prices, ensuring a thorough evaluation of model effectiveness.</li>
    </ul>
    <p>Hyperparameter tuning was conducted, especially for the Random Forest model, to identify the optimal parameters, such as the number of trees in the forest and the maximum depth of each tree. This tuning aimed to maximize the model's predictive performance while avoiding overfitting or underfitting.</p>
    <p>The final model chosen was the Random Forest with K-fold Cross-Validation, due to its superior predictive performance and robust evaluation methodology. This model's high R² score and strong correlation with true prices indicate its reliability and strong explanatory power for predicting HDB resale prices.</p>
</div>
<h2>Installation and Setup</h2>
<p>To set up this project locally:</p>
<ol>
    <li>Clone the repository to your local machine.</li>
    <li>Navigate to the project directory.</li>
    <li>Install the required dependencies:
        <pre><code>pip install -r requirements.txt</code></pre>
    </li>
    <li>Run the Streamlit application:
        <pre><code>streamlit run streamlit_app.py</code></pre>
    </li>
</ol>
<h2>Acknowledgments</h2>
<p>
    - Dataset source: <a href="https://www.hdb.gov.sg/residential/selling-a-flat/overview/resale-statistics">HDB Resale Dataset</a><br>
    - Streamlit: <a href="https://streamlit.io/">Streamlit website</a>
</p>
