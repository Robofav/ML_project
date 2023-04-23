from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

app = Flask(_name_)

# Load preprocessed data
df_rfm = pd.read_csv('preprocessed_data.csv')

# Perform K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(df_rfm[['InvoiceNo', 'Basket Price', 'categ_0', 'categ_1', 'categ_2','categ_3', 'categ_4']])
df_rfm['Cluster'] = kmeans.predict(df_rfm[['Basket Price', 'categ_0', 'categ_1', 'categ_2','categ_3', 'categ_4']])

# Define route for customer segmentation visualization
@app.route('/segmentation')
def segmentation():
    # Get descriptive statistics for each segment
    segment_stats = df_rfm.groupby('Cluster').agg({
        'Basket Price': 'mean',
        'categ_0': 'mean',
        'categ_1': 'mean',
        'categ_2': 'mean',
        'categ_3': 'mean',
        'categ_4': 'mean',
        'CustomerID': 'count'
    }).rename(columns={'CustomerID': 'Count'})
    
    # Render customer segmentation template with segment statistics
    return render_template('customer_segmentation.html', segment_stats=segment_stats)

if _name_ == '_main_':
    app.run(debug=True)