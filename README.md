# Mechanical Shop Recommendation System

This repository contains code for a recommendation system for a mechanical shop. The system utilizes collaborative filtering and content-based filtering techniques, as well as a neural network model for hybrid filtering. It also includes data preprocessing steps and data visualization.
Columns Explanation
# Preprocessing and Data Loading
•	customer_data: A DataFrame that contains historical customer data. It is loaded from a CSV file called "synthetic_data.csv".
# Collaborative Filtering
•	user_item_matrix: A pivot table that represents the user-item matrix. It is computed from the customer_data DataFrame and contains user ratings for each item. If a user hasn't rated an item, the value is filled with 0.
•	user_similarity: A matrix that represents the similarity between users using cosine similarity. It is computed based on the user_item_matrix.
•	N: The number of similar users to consider for each user.
•	top_similar_users: A dictionary that stores the top N similar users for each user. It is computed based on the user_similarity matrix.
# Content-based Filtering.
•	item_matrix: A matrix representing the item-item matrix using TF-IDF vectors. It is computed from the item descriptions in the customer_data DataFrame.
•	item_similarity: A matrix representing the similarity between items using cosine similarity. It is computed based on the item_matrix.
•	top_similar_items: A dictionary that stores the top N similar items for each item. It is computed based on the item_similarity matrix.
# Neural Network Model for Hybrid Filtering
•	user_input: Input layer for the user ID.
•	item_input: Input layer for the item ID.
•	embedding_dim: The dimension of the embedding for users and items.
•	num_items: The number of unique items in the customer_data DataFrame.
•	user_embedding: Embedding layer for the user ID.
•	item_embedding: Embedding layer for the item ID.
•	dot_product: Dot product between user and item embeddings.
•	output: Output layer of the model.
•	model: The compiled neural network model that takes user and item inputs and predicts the purchase probability.
# Data Visualization
•	user_item_matrix: Heatmap plot of the user-item matrix.
•	item_similarity: Heatmap plot of the item-item similarity matrix.
# Instructions
To use the recommendation system and visualize the data, follow these steps:
1.	Load the historical customer data from the "synthetic_data.csv" file.
2.	Compute the user-item matrix using collaborative filtering.
3.	Compute the similarity between users using cosine similarity.
4.	Get the top N similar users for each user.
5.	Compute the item-item matrix using TF-IDF vectors for content-based filtering.
6.	Compute the similarity between items using cosine similarity.
7.	Get the top N similar items for each item.
8.	Define the neural network model for hybrid filtering.
9.	Preprocess the data by encoding the user_id column with unique integer values.
10.	Train the model using the user_id, item_id, and purchase columns.
11.	Visualize the user-item matrix heatmap.
12.	Visualize the item-item matrix heatmap.
Note: Make sure to have the required dependencies installed, such as pandas, sklearn, tensorflow, matplotlib, seaborn, and numpy.
