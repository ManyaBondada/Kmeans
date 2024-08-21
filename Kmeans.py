import numpy as np
import re
import pandas as pd
import string
import random as rd
import math


class Kmeans:
    ## initalize data
    def __init__(self, dataFile, delimiter, header=True):
        column_names = ['Tweet ID', 'Tweet Timestamp', 'Tweet Content']
        self.raw_input = pd.read_csv(dataFile, delimiter = delimiter, names=column_names)
        self.X = self.raw_input.drop(['Tweet ID', 'Tweet Timestamp'], axis=1)

    ## code for pre-processing the dataset
    def preprocess(self):
        self.X['Tweet Content'] = self.X['Tweet Content'].apply(self.remove_mentions)
        self.X['Tweet Content'] = self.X['Tweet Content'].apply(self.remove_hashtags)
        self.X['Tweet Content'] = self.X['Tweet Content'].apply(self.remove_urls)
        self.X['Tweet Content'] = self.X['Tweet Content'].apply(self.remove_amps)
        self.X['Tweet Content'] = self.X['Tweet Content'].apply(self.remove_punct)
        self.tweets = self.X.apply(lambda x: x.astype(str).str.lower())

        return 0
    
    ## pre-processing helper functions

    # remove words starting with "@"
    @staticmethod
    def remove_mentions(tweet):
        return ' '.join(re.sub(r'@\w+', '', word) for word in tweet.split())
    
    # remove "#" from content
    @staticmethod
    def remove_hashtags(tweet):
        return ' '.join(re.sub(r'#', '', word) for word in tweet.split())
    
    # remove urls from content
    @staticmethod
    def remove_urls(tweet):
        return ' '.join(re.sub(r'http\S+', '', word) for word in tweet.split())

    # remove "&amp;" from content
    @staticmethod
    def remove_amps(tweet):
        return ' '.join(re.sub(r'&\w+', '', word) for word in tweet.split())
    
    # remove stray punctuation from content
    @staticmethod
    def remove_punct(tweet):
        translator = str.maketrans('', '', string.punctuation)
        return tweet.translate(translator)
    
    ## intialize k-means with tokenized words
    def kmeans_init(self):
        arr = self.tweets.to_numpy()
        self.tweets_arr = [sentence[0].split() for sentence in arr]


    ## perform k_means algorithm
    def k_means_clustering(self, k, max_iterations=100, epsilon=1e-4):
        # Randomly initialize k unique centroids
        centroids = []
        centroids = rd.sample(self.tweets_arr, k)

        for iter in range(max_iterations):

            # Assign each data point to the nearest centroid
            clusters = []
            for i in range(k):
                clusters.append([])

            for point in self.tweets_arr:
                distances = []
                for centroid in centroids:
                    distances.append(self.jaccard_distance(set(point), centroid))
                closest_centroid = np.argmin(distances)
                clusters[closest_centroid].append(point)

            # find new centroid by calculating the tweet that has minimum distance to all other tweets in cluster
            new_centroids = []
            for cluster in clusters:
                # check that the cluster is not empty
                if not cluster:  
                    continue

                min_distance_sum = math.inf
                best_centroid = None

                # find the tweet with min distance to other tweets in cluster
                for i, point in enumerate(cluster):
                    distance_sum = 0
                    for j, point2 in enumerate(cluster):
                        if i != j:
                            distance_sum += self.jaccard_distance(set(point), set(point2))

                    if distance_sum < min_distance_sum:
                        min_distance_sum = distance_sum
                        best_centroid = point

                new_centroids.append(set(best_centroid))

            # Check for convergence by calculating distance between previous and new centroids
            if all(self.jaccard_distance(set(prev), set(new)) < epsilon for prev, new in zip(centroids, new_centroids)):
                break
            
            centroids = new_centroids

        # Calculate SSE (Sum of Squared Errors) and cluster sizes
        sse = 0
        cluster_sizes = []
        for i, cluster in enumerate(clusters):
            cluster_sizes.append(len(cluster))
            for point in cluster:
                sse += self.jaccard_distance(set(point), centroids[i]) ** 2

        return sse, cluster_sizes
    
    # calculates the dissimiliarity between 2 points
    @staticmethod
    def jaccard_distance(set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        if union <= 0:
            return 0.0
        else:
            return 1.0 - (intersection / union)
    
    # call the kmeans algorithm and prints out results
    def get_results(self, k_arr=[1, 10, 25, 35, 50]):
        print("Starting kmeans algorithm...")

        column_names = ['k-value', 'SSE', 'Size of each Clusters']
        results = pd.DataFrame(columns=column_names)    

        for k in k_arr:
            sse, cluster_sizes = self.k_means_clustering(k)
            temp_df = pd.DataFrame({'k-value': [k], 'SSE': [sse], 'Size of each Clusters': [cluster_sizes]})
            results = pd.concat([results, temp_df], ignore_index=True)

        print("Printing results...")
        print("-------------------")
        for index, result in results.iterrows():
            print('K-value =', result['k-value'])
            print('SSE =', result['SSE'])

            print('Cluster sizes:')
            for size in result['Size of each Clusters']:
                print(size, end=' ')
            print('\n')

if __name__ == "__main__":
    kmeans_tweets = Kmeans("https://raw.githubusercontent.com/ManyaBondada/CS-4375.001.HW3_data/main/usnewshealth.txt", delimiter='|')
    kmeans_tweets.preprocess()
    kmeans_tweets.kmeans_init()
    kmeans_tweets.get_results()
