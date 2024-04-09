from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load the pre-trained Word2Vec model
model = Word2Vec.load('model.bin')

# Get the word vectors and corresponding words
word_vectors = model.wv.vectors
words = model.wv.index_to_key

# Perform clustering using K-means
num_clusters = 3  # Number of clusters
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(word_vectors)
cluster_indices = kmeans.predict(word_vectors)

# Reduce the dimensionality using t-SNE
tsne = TSNE(n_components=2)
reduced_dim = tsne.fit_transform(word_vectors)

# Visualize the clusters
# plt.figure(figsize=(10, 10))
plt.figure()
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use a Chinese font that supports the characters
plt.rcParams['axes.unicode_minus'] = False  # Ensure that minus signs are displayed correctly
# for i in range(len(words)):
for i in range(100):
    x, y = reduced_dim[i, :]
    plt.scatter(x, y, c=cluster_indices[i], cmap='viridis')
    plt.text(x, y, words[i])

plt.show()
