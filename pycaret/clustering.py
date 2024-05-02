from pycaret.datasets import get_data
data = get_data('data2')

from pycaret.clustering import *
s = setup(data, normalize = True)

kmeans = create_model('kmeans')

print(kmeans)

evaluate_model(kmeans)

plot_model(kmeans, plot = 'elbow')
plot_model(kmeans, plot = 'silhouette')

plot_model(kmeans, plot = 'tsne')


result = assign_model(kmeans)
result.head()


predictions = predict_model(kmeans, data = data)
predictions.head()

save_model(kmeans, 'kmeans_pipeline')

loaded_model = load_model('kmeans_pipeline')
print(loaded_model)