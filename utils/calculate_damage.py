import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_edge_coordinates(pil_image):
    """
    Get edge coordinates from a PIL.Image.Image using Canny edge detection.
    """
    image_array = np.array(pil_image.convert("L"))
    edges = cv2.Canny(image_array, threshold1=50, threshold2=100)
    coordinates = np.column_stack(np.where(edges > 0))
    return coordinates

def compute_cosine_similarity(centroids1, centroids2):
    """
    Menghitung rata-rata cosine similarity antara dua set centroid.
    """
    similarity = cosine_similarity(centroids1, centroids2)
    return np.mean(similarity)

def centroid_similarity_Alligator(query_centroids):
    category1_centroids = np.array([[53.812164, 53.91863], [59.56771, 133.16147], [231.16188, 113.712234]])
    category2_centroids = np.array([[120.51513, 238.16223], [216.13316, 144.85582], [83.49493, 76.65499]])
    category3_centroids = np.array([[85.27962, 62.488113], [81.69331, 359.1907], [77.99171, 176.01576]])

    similarity_to_category1 = compute_cosine_similarity(query_centroids, category1_centroids)
    similarity_to_category2 = compute_cosine_similarity(query_centroids, category2_centroids)
    similarity_to_category3 = compute_cosine_similarity(query_centroids, category3_centroids)

    similarities = {
        "AlligatorCrack Low": similarity_to_category1,
        "AlligatorCrack Medium": similarity_to_category2,
        "AlligatorCrack High": similarity_to_category3,
    }

    return similarities

def centroid_similarity_LongCrack(query_centroids):
    category1_centroids = np.array([[30.218868, 118.63019], [39.53809, 23.395407], [129.51843, 18.040222]])
    category2_centroids = np.array([[183.75424, 48.052547], [296.18903, 38.131664], [84.007416, 62.99691]])
    category3_centroids = np.array([[ 59.420048, 56.696503], [337.30292, 162.28363], [193.32925, 375.1406]])

    similarity_to_category1 = compute_cosine_similarity(query_centroids, category1_centroids)
    similarity_to_category2 = compute_cosine_similarity(query_centroids, category2_centroids)
    similarity_to_category3 = compute_cosine_similarity(query_centroids, category3_centroids)

    similarities = {
        "LongitudinalCrack Low": similarity_to_category1,
        "LongitudinalCrack Medium": similarity_to_category2,
        "LongitudinalCrack High": similarity_to_category3,
    }

    return similarities

def centroid_similarity_OtherCrack(query_centroids):
    category1_centroids = np.array([[60.822815, 223.83606], [33.158257, 45.139465], [46.817276, 421.22812]])
    category2_centroids = np.array([[34.394737, 107.26316], [13.724753, 20.106932], [21.202703, 217.33784]])
    category3_centroids = np.array([[27.515152, 256.27274], [190.48914, 135.61957], [57.820225, 48.7191]])

    similarity_to_category1 = compute_cosine_similarity(query_centroids, category1_centroids)
    similarity_to_category2 = compute_cosine_similarity(query_centroids, category2_centroids)
    similarity_to_category3 = compute_cosine_similarity(query_centroids, category3_centroids)

    similarities = {
        "OtherCrack Low": similarity_to_category1,
        "OtherCrack Medium": similarity_to_category2,
        "OtherCrack High": similarity_to_category3,
    }

    return similarities

def centroid_similarity_Patching(query_centroids):
    category1_centroids = np.array([[60.822815, 223.83606], [33.158257, 45.139465], [46.817276, 421.22812]])
    category2_centroids = np.array([[34.394737, 107.26316], [13.724753, 20.106932], [21.202703, 217.33784]])
    category3_centroids = np.array([[27.515152, 256.27274], [190.48914, 135.61957], [57.820225, 48.7191]])

    similarity_to_category1 = compute_cosine_similarity(query_centroids, category1_centroids)
    similarity_to_category2 = compute_cosine_similarity(query_centroids, category2_centroids)
    similarity_to_category3 = compute_cosine_similarity(query_centroids, category3_centroids)

    similarities = {
        "Patching Low": similarity_to_category1,
        "Patching Medium": similarity_to_category2,
        "Patching High": similarity_to_category3,
    }

    return similarities

def similarity_algoritm(images, category):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    query_image_path = images
    query_edges = get_edge_coordinates(query_image_path)

    k = 3 
    _, labels, query_centroids = cv2.kmeans(np.float32(query_edges), k, None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS)

    if category == "AlligatorCrack":
        similarities = centroid_similarity_Alligator(query_centroids)
    elif category == "LongCrack":
        similarities = centroid_similarity_LongCrack(query_centroids)
    elif category == "Patching":
        similarities = centroid_similarity_Patching(query_centroids)
    elif category == "OtherCrack":
        similarities = centroid_similarity_OtherCrack(query_centroids)

    closest_category = max(similarities, key=similarities.get)  # Karena cosine similarity, semakin besar semakin mirip

    # Output hasil perbandingan
    print("Gambar baru paling mirip dengan:", closest_category)

    return similarities