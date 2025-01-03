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

def centroid_similarity_Alligator():
    category1_centroids = np.array([[53.812164, 53.91863], [59.56771, 133.16147], [231.16188, 113.712234]])
    category2_centroids = np.array([[120.51513, 238.16223], [216.13316, 144.85582], [83.49493, 76.65499]])
    category3_centroids = np.array([[85.27962, 62.488113], [81.69331, 359.1907], [77.99171, 176.01576]])

    return category1_centroids, category2_centroids, category3_centroids

def centroid_similarity_LongCrack():
    pass

def centroid_similarity_OtherCrack():
    pass

def centroid_similarity_Patching():
    pass

def centroid_similarity_Potholes():
    pass

def similarity_to_alligator(images, category):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    query_image_path = images
    query_edges = get_edge_coordinates(query_image_path)

    k = 3 
    _, labels, query_centroids = cv2.kmeans(np.float32(query_edges), k, None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS)

    print("Centroid Gambar Baru:\n", query_centroids)

    category1_centroids, category2_centroids, category3_centroids = centroid_similarity_Alligator()

    similarity_to_category1 = compute_cosine_similarity(query_centroids, category1_centroids)
    similarity_to_category2 = compute_cosine_similarity(query_centroids, category2_centroids)
    similarity_to_category3 = compute_cosine_similarity(query_centroids, category3_centroids)

    similarities = {
        "AlligatorCrack Rendah": similarity_to_category1,
        "AlligatorCrack Sedang": similarity_to_category2,
        "AlligatorCrack Tinggi": similarity_to_category3,
    }

    closest_category = max(similarities, key=similarities.get)  # Karena cosine similarity, semakin besar semakin mirip

    # Output hasil perbandingan
    print("Cosine Similarity ke AlligatorCrack Rendah:", similarity_to_category1)
    print("Cosine Similarity ke AlligatorCrack Sedang:", similarity_to_category2)
    print("Cosine Similarity ke AlligatorCrack Tinggi:", similarity_to_category3)
    print("Gambar baru paling mirip dengan:", closest_category)

    return similarities