import cv2
import numpy as np
import time
from skimage.morphology import skeletonize
import networkx as nx


class PalmBasic:

    def __init__(self):
        pass

    def gaussian_blur(self, img, kernel_size=(5, 5), sigma=2, blur=True):
        if blur:
            return cv2.GaussianBlur(img, kernel_size, sigma)
        else:
            return img
        

    def threshold_image(self, img, threshold_val=20, bi_threshold=True):
        if bi_threshold:
            _, thresholded = cv2.threshold(img, threshold_val, 255, cv2.THRESH_BINARY)
        else:
            _, thresholded = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresholded
    
    def compute_edist(self, point1,point2):
        return np.sqrt(np.sum((point1-point2)**2))
        
    def find_largest_component(self, img):
        
        _, labels, stats, _ = cv2.connectedComponentsWithStats(img)
        largest_component_label = np.argmax(stats[1:, -1]) + 1
        max_comp_img = np.zeros_like(img)
        max_comp_img[labels == largest_component_label] = 255
        max_comp_contour_coord = self.find_contour(max_comp_img)
        
        max_comp_contour_img = cv2.drawContours(np.zeros_like(img, dtype=np.uint8), [max_comp_contour_coord], -1, (255), thickness=1)
        
        return max_comp_img, max_comp_contour_coord,max_comp_contour_img

    def find_contour(self, img):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(contours, key=cv2.contourArea)
        return cnt

    def fill_contour(self, img, contour, filled=False):
        img_contour = np.zeros_like(img, dtype=np.uint8)
        thickness = cv2.FILLED if filled else 1
        cv2.drawContours(img_contour, [contour], -1, (255), thickness=thickness)
        return img_contour

    def erode_binary_image(self, img, kernel_size=(7, 7), iterations=1, th_binary=180):
        kernel = np.ones(kernel_size, np.uint8)
        eroded = cv2.erode(img, kernel, iterations=iterations)
        _, binaried = cv2.threshold(eroded, th_binary, 255, cv2.THRESH_BINARY)
        return binaried

    def resize_image(self, img, ratio):
        img_resized = cv2.resize(img, (img.shape[1] // ratio, img.shape[0] // ratio))
        return img_resized
    
    def hull_image(self, img, img_contour):
        
        epsilon = 0.01*cv2.arcLength(img_contour,True)
        approx = cv2.approxPolyDP(img_contour,epsilon,True)
        
        hull_coord = cv2.convexHull(approx, clockwise=False,returnPoints=True)
        
        hull_img = cv2.drawContours(np.zeros_like(img, dtype=np.uint8), [hull_coord], -1, 255, thickness=cv2.FILLED)
        return hull_coord, hull_img 

    def get_skeleton(self, img):

        # Make sure image is binary in skimage's expected format
        img = img == 255

        # Perform the skeletonization
        skeleton = skeletonize(img)*255
        
            
        return skeleton
    
    def euclidean_distance(self, point1, point2):
        # 将点转换为 numpy 向量
        point1 = np.array(point1)
        point2 = np.array(point2)

        # 计算两点之间的差
        difference = point1 - point2

        # 计算差的平方
        squared_difference = np.square(difference)

        # 计算所有差的平方的总和
        sum_of_squared_difference = np.sum(squared_difference)

        # 对总和开根号，得到欧氏距离
        distance = np.sqrt(sum_of_squared_difference)

        return distance
    
    def find_closest_path_graph(self,matrix):

        cleaned_matrix = np.zeros_like(matrix)
        
        #创建图的时间
        # start_time = time.time()
        graph = self.build_undirected_graph(matrix)
        # end_time = time.time()
        # execution_time = end_time - start_time
        # print("Build graph time: {:.4f} seconds".format(execution_time))
        
        nodes_with_degree_1 = [node for node in graph.nodes() if graph.degree[node] == 1]
        
        leftmost_node = None
        min_x = float('inf')

        # Traverse the nodes with degree 1 and find the leftmost node
        for node in nodes_with_degree_1:
            x = node[1]  # Assuming (x, y) format for node coordinates

            if x < min_x:
                min_x = x
                leftmost_node = node
        source_node = leftmost_node
        
        longest_shortest_path = []
        # Traverse all other nodes in nodes_with_degree_1
        for node in nodes_with_degree_1:
            if node != source_node:
                # Find the shortest path between the source and current node
                shortest_path = nx.shortest_path(graph, source=source_node, target=node)
                if len(shortest_path) > len(longest_shortest_path):
                    longest_shortest_path = shortest_path

        for point in longest_shortest_path:
            cleaned_matrix[point[0],point[1]]=255
        
        
        return [cleaned_matrix,longest_shortest_path]
    
    def build_undirected_graph(self,matrix):
        # rows, cols = matrix.shape

        # Find non-zero (255) elements
        non_zero_elements = np.argwhere(matrix == 255)

        # Find min and max row and column
        min_row, min_col = np.min(non_zero_elements, axis=0)
        max_row, max_col = np.max(non_zero_elements, axis=0)

        graph = nx.Graph()

        # Directions for neighbors (left, upper left, up, upper right)
        directions = [(-1, 0), (-1, -1), (0, -1), (1, -1)]

        # Traverse only the confined area
        for i in range(min_row, max_row + 1):
            for j in range(min_col, max_col + 1):
                if matrix[i, j] == 255:
                    current_node = (i, j)
                    graph.add_node(current_node)

                    # Check part of the neighboring points
                    for direction in directions:
                        neighbor = (i + direction[0], j + direction[1])

                        if (min_row <= neighbor[0] <= max_row) and (min_col <= neighbor[1] <= max_col) and matrix[neighbor[0], neighbor[1]] == 255:
                            graph.add_edge(current_node, neighbor)

        return graph
    

    def judge_valley(self,path):
    
        # points = np.argwhere(matrix == 255)

        # # Find the left-most and right-most points
        # rightmost = max(points, key=lambda p: p[1])
        
        rightmost_node = max(path, key=lambda node: node[1])

        # Find the index of the leftmost node in the longest shortest path
        rightmost_index = path.index(rightmost_node)
        
        nodes_between1 = path[rightmost_index + 1:] 
        nodes_between2 = path[:rightmost_index + 1]
        nodes_between= min(len(nodes_between1),len(nodes_between2))

        flag_valley = False  
        if nodes_between>(len(path)/3):
            flag_valley = True  

        return flag_valley,rightmost_index

    def find_farthest_point(self,path,th):
        len_path = len(path)
        path = np.array(path)
        path = path[:,[1,0]]
    
        left_ind = int(len_path/5*2)
        left_node = path[left_ind]   
        right_ind = len_path-1
        right_node = path[right_ind]   
        fix_vect = left_node - right_node
        
        estimated_nodes = path[left_ind+1:-1]
        
        estimated_vects = estimated_nodes - right_node
        
        angle_degrees = self.compute_clockwise_angle_out(estimated_vects,fix_vect)
        
        flag_inter = np.abs(np.abs(np.mean(angle_degrees))-np.mean(np.abs(angle_degrees)))>1.5
 
        while flag_inter and left_ind<(len_path-1) and right_ind>0:
            if left_ind<int(len_path/3):
                left_ind+=2
                left_node = path[left_ind] 

            elif right_ind >int((len_path/5*4)):
                right_ind-=2
                right_node = path[right_ind]  
   
            else:
                break 
                 
            fix_vect = left_node - right_node
            
            estimated_nodes = path[left_ind+1:right_ind]
            
            estimated_vects = estimated_nodes - right_node
            
            angle_degrees = self.compute_clockwise_angle_out(estimated_vects,fix_vect)
            
            flag_inter = np.abs(np.abs(np.mean(angle_degrees))-np.mean(np.abs(angle_degrees)))>1.5

        path_valid = path[left_ind:right_ind+1]
        
        distances = self.compute_perpendicular_dist(path_valid)
        
        max_idx = np.argmax(distances)
        
        if distances[max_idx] > th:
            
            path_valid = path_valid[:max_idx+4]

        a = 1
        return path_valid
        
    
    def compute_perpendicular_dist(self,path):    
        
        x1, y1 = path[0]
        x2, y2 = path[-1]
        
        path_valid = path[1:-1]

        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        
        distances = abs(A * path_valid[:,0] + B * path_valid[:,1] + C) / (np.sqrt(A**2 + B**2)+0.00000000001)

            
        return distances

    def compute_clockwise_angle_out(self,fix_vect,moved_vect):
        cross_product = np.cross(moved_vect,fix_vect)
        
        dot_product = np.dot(moved_vect, fix_vect.T)

        
        angle_relative = np.arctan2(cross_product, dot_product)
        angle_degrees = np.degrees(angle_relative)
        
        return angle_degrees

    def find_closest_white_point(self, image, point):
        # Get all the non-zero (i.e., 255) points in the binary image
        non_zero_points = cv2.findNonZero(image)

        # Compute the Euclidean distances from the given point to all non-zero points
        distances = np.linalg.norm(non_zero_points - point, axis=2)

        # Get the index of the minimum distance
        min_index = np.argmin(distances)

        # Return the closest non-zero point
        return tuple(non_zero_points[min_index][0]),distances[min_index][0]  
 
 
   