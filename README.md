# Disaster-response-mapping
An AI powered solution that analyses geospatial aerial imagery to map affected areas, people, vehicles, detect terrain changes and generate 3D models assisting in rescue planning and prioritisation. Provides real-time insights for emergency response teams 

Disaster response mapping is critical as natural disasters such as earthquakes, floods and landslides can cause widespread damage. Current approaches rely heavily on manual analysis of satellite or drone imagery, which is time-consuming, prone to human error, and difficult to scale across large disaster zones. Moreover, 2D imagery alone cannot capture the full 3D structure of damaged areas, making it hard to assess structural damage, flooded zones, or terrain changes accurately. The 3D structure also allows easy comprehensio of the situation, enabling faster and more effective search and rescue operations.

1. Our project aims to build an AI system that automatically processes aerial and satellite imagery to:

2. Detect and segment damaged buildings, debris, flooded areas, and roads.

3. Identify humans and vehicles for prioritizing rescue efforts.

4. Estimate depth and reconstruct 3D terrain and structures using the point cloud framework.

5. Generate actionable maps for disaster response teams in near real-time.

This approach leverages computer vision, depth estimation, pose detection, semantic segmentation, and 3D point cloud analysis to provide a comprehensive, scalable, and rapid disaster response tool.

However, in this process, one faces many challenges in terms of increasing the model's viability with respect to its performance in a real-life scenario.

Integrating Multi-Modal Data
One of the main challenges is combining different types of data, such as RGB images, depth maps, LiDAR point clouds, and human or vehicle detections, into a single coherent analysis pipeline. Aligning and synchronizing these data sources is complex but essential for accurate 3D scene understanding.

Achieving Accurate Semantic Segmentation
Large disaster zones contain diverse terrains, buildings of different types, and varying environmental conditions. Performing pixel-wise segmentation to correctly identify flooded areas, debris, roads, and damaged structures is difficult, especially when visual conditions change across images.

Detecting Humans and Vehicles in Aerial Views
People and vehicles are often very small relative to satellite or drone imagery. They may also be partially occluded, in shadows, or seen from unusual angles. Accurately detecting them in such scenarios is a significant technical challenge.

Estimating Depth and Reconstructing 3D Scenes
Depth estimation from monocular or stereo images is difficult in disaster-affected areas because the terrain is irregular and structures are damaged. Additionally, point cloud data obtained from LiDAR or photogrammetry often contains noise, which makes 3D reconstruction and analysis more complex.

Meeting Real-Time Processing Requirements
Emergency response demands near real-time analysis. Many of the models used in this project, such as semantic segmentation networks or PointNet for 3D data, are computationally intensive. Optimizing these models to run quickly while maintaining accuracy is challenging.

Limited Data and Annotation Availability
High-quality annotated datasets for disaster scenarios are limited. Creating new datasets is time-consuming, and models must often generalize to new disaster types, geographic regions, or imaging conditions that they have not seen before.

Ensuring Model Robustness and Generalization
Models must perform reliably under diverse lighting, weather, and terrain conditions. There is a risk of overfitting to a specific dataset or disaster type, which could reduce the model’s effectiveness in real-world applications.
