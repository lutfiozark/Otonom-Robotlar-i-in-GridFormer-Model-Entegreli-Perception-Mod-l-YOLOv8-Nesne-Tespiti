/**
 * @file bbox2costmap_node.cpp
 * @brief Convert YOLO detection bounding boxes to PointCloud2 for Nav2 costmap
 * 
 * Subscribes to Detection2DArray and publishes PointCloud2 with obstacle points
 * for Nav2 obstacle layer integration
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <cmath>

class BBox2CostmapNode : public rclcpp::Node
{
public:
    BBox2CostmapNode() : Node("bbox2costmap_node")
    {
        // Parameters
        this->declare_parameter("detection_topic", "/camera/image_detections");
        this->declare_parameter("cloud_topic", "/bbox_cloud");
        this->declare_parameter("assumed_depth", 1.0);
        this->declare_parameter("camera_fov", 60.0);
        this->declare_parameter("image_width", 640.0);
        this->declare_parameter("image_height", 480.0);
        this->declare_parameter("camera_frame", "camera_link");
        this->declare_parameter("target_frame", "base_link");
        this->declare_parameter("min_confidence", 0.5);
        
        // Get parameters
        detection_topic_ = this->get_parameter("detection_topic").as_string();
        cloud_topic_ = this->get_parameter("cloud_topic").as_string();
        assumed_depth_ = this->get_parameter("assumed_depth").as_double();
        camera_fov_ = this->get_parameter("camera_fov").as_double();
        image_width_ = this->get_parameter("image_width").as_double();
        image_height_ = this->get_parameter("image_height").as_double();
        camera_frame_ = this->get_parameter("camera_frame").as_string();
        target_frame_ = this->get_parameter("target_frame").as_string();
        min_confidence_ = this->get_parameter("min_confidence").as_double();
        
        // Calculate camera intrinsics from FOV
        // focal_length = (image_width / 2) / tan(fov/2)
        double fov_rad = camera_fov_ * M_PI / 180.0;
        fx_ = (image_width_ / 2.0) / std::tan(fov_rad / 2.0);
        fy_ = fx_; // Assume square pixels
        cx_ = image_width_ / 2.0;
        cy_ = image_height_ / 2.0;
        
        // Initialize TF2
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        
        // Subscribers and Publishers
        detection_sub_ = this->create_subscription<vision_msgs::msg::Detection2DArray>(
            detection_topic_, 10,
            std::bind(&BBox2CostmapNode::detection_callback, this, std::placeholders::_1)
        );
        
        cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            cloud_topic_, 10
        );
        
        RCLCPP_INFO(this->get_logger(), "BBox2Costmap Node initialized");
        RCLCPP_INFO(this->get_logger(), "Detection topic: %s", detection_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "Cloud topic: %s", cloud_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "Assumed depth: %.2f m", assumed_depth_);
        RCLCPP_INFO(this->get_logger(), "Camera FOV: %.1f degrees", camera_fov_);
        RCLCPP_INFO(this->get_logger(), "Camera intrinsics: fx=%.1f, fy=%.1f, cx=%.1f, cy=%.1f", 
                   fx_, fy_, cx_, cy_);
    }

private:
    void detection_callback(const vision_msgs::msg::Detection2DArray::SharedPtr msg)
    {
        // Create point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        cloud->header.frame_id = camera_frame_;
        cloud->header.stamp = pcl_conversions::toPCL(msg->header.stamp);
        
        // Process each detection
        for (const auto& detection : msg->detections)
        {
            // Skip low confidence detections
            if (!detection.results.empty())
            {
                double confidence = detection.results[0].hypothesis.score;
                if (confidence < min_confidence_) {
                    continue;
                }
            }
            
            // Get bounding box center
            double bbox_center_x = detection.bbox.center.position.x;
            double bbox_center_y = detection.bbox.center.position.y;
            double bbox_width = detection.bbox.size_x;
            double bbox_height = detection.bbox.size_y;
            
            // Convert 2D bbox center to 3D point
            pcl::PointXYZ point = pixel_to_3d_point(bbox_center_x, bbox_center_y, assumed_depth_);
            cloud->points.push_back(point);
            
            // Optionally add more points around the bbox for better obstacle representation
            // Add points at bbox corners for more robust obstacle marking
            std::vector<std::pair<double, double>> bbox_corners = {
                {bbox_center_x - bbox_width/4, bbox_center_y - bbox_height/4},  // Top-left quarter
                {bbox_center_x + bbox_width/4, bbox_center_y - bbox_height/4},  // Top-right quarter
                {bbox_center_x - bbox_width/4, bbox_center_y + bbox_height/4},  // Bottom-left quarter
                {bbox_center_x + bbox_width/4, bbox_center_y + bbox_height/4}   // Bottom-right quarter
            };
            
            for (const auto& corner : bbox_corners)
            {
                pcl::PointXYZ corner_point = pixel_to_3d_point(corner.first, corner.second, assumed_depth_);
                cloud->points.push_back(corner_point);
            }
        }
        
        cloud->width = cloud->points.size();
        cloud->height = 1;
        cloud->is_dense = true;
        
        // Convert to ROS message and publish
        sensor_msgs::msg::PointCloud2 cloud_msg;
        pcl::toROSMsg(*cloud, cloud_msg);
        cloud_msg.header.frame_id = camera_frame_;
        cloud_msg.header.stamp = msg->header.stamp;
        
        cloud_pub_->publish(cloud_msg);
        
        if (!cloud->points.empty()) {
            RCLCPP_DEBUG(this->get_logger(), 
                        "Published point cloud with %zu points from %zu detections", 
                        cloud->points.size(), msg->detections.size());
        }
    }
    
    pcl::PointXYZ pixel_to_3d_point(double u, double v, double depth)
    {
        // Convert pixel coordinates to 3D point using camera intrinsics
        // Standard pinhole camera model: X = (u - cx) * depth / fx
        double x = (u - cx_) * depth / fx_;
        double y = (v - cy_) * depth / fy_;
        double z = depth;
        
        pcl::PointXYZ point;
        point.x = z;  // Forward (camera Z is forward)
        point.y = -x; // Left (camera X is right, robot Y is left)
        point.z = -y; // Up (camera Y is down, robot Z is up)
        
        return point;
    }
    
    // Parameters
    std::string detection_topic_;
    std::string cloud_topic_;
    double assumed_depth_;
    double camera_fov_;
    double image_width_;
    double image_height_;
    std::string camera_frame_;
    std::string target_frame_;
    double min_confidence_;
    
    // Camera intrinsics
    double fx_, fy_, cx_, cy_;
    
    // ROS components
    rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr detection_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
    
    // TF2
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<BBox2CostmapNode>();
    
    RCLCPP_INFO(node->get_logger(), "Starting BBox2Costmap node...");
    
    try {
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(node->get_logger(), "Exception in BBox2Costmap node: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
} 