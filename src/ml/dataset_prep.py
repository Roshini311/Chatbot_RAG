from typing import List, Tuple

CATEGORIES = [
    "Artificial Intelligence",
    "Cyber Security",
    "Cloud Computing",
    "Robotics",
    "Data Science"
]

SAMPLE_DATASET = [
    # Artificial Intelligence
    ("Neural networks, deep learning, LLMs, fine-tuning, transformers, natural language processing, reinforcement learning.", 0),
    ("Machine learning models, gradient descent, prompt engineering, agentic AI, computer vision, backpropagation.", 0),
    ("Large language models, generative AI, RAG retrieval, vector search, embedding models, attention mechanism.", 0),
    
    # Cyber Security
    ("Penetration testing, encryption, zero trust architecture, firewall, malware analysis, vulnerability management.", 1),
    ("Network security, threat detection, cryptography, public key infrastructure, DDoS mitigation, security operations.", 1),
    ("Identity access management, SIEM logging, endpoint protection, ransomware defense, incident response.", 1),

    # Cloud Computing
    ("Amazon Web Services AWS, Azure cloud, Google Cloud GCP, Kubernetes clusters, Docker containers, serverless lambda.", 2),
    ("DevOps pipelines, infrastructure as code, Terraform, microservices architecture, load balancing, virtual machines.", 2),
    ("Cloud storage buckets, multi-cloud strategy, VPC networking, auto-scaling groups, cloud observability.", 2),

    # Robotics
    ("Robotic arms, kinematics, autonomous navigation, ROS robot operating system, actuators, lidar sensors, control systems.", 3),
    ("Industrial automation, mobile robots, slam mapping, path planning, servos, humanoid robotics, inverse kinematics.", 3),
    ("Mechatronics design, sensor fusion, drone navigation, telemetry control, motor drivers, robotic perception.", 3),

    # Data Science
    ("Exploratory data analysis, pandas dataframe, statistical inference, feature engineering, data cleaning, visualization.", 4),
    ("Hypothesis testing, regression analysis, random forest, correlation matrix, data pipeline, confusion matrix.", 4),
    ("Bi tool dashboards, data warehousing, ETL pipelines, clustering analysis, big data analytics, SQL queries.", 4)
]

def get_training_data() -> Tuple[List[str], List[int]]:
    texts = [item[0] for item in SAMPLE_DATASET]
    labels = [item[1] for item in SAMPLE_DATASET]
    return texts, labels
