#ifndef MAPPER_INLIERS_H
#define MAPPER_INLIERS_H

struct EstimatorOptions {
    EstimatorOptions(void);
    int max_num_iterations;
    double threshold_distance;
};
#endif //MAPPER_INLIERS_H
