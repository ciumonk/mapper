/// Author: HP

#include <stdio.h>
#include <iostream>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/opencv.hpp>
#include "ceres/ceres.h"
#include "inliers.h"

using namespace std;
using namespace cv;

typedef Eigen::MatrixXd MatE;
typedef Eigen::VectorXd Vec;
typedef Eigen::Matrix<double, 3, 3> Mat3;
typedef Eigen::Matrix<double, 2, 1> Vec2;
typedef Eigen::Vector3d Vec3;

namespace Mapper {
    void removeByDistance(vector<DMatch> &matches, vector<DMatch> &goodMatches, float ratio) {
      // Compute distance threshold as (Average distance between all matched pairs)*ratio
      float average = std::accumulate(matches.begin(), matches.end(), 0.0f,
                                      [](const float &a, DMatch b) { return a + b.distance; }) / matches.size();
      average = average * ratio;
      // Keep only matches that are within the threshold
      for_each(matches.begin(), matches.end(), [&](DMatch el) {
          if (el.distance < average)
            goodMatches.push_back(el);
      });
    }

    // keep matches that appear in both L-R check and R-L check
    void removeByAssymetry(vector<DMatch> &matchesLR, vector<DMatch> &matchesRL, vector<DMatch> &goodMatches) {
      for (vector<DMatch>::iterator itLR = matchesLR.begin(); itLR != matchesLR.end(); ++itLR) {
        for (vector<DMatch>::iterator itRL = matchesRL.begin(); itRL != matchesRL.end(); ++itRL) {
          if ((*itLR).queryIdx == (*itRL).trainIdx &&
              (*itRL).queryIdx == (*itLR).trainIdx) {
            goodMatches.push_back(*itLR);
            break;
          }
        }
      }
    }

    // Remove outliers using the fundamental matrix
    // Use iterations to set the number of passes
    Mat removeRansac(vector<DMatch> &inMatches, const vector<KeyPoint> &kL, const vector<KeyPoint> &kR,
                     vector<DMatch> &outMatches, int iterations) {
      Mat fundamental;
      vector<DMatch> goodMatches;
      vector<Point2f> kL2f, kR2f;
      outMatches = inMatches;
      for (int i = iterations; i >= 1; i--) {
        kL2f.clear();
        kR2f.clear();
        // convert DMatches to vecPoint2f
        for (vector<DMatch>::const_iterator it = outMatches.begin(); it != outMatches.end(); ++it) {
          kL2f.push_back(kL[it->queryIdx].pt);
          kR2f.push_back(kR[it->trainIdx].pt);
        }
        // If enough points for 8-pt algo or RANSAC
        if (kL2f.size() > 8 && kR2f.size() > 8) {
          vector<uchar> inliers(kL2f.size(), 0);
          cout << (iterations > 1 && i == 1 ? "8POINT" : "RANSAC") << endl;
          fundamental = findFundamentalMat(Mat(kL2f), Mat(kR2f),
                                           iterations > 1 && i == 1 ? CV_FM_8POINT : CV_FM_RANSAC, //8-pt for final pass
                                           3.0,   // RANSAC max dist
                                           0.99f, // RANSAC confidence level
                                           inliers);

          // Use the inlier mask to select good matches
          vector<DMatch>::const_iterator itMatch = outMatches.begin();
          for (vector<uchar>::const_iterator itInlier = inliers.begin();
               itInlier != inliers.end(); ++itInlier, ++itMatch) {
            if (*itInlier) {
              goodMatches.push_back(*itMatch);
            }
          }
          outMatches = goodMatches;
          goodMatches.clear();
        }
      }
      return fundamental;
    }

    // Threshold matches by distance, symmetry and Fundamental matrix
    Mat extractInliers(vector<DMatch> &matchesL, vector<DMatch> &matchesR, vector<KeyPoint> &kL, vector<KeyPoint> &kR,
                       vector<DMatch> &matches, int iterations, float ratio) {
      vector<DMatch> goodMatches;
      vector<DMatch> matchesLR;
      vector<DMatch> matchesRL;
      // Prune matches with distance very different than average distance
      removeByDistance(matchesL, matchesLR, ratio);
      removeByDistance(matchesR, matchesRL, ratio);
      // Prune asymmetric matches
      removeByAssymetry(matchesLR, matchesRL, goodMatches);
      // Finally use the F matrix to mask inliers
      return removeRansac(goodMatches, kL, kR, matches, iterations);
    }
} //namespace Mapper

// Ceres options, ad-hoc
EstimatorOptions::EstimatorOptions(void) :
        max_num_iterations(100),
        threshold_distance(1e-10) { }

namespace CeresSolver {
    // Compute Sampson distance
    // Ideally yFx should be 0
    double PtsDistance(const Mat3 &F, const Vec2 &x1, const Vec2 &x2) {
      Vec3 x(x1(0), x1(1), 1.0);
      Vec3 y(x2(0), x2(1), 1.0);

      Vec3 Fx = F * x;
      Vec3 FTy = F.transpose() * y;
      double yFx = y.dot(Fx);
      double sq = (yFx) * (yFx);
      double normX = Fx.head<2>().squaredNorm();
      double normY = FTy.head<2>().squaredNorm();
      return sq * (1 / normX + 1 / normY);
      //return  sq / (normX + normY);
    }

    // Residuals are given by how different x`Fx is from 0
    class FundamentalCostFunctor {
    public:
        FundamentalCostFunctor(const Vec2 &x, const Vec2 &y) : x_(x), y_(y) { }

        template<typename T>
        bool operator()(const T *fundamental_parameters, T *residuals) const {
          typedef Eigen::Matrix<T, 3, 3> Mat3;
          typedef Eigen::Matrix<T, 3, 1> Vec3;

          Mat3 F(fundamental_parameters);

          Vec3 x(T(x_(0)), T(x_(1)), T(1.0));
          Vec3 y(T(y_(0)), T(y_(1)), T(1.0));

          Vec3 Fx = F * x;
          Vec3 FTy = F.transpose() * y;
          T yFx = y.dot(Fx);
          // Sampson
          // TODO(Horia) try with one summed residual
          residuals[0] = yFx * T(1) / Fx.head(2).norm();
          residuals[1] = yFx * T(1) / FTy.head(2).norm();

          return true;
        }

        const MatE x_;
        const MatE y_;
    }; // END class FundamentalCostFunctor

    // Used to check if solver has converged
    class TerminationCheckingCallback : public ceres::IterationCallback {
    public:
        TerminationCheckingCallback(const MatE &x1, const MatE &x2,
                                    const EstimatorOptions &options,
                                    Mat3 *F)
                : options_(options), x1_(x1), x2_(x2), F_(F) { }

        virtual ceres::CallbackReturnType operator()(
                const ceres::IterationSummary &summary) {

          if (!summary.step_is_successful) {
            return ceres::SOLVER_CONTINUE;
          }

          // Calculate average of symmetric epipolar distance.
          // TODO(Horia) change PtsDistance so that it takes whole Mats and avoid this multi-call nonsense
          double avg_distance = 0.0;
          for (int i = 0; i < x1_.rows(); i++) {
            avg_distance += CeresSolver::PtsDistance(*F_, x1_.row(i), x2_.row(i));
          }
          avg_distance /= x1_.rows();
          cout << "AVGDIST: " << avg_distance << endl;
          if (avg_distance <= options_.threshold_distance) {
            return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
          }
          return ceres::SOLVER_CONTINUE;
        }

    private:
        const EstimatorOptions &options_;
        const MatE &x1_;
        const MatE &x2_;
        Mat3 *F_;
    }; // END class TerminationCheckingCallback

    bool EstimateFundamental(const MatE &x1, const MatE &x2,
                             const EstimatorOptions &options,
                             Mat3 *F) {
      ceres::Problem problem;
      for (int i = 0; i < x1.rows(); i++) {
        FundamentalCostFunctor *F_cost_function = new FundamentalCostFunctor(x1.row(i),
                                                                             x2.row(i));
        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction< // automatic differentiation
                        FundamentalCostFunctor,
                        2,  // num_residuals
                        9   // Number of parameters in block 0.
                >(F_cost_function),
                NULL, //no LOSS function
                F->data());
      }

      // More or less ad-hoc Ceres options
      ceres::Solver::Options solver_options;
      // ceres::DENSE_QR for small problems
      solver_options.linear_solver_type = ceres::DENSE_QR;
      // TODO(Horia) check docs on options
      solver_options.max_num_iterations = options.max_num_iterations;
      solver_options.update_state_every_iteration = true;

      // Termination check
      TerminationCheckingCallback callback(x1, x2, options, F);
      solver_options.callbacks.push_back(&callback);

      ceres::Solver::Summary summary;
      ceres::Solve(solver_options, &problem, &summary);

      cout << "Summary:\n" << summary.FullReport();
      cout << "Final refined matrix:\n" << *F;

      return summary.IsSolutionUsable();
    }
}  // namespace CeresSolver


// Draw matches side-by-side&return image
Mat drawImagesTogetherAndShow(Mat &imageL, Mat &imageR) {
  Size sizeL = imageL.size();
  Size sizeR = imageR.size();
  Mat image(sizeL.height, sizeL.width + sizeR.width, CV_8UC1);
  //TODO(Horia) pad if heights are mismatched
  imageL.copyTo(image(Rect(0, 0, sizeL.width, sizeL.height)));
  imageR.copyTo(image(Rect(sizeL.width, 0, sizeR.width, sizeR.height)));
  imshow("IMAGES", image);
  waitKey(0);
  return image;
}

int main(int argc, char **argv) {
  Mat img_object;
  Mat img_scene;
  //img_object = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
  //img_scene = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );
  img_scene = imread("image_left.png", CV_LOAD_IMAGE_GRAYSCALE);
  img_object = imread("image_right.png", CV_LOAD_IMAGE_GRAYSCALE);

  if (!img_object.data || !img_scene.data) {
    std::cout << "error reading images" << std::endl;
    return -1;
  }
  //drawImagesTogetherAndShow(img_scene, img_object);

  // set-up ORB feature detector, default values for now
  // params are: num_features, image pyr scale, nr pyrs, edge threshold, first level, WTA_K, feat.rank score, patchsize
  cv::ORB orb;
  cv::OrbFeatureDetector detector(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31);
  std::vector<KeyPoint> keypoints_object, keypoints_scene;
  // detect kps
  detector.detect(img_object, keypoints_object);
  detector.detect(img_scene, keypoints_scene);
  cout << "Number of keypoints detected obj:" << keypoints_object.size() << std::endl;
  cout << "Number of keypoints detected scene:" << keypoints_scene.size() << std::endl;
  // set-up feat. extractor
  cv::OrbDescriptorExtractor extractor;
  Mat descriptors_object, descriptors_scene;
  // extract kps
  extractor.compute(img_object, keypoints_object, descriptors_object);
  extractor.compute(img_scene, keypoints_scene, descriptors_scene);

  // BruteForce matcher using Hamming distance; use NORM_HAMMING2 if WTA_K>=3
  cv::BFMatcher matcher(cv::NORM_HAMMING);
  std::vector<DMatch> matchesL;
  std::vector<DMatch> matchesR;
  // match both L-R and R-L
  matcher.match(descriptors_object, descriptors_scene, matchesL);
  matcher.match(descriptors_scene, descriptors_object, matchesR);
  cout << "Number of keypoints matched L:" << matchesL.size() << std::endl;
  cout << "Number of keypoints matched R:" << matchesR.size() << std::endl;

  vector<DMatch> good_matches;
  Mat F;
  // extract good matches and save seed F
  F = Mapper::extractInliers(matchesL, matchesR, keypoints_object, keypoints_scene, good_matches, 1, 1.2f);

  if (good_matches.size() < 4) {
    cout << "Not enough good matches remaining!" << endl;
    return 0;
  }
  cout << "Number of good matches:" << good_matches.size() << std::endl;
  cout << "Number of good keypoints:" << keypoints_object.size() << std::endl;

  vector<Point2f> kL2f, kR2f;
  // extract keypoints corresponding to good matches
  for (vector<DMatch>::const_iterator it = good_matches.begin(); it != good_matches.end(); ++it) {
    kL2f.push_back(keypoints_object[it->queryIdx].pt);
    kR2f.push_back(keypoints_scene[it->trainIdx].pt);
  }

  // initialize Eigen mat with seed F matrix
  //TODO(Horia) does cv2eigen copy data or just headers+pointer to data?
  //TODO(Horia) try with Eigen::Map
  Mat3 estimated_matrix;
  cv2eigen(F, estimated_matrix);

  // Ceres options
  EstimatorOptions options;
  //options.threshold_distance = 1;

  // converting cv::Mats to Eigen::Mats
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> kL_E;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> kR_E;
  // here cv::Mat is 2-channel 1-column, so reshape to 1-channel 2-column
  cv2eigen(Mat(kL2f).reshape(1), kL_E);
  cv2eigen(Mat(kR2f).reshape(1), kR_E);
  // estimate F
  bool converged = CeresSolver::EstimateFundamental(kL_E, kR_E, options, &estimated_matrix);
  cout << endl << "Converged: " << (converged ? "TRUE" : "FALSE") << endl;
  cout << endl << "Results" << endl;
  cout << "F old:" << F << endl;
  cout << "F new:" << estimated_matrix << endl;

  std::vector<Point2f> good_obj;
  std::vector<Point2f> good_scene;
  good_matches.clear();
  int matchCount = 0;
  double avg_distance = 0.0;

  // Remove outliers using F
  // distance threshold set to 0.9 by eyeballing, find better metric than average distance
  // TODO(Horia) faster way to get inliers using F? Similar method used in OpenCV findFundMat so must investigate
  for (int i = 0; i < kL_E.rows(); i++) {
    double dist = CeresSolver::PtsDistance(estimated_matrix, kL_E.row(i), kR_E.row(i));
    avg_distance += dist;
    if (dist < 0.9f) {
      good_obj.push_back(Point2f(kL_E.row(i)(0), kL_E.row(i)(1)));
      good_scene.push_back(Point2f(kR_E.row(i)(0), kR_E.row(i)(1)));
      // quick way of rebuilding the DMatch vector
      //TODO(Horia) find a better way to rebuild good_matches
      good_matches.push_back(DMatch(matchCount, matchCount, 0, 1.0f));
      matchCount++;
    }
  }
  avg_distance /= kL_E.rows();
  cout << "Average distance is: " << avg_distance << endl;
  cout << "Matches remaining: " << good_obj.size() << endl;

  Mat img_matches;
  // convert Point2f to Keypoint
  std::vector< KeyPoint > keypointsL,keypointsR;
  cv::KeyPoint::convert	(	good_obj, keypointsL);
  cv::KeyPoint::convert	(	good_scene, keypointsR);
  // draw matches
  drawMatches(img_object, keypointsL, img_scene, keypointsR,
              good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
              vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

  // find homography matrix using newest inliers
  Mat H = findHomography(good_obj, good_scene, CV_RANSAC);

  // outline of left image
  std::vector<Point2f> obj_corners(4);
  obj_corners[0] = Point(0, 0);
  obj_corners[1] = Point(img_object.cols, 0);
  obj_corners[2] = Point(img_object.cols, img_object.rows);
  obj_corners[3] = Point(0, img_object.rows);
  std::vector<Point2f> scene_corners(4);

  // warp outline of left image
  perspectiveTransform(obj_corners, scene_corners, H);

  // draw outline of warped left image
  Point2f offset((float) img_object.cols, 0);
  line(img_matches, scene_corners[0] + offset, scene_corners[1] + offset, Scalar(0, 255, 0), 4);
  line(img_matches, scene_corners[1] + offset, scene_corners[2] + offset, Scalar(0, 255, 0), 4);
  line(img_matches, scene_corners[2] + offset, scene_corners[3] + offset, Scalar(0, 255, 0), 4);
  line(img_matches, scene_corners[3] + offset, scene_corners[0] + offset, Scalar(0, 255, 0), 4);

  // show matches, then warped left image
  resize(img_matches, img_matches, Size(1280, 640));
  imshow("Good Matches & Object detection", img_matches);
  waitKey(0);
  warpPerspective(img_object, img_object, H, img_object.size());
  drawImagesTogetherAndShow(img_object, img_scene);
  waitKey(0);

  return 0;
}
