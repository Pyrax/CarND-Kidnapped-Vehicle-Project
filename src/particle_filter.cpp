/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::uniform_int_distribution;
using std::uniform_real_distribution;
using std::numeric_limits;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * Add random Gaussian noise to each particle.
   */
  this->num_particles = 100;

  normal_distribution<> dist_x{x, std[0]};
  normal_distribution<> dist_y{y, std[1]};
  normal_distribution<> dist_theta{theta, std[2]};

  this->particles.reserve(static_cast<unsigned long>(this->num_particles));
  for (int i = 0; i < this->num_particles; ++i) {
    const Particle newParticle {
      i,
      dist_x(this->random_engine),
      dist_y(this->random_engine),
      dist_theta(this->random_engine),
      1.0
    };
    this->particles.push_back(newParticle);
  }

  this->is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   */
  // Initialize gaussian distribution for noise
  normal_distribution<> noise_x{0, std_pos[0]};
  normal_distribution<> noise_y{0, std_pos[1]};
  normal_distribution<> noise_theta{0, std_pos[2]};

  // Predict each particle's new state using bicycle motion model
  for (int i = 0; i < this->num_particles; ++i) {
    Particle &p = this->particles[i];

    double v_to_theta = velocity / yaw_rate;

    if (fabs(yaw_rate) >= 0.00001) {
      p.x += v_to_theta * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
      p.y += v_to_theta * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
      p.theta += yaw_rate * delta_t;
    } else {
      p.x += velocity * delta_t * cos(p.theta);
      p.y += velocity * delta_t * sin(p.theta);
    }

    // Add gaussian noise
    p.x += noise_x(this->random_engine);
    p.y += noise_y(this->random_engine);
    p.theta += noise_theta(this->random_engine);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   */
  // Nearest Neighbour search
  for (auto &obs : observations) {
    int nearest_id = -1;
    double nearest_dist = numeric_limits<double>::max();

    for (auto pred : predicted) {
      double current_dist = dist(pred.x, pred.y, obs.x, obs.y);
      if (current_dist < nearest_dist) {
        nearest_id = pred.id;
        nearest_dist = current_dist;
      }
    }

    // Assign ID of nearest landmark from the map to each observation
    obs.id = nearest_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * Update the weights of each particle using a multi-variate Gaussian
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   */
  // Update weights for each particle
  for (int i = 0; i < this->num_particles; ++i) {
    double p_x = this->particles[i].x;
    double p_y = this->particles[i].y;
    double p_theta = this->particles[i].theta;

    // Get prediction measurements for each particle which are within sensor range
    vector<LandmarkObs> predictions {};
    for (auto map_landmark : map_landmarks.landmark_list) {
      int mark_id = map_landmark.id_i;
      double mark_x = map_landmark.x_f;
      double mark_y = map_landmark.y_f;

      if (dist(p_x, p_y, mark_x, mark_y) <= sensor_range) {
        predictions.push_back(LandmarkObs{mark_id, mark_x, mark_y});
      }
    }

    // Convert observations from car coordinates to map coordinates
    vector<LandmarkObs> t_observations {};
    t_observations.reserve(observations.size());
    for (auto obs : observations) {
      t_observations.push_back(LandmarkObs{
        obs.id,
        obs.x * cos(p_theta) - obs.y * sin(p_theta) + p_x,
        obs.x * sin(p_theta) + obs.y * cos(p_theta) + p_y,
      });
    }

    // Associate observations with nearest landmarks
    this->dataAssociation(predictions, t_observations);

    double final_weight = 1.0;
    for (auto obs : t_observations) {
      double pred_x = 0.0;
      double pred_y = 0.0;

      for (auto pred : predictions) {
        if (pred.id == obs.id) {
          pred_x = pred.x;
          pred_y = pred.y;
          break;
        }
      }

      final_weight *= multivariate_gaussian(std_landmark[0], std_landmark[1], obs.x, obs.y, pred_x, pred_y);
    }
    this->particles[i].weight = final_weight;
  }
}

void ParticleFilter::resample() {
  /**
   * Resample particles with replacement with probability proportional
   *   to their weight.
   */
  uniform_int_distribution<> dist_index{0, this->num_particles-1};

  int index = dist_index(this->random_engine);
  double beta = 0.0;
  double max_weight = this->getMaxParticleWeight();

  uniform_real_distribution<> dist_beta{0, 2.0 * max_weight};
  vector<Particle> resampled_particles {};

  // Use resampling wheel algorithm
  for (int i = 0; i < this->num_particles; ++i) {
    beta += dist_beta(this->random_engine);

    while (this->particles[index].weight < beta) {
      beta -= this->particles[index].weight;
      index = (index + 1) % this->num_particles;
    }
    resampled_particles.push_back(this->particles[index]);
  }

  this->particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

const double ParticleFilter::getMaxParticleWeight() const {
  double max_weight = 0.0;

  for (const auto &p : this->particles) {
    if (p.weight > max_weight) {
      max_weight = p.weight;
    }
  }
  return max_weight;
}
