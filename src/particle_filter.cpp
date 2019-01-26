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
using std::default_random_engine;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * Add random Gaussian noise to each particle.
   */
  this->num_particles = 100;

  default_random_engine gen{};
  normal_distribution<> dist_x{x, std[0]};
  normal_distribution<> dist_y{y, std[1]};
  normal_distribution<> dist_theta{theta, std[2]};

  this->particles.resize(static_cast<unsigned long>(this->num_particles));
  for (int i = 0; i < this->num_particles; ++i) {
    const Particle newParticle {i, dist_x(gen), dist_y(gen), dist_theta(gen), 1.0};
    this->particles.push_back(newParticle);
  }

  this->is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   */
  default_random_engine gen{};

  // Initialize gaussian distribution for noise
  normal_distribution<> noise_x{0, std_pos[0]};
  normal_distribution<> noise_y{0, std_pos[1]};
  normal_distribution<> noise_theta{0, std_pos[2]};

  // Predict each particle's new state using bicycle motion model
  for (int i = 0; i < this->num_particles; ++i) {
    Particle &p = this->particles[i];

    double v_to_theta = velocity / yaw_rate;

    if (fabs(yaw_rate) >= 0.00001) {
      p.x += p.x + (v_to_theta * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta)));
      p.y += p.y + (v_to_theta * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t)));
      p.theta += p.theta + yaw_rate * delta_t;
    } else {
      p.x += velocity * delta_t * cos(p.theta);
      p.y += velocity * delta_t * sin(p.theta);
    }

    // Add gaussian noise
    p.x += noise_x(gen);
    p.y += noise_y(gen);
    p.theta += noise_theta(gen);
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
  for (auto obs : observations) {
    LandmarkObs *nearest_mark = nullptr;
    double nearest_dist = 9999999.0;

    for (auto pred : predicted) {
      double current_dist = dist(nearest_mark->x, nearest_mark->y, obs.x, obs.y);
      if (current_dist < nearest_dist) {
        nearest_mark = &pred;
        nearest_dist = current_dist;
      }
    }

    if (nearest_mark != nullptr) {
      // Assign ID of nearest landmark from the map to each observation
      obs.id = nearest_mark->id;
    }
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
    t_observations.resize(observations.size());
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
      double mark_x = 0.0;
      double mark_y = 0.0;

      for (auto map_landmark : map_landmarks.landmark_list) {
        if (map_landmark.id_i == obs.id) {
          mark_x = map_landmark.x_f;
          mark_y = map_landmark.y_f;
          break;
        }
      }

      final_weight *= multivariate_gaussian(std_landmark[0], std_landmark[1], obs.x, obs.y, mark_x, mark_y);
    }
    this->particles[i].weight = final_weight;
  }
}

void ParticleFilter::resample() {
  /**
   * Resample particles with replacement with probability proportional
   *   to their weight.
   */
  default_random_engine gen{};
  uniform_int_distribution<> dist_index{0, this->num_particles};

  int index = dist_index(gen);
  double beta = 0.0;

  vector<Particle> resampled_particles {};

  for (int i = 0; i < this->num_particles; ++i) {
    double max_weight = 0.0;
    for (auto p : this->particles) {
      if (p.weight > max_weight) {
        max_weight = p.weight;
      }
    }

    uniform_real_distribution<> dist_beta{0, max_weight};
    beta += dist_beta(gen);

    double p_weight = this->particles[index].weight;
    while (p_weight < beta) {
      beta -= p_weight;
      index = (index + 1) % this->num_particles;
    }
    resampled_particles.push_back(this->particles[index]);
  }
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