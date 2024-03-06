

import pygame
import random
import math
from math import pi
from math import atan2
from math import sin
from math import cos
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools
from operator import attrgetter
import pandas as pd
from sklearn.cluster import MeanShift
import matplotlib.cm as cm
import os


AUTO_PRED = False
PRED_DRAW = True
ARROW_KEYS = False
PRED_FOV = False

cursor_x, cursor_y = 0, 0


PRED_VIEWING_RANGE = pi/3  # Irrelevent for auto_sim


# THESE MUST ALL BE INT VALUES
FPS = 60  # for player simulations, capped at 60 - if doing auto sim can use 30 or probably 15
TIMESTEPS = 20*FPS
NUM_GENS = 40


PRED_PAUSE_TIME = round(0.5 * FPS, 1)  # 1 * FPS = 1 second

pygame.init()
current_width, current_height = pygame.display.Info(
).current_w, pygame.display.Info().current_h
pygame.quit()

# Calculate scaling factor


def scale_dimensions(current_width, current_height):

    # Calculate the scaling factor based on the current resolution compared to the desired resolution.

    scale_x = current_width / 800
    scale_y = current_height / 800
    return min(scale_x, scale_y)


scaling_factor = scale_dimensions(current_width, current_height)
WIDTH, HEIGHT = 800*scaling_factor, 800 * \
    scaling_factor  # 0,0 is at top left!!!!!!

SCALE = 10 * scaling_factor

BOID_TURNING = pi/FPS
PRED_TURNING = (3/2) * BOID_TURNING
BOID_ALPHA = pi/6
BOID_SPEED = 120 / FPS * scaling_factor
PRED_SPEED = 1.5 * BOID_SPEED


N_BOIDS = 50
BOID_RADIUS = 1*SCALE
PRED_RADIUS = 1.5*BOID_RADIUS

# need to be calculated as areas
SEPARATION_AREA = pi * (BOID_RADIUS * 1)**2
ALIGNMENT_AREA = pi * (BOID_RADIUS * 10)**2
COHESION_AREA = pi * (BOID_RADIUS * 15)**2
PREDATOR_FLEE_AREA = pi * (BOID_RADIUS * 15)**2
PRED_RANGE = BOID_RADIUS * 15  # this is the pred radius
NOISE_WEIGHT = 0.25


# even spacing where midpoint is 1 under recipricol
potential_a_vals = [0.5, 0.6, 0.7, 0.8,
                    0.9, 1., 1.1111, 1.25, 1.429, 1.6667, 2]


########## GENERAL FUNCTIONS #########

def a_list_decoder(input):
    for i, number in enumerate(potential_a_vals):
        if input == number:
            return i/10


def write_new_folder(foldername):
    # Function to create a new folder in the current directory
    # Foldername is a string

    # Get the current working directory
    current_directory = os.getcwd()

    # Combine the current directory with the foldername to form the new directory path
    new_directory = os.path.join(current_directory, foldername)

    flag = False
    i = 1
    while flag == False:
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)
            flag = True
        else:
            new_directory += str(i)
        i += 1
    return new_directory + str('/')


def scatter_hist(x, y, ax, ax_histx, ax_histy, filepath):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    n_points = len(x)
    alpha = 10/n_points

    # the scatter plot:
    ax.scatter(x, y, alpha=alpha)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('Predator Weighting')
    plt.ylabel('Alignment Weighting')

    # now determine nice limits by hand:
    binwidth = 0.0999999
    xymax = 1.099999
    bins = np.arange(0, xymax, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')
    plt.draw()
    plt.savefig(filepath)


def moving_average(data, window_size):
    # Define the weights for the moving average
    weights = np.ones(window_size) / window_size
    # Use numpy's convolution function to calculate the moving average
    return np.convolve(data, weights, mode='valid')


# Pred Class
class Predator:
    def __init__(self, x, y, W, H, auto=False, success_bool=True):
        # Success = True adds diminishing returns when hunting groups
        self.x = x
        self.y = y
        self.angle = (random.random()-0.5)*2*pi  # used for bearing movement
        self.blind_angle = 2*pi/3  # bigger so that the vision is a cone
        self.auto = auto
        self.init_delay = round(4.5*FPS, 0)

        self.width = W
        self.height = H

        # this needs to be called after eating to reset speed to what it was before
        self.const_speed = PRED_SPEED
        self.speed = self.const_speed
        self.turning_radius = PRED_TURNING

        self.radius = PRED_RADIUS
        self.eating_bool = False
        self.time_to_eat = round(PRED_PAUSE_TIME, 0)
        self.success_bool = success_bool
        self.near_count = 0

    def pi_range(self, angle):
        # ensures all angles are between -pi and pi, maybe should be a while loop but who cares

        if angle < -math.pi:
            angle += 2*math.pi
        if angle > math.pi:
            angle -= 2*math.pi
        return (angle)

    def success_calc(self, near_count):
        # more efficient and similar end result to calculate near_count individually for each boid, similar becuse they will be in roughly the same position
        if self.success_bool:
            return math.e**(-near_count/(N_BOIDS/3))
        else:
            return 1

    def eating_init(self):
        self.eating_bool = True
        self.speed = 0
        self.eating_counter = self.time_to_eat

    def eating_func(self):
        self.eating_counter -= 1
        if self.eating_counter < 0:
            self.eating_bool = False
            self.speed = self.const_speed

    def move(self, population):

        if self.auto:
            distance = self.height**2 + self.width**2
            dx = self.width
            dy = self.height

            for boid in population:
                test_dx = boid.x - self.x
                test_dy = boid.y - self.y

                d_check = math.sqrt((test_dx) ** 2 + (test_dy) ** 2)
                if d_check < distance:
                    distance = d_check
                    dx = test_dx
                    dy = test_dy

            if distance == 0:
                distance += 0.00000000000001
            self.angle = atan2(dy, dx)
            # check to make sure the distance is standardised

            self.x += (dx / distance) * self.speed
            self.y += (dy / distance) * self.speed

        else:
            cursor_x, cursor_y = pygame.mouse.get_pos()
            dx = cursor_x - self.x
            dy = cursor_y - self.y

            target_angle = atan2(dy, dx)

            angle_diff = self.pi_range(target_angle-self.angle)

            if abs(angle_diff) <= self.turning_radius:
                self.angle = target_angle
            else:
                self.angle += np.sign(angle_diff)*self.turning_radius

            self.angle = self.pi_range(self.angle)

            self.x += cos(self.angle)*self.speed
            self.y += sin(self.angle)*self.speed

            if self.x < 0:
                self.x = 0
            if self.x > WIDTH:
                self.x = WIDTH
            if self.y < 0:
                self.y = 0
            if self.y > HEIGHT:
                self.y = HEIGHT

    def draw(self, screen):

        if ARROW_KEYS or self.auto:
            pygame.draw.circle(screen, (255, 165, 0),
                               (int(self.x), int(self.y)), self.radius)
        else:

            arrow_length = self.radius
            # Calculate the points of the arrow shape
            arrow_points = [


                (self.x + arrow_length * math.cos(self.angle),
                 self.y + arrow_length * math.sin(self.angle)),

                (self.x + (arrow_length/3) * math.cos(self.angle - math.pi/2),
                    self.y + (arrow_length/3) * math.sin(self.angle - math.pi/2)),
                (
                    self.x + (arrow_length/3) *
                    math.cos(self.angle + math.pi/2),
                    self.y + (arrow_length/3) * math.sin(self.angle + math.pi/2))
            ]
            pygame.draw.polygon(screen, (255, 165, 0), arrow_points)

    def update(self, screen, population):
        self.near_count = 0

        if self.eating_bool:
            self.eating_func()
        self.move(population)
        if PRED_DRAW:
            self.draw(screen)
        self.init_delay -= 1
        if self.init_delay > 0:
            self.eating_init()


class Boid:
    def __init__(self, evolution_vars, WIDTH, HEIGHT, respawn=False):
        # evolution vars = list where first entry is 'a' - [a, pred_weight, alignment_weight_balance]
        self.evolution_vars = evolution_vars

        self.a = evolution_vars[0]  # vertical axis of ellipse
        self.b = 1/self.a  # horrizontal axis of ellipse

        self.pred_weight = evolution_vars[1]  # predator weighting var
        self.social_weights = (1-self.pred_weight)
        self.alignment_weight = evolution_vars[2] * self.social_weights
        self.cohesion_weight = (1 - evolution_vars[2]) * self.social_weights

        self.width = WIDTH
        self.height = HEIGHT
        self.speed = BOID_SPEED

        self.x = random.randint(0, self.width)  # random starting position
        self.y = random.randint(0, self.height)

        # heading, initially random
        self.angle = random.uniform(-math.pi, math.pi)
        self.turning_radius = BOID_TURNING
        self.alive = True
        self.time_of_death = np.NaN

        self.radius = BOID_RADIUS  # radius
        self.alpha = BOID_ALPHA  # blind angle behind - update for elipse

        self.area_factor = self.blind_area_addition_factor_calc()
        self.separation_area = SEPARATION_AREA
        self.alignment_area = self.area_factor * ALIGNMENT_AREA
        self.cohesion_area = self.area_factor * COHESION_AREA
        self.predator_flee_area = self.area_factor * PREDATOR_FLEE_AREA

        self.predator_timer = 0  # current counter for time near pred
        self.total_pred_time = 0  # total counter ^
        self.pred_distance = math.inf  # initial dist from pred

        self.tri_points = np.array([(0, 0), (0, 0), (0, 0)], dtype=tuple)
        self.near_count = 0  # used for success weighting shiz

    def generate_a(self):

        return random.choice(potential_a_vals)

    def generate_pred_weight(self):
        # 0<pred weight<1

        return round(random.random(), 1)
        # return 0

    def generate_alignment_weight(self):
        # 0 < alignment weight < 1
        return round(random.random(), 1)

    def blind_area_addition_factor_calc(self):
        # calculates the percentage of the total area of the elipse covered by the blind spot and works out the number you must multiply the standard areas by
        # to give each boid the same viewing area

        # since will always be behind, can calculate theta 1 and theta 2, distance from theta = 0 to each radius
        # see https://rechneronline.de/pi/elliptical-sector.php

        a = self.a
        b = self.b
        blind_angle = self.alpha

        if a > b:
            t1 = pi + (pi - blind_angle)/2
            t2 = pi + (pi + blind_angle)/2
            area_calc = ((a*b)/2) * (blind_angle - atan2(((b-a) * sin(2*t2)), (a+b + (b-a) * cos(2*t2)))
                                     + atan2(((b-a) * sin(2*t1)), (a+b + (b-a) * cos(2*t1))))

        else:
            t1 = pi - blind_angle/2
            t2 = pi + blind_angle/2
            area_calc = ((a*b)/2) * (blind_angle - atan2(((a-b) * sin(2*t2)), (a+b + (a-b) * cos(2*t2)))
                                     + atan2(((a-b) * sin(2*t1)), (a+b + (a-b) * cos(2*t1))))

        k = area_calc/pi

        return 1/(1-k)

    def ellipse_calc(self, x, y, area):
        # returns true if found within area else false

        new_x = x*cos(self.angle-pi/2) - y * \
            sin(self.angle-pi/2)
        new_y = x*sin(self.angle-pi/2) + y*cos(self.angle-pi/2)

        if ((new_x/self.a)**2 + (new_y/self.b)**2) < area/pi:
            return True
        else:
            return False

    def circle_calc(self, x, y, area):
        # returns true if found within area else false
        if ((x)**2 + (y)**2) < area/pi:
            return True
        else:
            return False

    def normalise(self, vec):
        x, y = vec[0:2]
        total = math.sqrt(x**2 + y**2)
        if total != 0:
            return [x/total, y/total]
        else:
            return [0, 0]

    def toroidal_boundaries(self, dx, dy):
        # toroidal boundaries with for distances
        if dx > WIDTH/2:
            dx -= WIDTH
        elif dx < -WIDTH/2:
            dx += WIDTH

        if dy > HEIGHT/2:
            dy -= HEIGHT
        elif dy < -HEIGHT/2:
            dy += HEIGHT
        return (dx, dy)

    def pi_range(self, angle):
        # ensures all angles are between -pi and pi

        if angle < -math.pi:
            angle += 2*math.pi
        if angle > math.pi:
            angle -= 2*math.pi
        return (angle)

    def angle_smoother(self, angle_diff):
        # impliments maximum turning radius per timestep
        angle_diff = self.pi_range(angle_diff)

        if abs(angle_diff) <= self.turning_radius:
            return (angle_diff)
        elif abs(angle_diff) > self.turning_radius:
            return (np.sign(angle_diff)*self.turning_radius)

    def blind_spot(self, dx, dy):
        bearing_of_neighbour = math.atan2(dy, dx)
        angle_diff = self.angle-bearing_of_neighbour
        angle_diff = abs(self.pi_range(angle_diff))
        if angle_diff <= math.pi - self.alpha:
            return (True)  # True if able to see the other particle
        else:
            return (False)

    def seen_by_pred(self, predator):
        # function used for drawing when the pred has FOV

        pred_x, pred_y = predator.x, predator.y
        pred_angle = predator.angle
        dx, dy = self.x - pred_x, self.y - pred_y
        bearing = math.atan2(dy, dx)  # bearing of prey in relation to predator
        angle_diff = abs(self.pi_range(pred_angle-bearing))
        if angle_diff < PRED_VIEWING_RANGE:  # this is the FOV of the predator
            return (True)
        elif math.sqrt(dx**2 + dy**2) < PRED_RANGE:
            return (True)

        else:
            return (False)

    def noise(self):
        # finds noise value in angle form, convert it to force vector, definition of trig -> normalised

        noise = self.pi_range(random.uniform(-pi, pi) + self.angle)

        return [cos(noise), sin(noise)]

    def move(self):

        self.x += self.speed * math.cos(self.angle)
        self.y += self.speed * math.sin(self.angle)
        if self.x < 0:
            self.x = WIDTH
        if self.x > WIDTH:
            self.x = 0
        if self.y < 0:
            self.y = HEIGHT
        if self.y > HEIGHT:
            self.y = 0

    def can_see(self, dx, dy, boid):
        # function for checking if an item is within a range and if so calculates the force generated

        distance = math.sqrt(dx**2 + dy**2)

        if distance != 0:
            # this is naughty coding but two boids on top of eachother will separate due to noise then separation force

            # if loop the different areas to see where it falls into

            # separation calculation
            # no bloind spot because it's personal space
            if self.circle_calc(dx, dy, self.separation_area):
                # negative since wants to move away
                self.sep_vec[0] -= dx/distance
                self.sep_vec[1] -= dy/distance
                self.sep_vec[2] += 1

            # alignment calculation
            elif self.ellipse_calc(dx, dy, self.alignment_area) and self.blind_spot(dx, dy):
                self.ali_vec[0] += math.cos(boid.angle)
                self.ali_vec[1] += math.sin(boid.angle)
                self.ali_vec[2] += 1

            # cohesion calculation
            elif self.ellipse_calc(dx, dy, self.cohesion_area) and self.blind_spot(dx, dy):
                self.coh_vec[0] += dx/distance
                self.coh_vec[1] += dy/distance
                self.coh_vec[2] += 1

    def boid_force_vec_calc(self, population):
        # calculates 3 triple vectors for each boid2boid behaviour
        self.sep_vec = np.zeros(3)
        self.ali_vec = np.zeros(3)
        self.coh_vec = np.zeros(3)

        for boid in population:
            if boid != self:
                dx = boid.x - self.x  # direction is towards neighbour
                dy = boid.y - self.y

                dx, dy = self.toroidal_boundaries(dx, dy)

                self.can_see(dx, dy, boid)

        if self.ali_vec[2] > 0:  # averaging alignment direction
            self.ali_vec[0] /= self.ali_vec[2]
            self.ali_vec[1] /= self.ali_vec[2]

        self.sep_vec[0:2] = self.normalise(self.sep_vec[0:2])
        self.ali_vec[0:2] = self.normalise(self.ali_vec[0:2])
        self.coh_vec[0:2] = self.normalise(self.coh_vec[0:2])

    def pred_force_vec_calc(self, predator):
        # checks to see if predator is in the viewing area and then creates force vector repelling
        self.pre_vec = np.zeros(3)
        dx = predator.x - self.x
        dy = predator.y - self.y
        self.pred_distance = math.sqrt(dx**2 + dy**2)

        if self.ellipse_calc(dx, dy, self.predator_flee_area) and self.blind_spot(dx, dy):
            self.pre_vec[0] -= dx
            self.pre_vec[1] -= dy
            self.pre_vec[2] += 1

        self.pre_vec[0:2] = self.normalise(self.pre_vec[0:2])

    def apply_forces(self, population, predator):
        # force calculation function decides new bearing

        self.boid_force_vec_calc(population)
        self.pred_force_vec_calc(predator)
        NOISE = self.noise()

        if self.sep_vec[2] != 0:  # seperation given priority
            # not given a noise term since its such close range
            TOTAL_FORCE = self.sep_vec[0:2]

        else:
            TOTAL_FORCE = [self.ali_vec[0] * self.alignment_weight + self.coh_vec[0] * self.cohesion_weight + self.pre_vec[0] * self.pred_weight + NOISE_WEIGHT * NOISE[0],
                           self.ali_vec[1] * self.alignment_weight + self.coh_vec[1] * self.cohesion_weight + self.pre_vec[1] * self.pred_weight + NOISE_WEIGHT * NOISE[1]]

        if TOTAL_FORCE[0]**2 + TOTAL_FORCE[1]**2 == 0:
            desired_heading = self.angle

        else:
            desired_heading = atan2(TOTAL_FORCE[1],
                                    TOTAL_FORCE[0])

        # applies max turning angle constraint
        angle_diff = desired_heading - self.angle
        delta_angle = self.angle_smoother(angle_diff)
        self.angle += delta_angle
        self.angle = self.pi_range(self.angle)

        self.move()

    def draw(self, screen, predator):

        colour = (255, 255, 255)

        if PRED_FOV:
            if self.seen_by_pred(predator):

                arrow_length = BOID_RADIUS
                # Calculate the points of the arrow shape

                self.tri_points[0] = (self.x + arrow_length * math.cos(self.angle),
                                      self.y + arrow_length * math.sin(self.angle))
                self.tri_points[1] = (self.x + (arrow_length/3) * math.cos(self.angle - math.pi/2),
                                      self.y + (arrow_length/3) * math.sin(self.angle - math.pi/2))

                self.tri_points[2] = (self.x + (arrow_length/3) * math.cos(self.angle + math.pi/2),
                                      self.y + (arrow_length/3) * math.sin(self.angle + math.pi/2))

                # ]
                pygame.draw.polygon(screen, colour, self.tri_points)

                # Draw the arrow

        else:
            arrow_length = BOID_RADIUS
            # Calculate the points of the arrow shape

            self.tri_points[0] = (self.x + arrow_length * math.cos(self.angle),
                                  self.y + arrow_length * math.sin(self.angle))
            self.tri_points[1] = (self.x + (arrow_length/3) * math.cos(self.angle - math.pi/2),
                                  self.y + (arrow_length/3) * math.sin(self.angle - math.pi/2))

            self.tri_points[2] = (self.x + (arrow_length/3) * math.cos(self.angle + math.pi/2),
                                  self.y + (arrow_length/3) * math.sin(self.angle + math.pi/2))
            pygame.draw.polygon(screen, colour, self.tri_points)

    def in_pred_range(self):
        # counts time in predator range for predator attack success and total for evolution

        if self.pred_distance <= PRED_RANGE and self.alive:  # self.alive stops ticking up after death
            self.predator_timer += 1
            self.total_pred_time += 1
        else:
            self.predator_timer = 0

    def pred_attacks(self, t, predator):

        if self.pred_distance < BOID_RADIUS+PRED_RADIUS and predator.eating_bool == False:
            self.near_count = self.sep_vec[2] + \
                self.ali_vec[2] + self.coh_vec[2]
            pred_success_threshold = predator.success_calc(self.near_count)
            if random.random() < pred_success_threshold:
                self.speed = 0
                self.alive = False
                self.time_of_death = t
                predator.eating_init()
            else:
                predator.eating_init()

    def fitness_calc(self, kill_counter):
        # fitness function - note, maybe take into account domains of danger for the predator
        # indirectly measures some sociability as well since a boid can react to its peers and avoid the predator without seeing it
        # tried linear and quadratic, mention this in the report

        # adding in extra time for eating
        total_time = TIMESTEPS + PRED_PAUSE_TIME * kill_counter

        if self.alive:
            # function from 0-1 based on how much the it was in the predators range. if it
            # total time near predator as a percentage of total run time
            p = self.total_pred_time / total_time

            self.fitness = -0.9*p + 1

        else:

            self.fitness = 0

    def mating(self, mate):
        # mating with one point crossover, returns new evolution vars for offspring

        crossover_point = random.randint(0, len(self.evolution_vars))
        offspring_vars = self.evolution_vars[:crossover_point] + \
            mate.evolution_vars[crossover_point:]

        return offspring_vars

    def mutate(self):
        r = random.random()
        mid_point = 1/3
        if r < mid_point:
            self.a = self.generate_a()
            self.b = 1/self.a
        elif r < mid_point*2:
            self.pred_weight = self.generate_pred_weight()
            self.social_weights = (1-self.pred_weight)/2
        else:
            self.evolution_vars[2] = round(random.random(), 1)

    def rewrite_evolution_vars(self):
        self.evolution_vars = [
            self.a, self.pred_weight, self.evolution_vars[2]]
        self.area_factor = self.blind_area_addition_factor_calc()
        self.cohesion_weight = (
            1 - self.evolution_vars[2]) * self.social_weights
        self.separation_area = SEPARATION_AREA
        self.alignment_area = self.area_factor * ALIGNMENT_AREA
        self.cohesion_area = self.area_factor * COHESION_AREA
        self.predator_flee_area = self.area_factor * PREDATOR_FLEE_AREA


class Flock:
    def __init__(self, pop_size, W, H, num_gens, respawn=False):
        self.num_gens = num_gens
        self.pop_size = pop_size
        self.width = W
        self.height = H
        self.generation = 0
        self.kill_count = 0
        self.respawn_bool = respawn
        self.mutation_probability = 0.2
        self.stat_index = 0

        self.populate()
        self.original_pop = self.population.copy()
        self.stats = pd.DataFrame(index=range(num_gens), columns=[
                                  'Kills', 'Mean Fitness'])
        self.pop_attr = pd.DataFrame(index=range(num_gens), columns=[
                                     f'Boid_{n}_{axis}' for n in range(self.pop_size) for axis in ['a', 'pw', 'sw']])
        self.spacial_statistics = pd.DataFrame(index=range(int(num_gens*(TIMESTEPS+1+(PRED_PAUSE_TIME*self.pop_size)))), columns=[
                                               'Gen', 'Timestep', 'Mouse_x', 'Mouse_y', 'Pred_x', 'Pred_y', 'Pred_angle']+[f'Boid_{n}_{axis}' for n in range(self.pop_size) for axis in ['x', 'y', 'angle']])

    def add_spacial_statistics(self, predator, timestep):
        predator_pos = [int(predator.x), int(predator.y),
                        round(predator.angle, 4)]
        mouse_x, mouse_y = pygame.mouse.get_pos()

        boid_pop_pos = []
        for boid in self.original_pop:
            if boid.alive:
                for i in [int(boid.x), int(boid.y), round(boid.angle, 4)]:
                    boid_pop_pos.append(i)
            else:
                for i in range(3):
                    boid_pop_pos.append(57005)  # nan if dead

        row = [self.generation] + [timestep] + \
            [mouse_x, mouse_y] + predator_pos+boid_pop_pos
        self.spacial_statistics.iloc[self.stat_index] = row
        self.stat_index += 1

    def populate(self):
        # Evolution_Vars = [a, pred_weight, alignment_weight]
        self.population = []
        first_third = math.floor(self.pop_size/3)
        final_third = self.pop_size - 2 * first_third

        for i in range(self.pop_size):
            # [a,pw,sw]
            evolution_vars = [random.choice(potential_a_vals), round(
                random.random(), 1), round(random.random(), 1)]
            self.population.append(
                Boid(evolution_vars, self.width, self.height))

    def update_flock(self, screen, PREDATOR, t):

        self.add_spacial_statistics(PREDATOR, t)

        for BOID in self.population:
            if BOID.alive == False:
                self.kill_count += 1
                self.population.remove(BOID)
            BOID.apply_forces(self.population, PREDATOR)
            if PRED_DRAW:
                BOID.draw(screen, PREDATOR)
            BOID.in_pred_range()
            BOID.pred_attacks(t, PREDATOR)

    def add_pop_attr(self):
        stats_row = []
        for i in self.original_pop:
            stats_row += [a_list_decoder(i.evolution_vars[0]),
                          i.evolution_vars[1], i.evolution_vars[2]]
        self.pop_attr.iloc[self.generation] = stats_row

    def flock_fitness(self):
        # calculates the fitness for all individuals after a run
        # adds the fitness and the evolutionary vars to the stats list

        self.ind_fitness_list = []
        a_stats_list = []
        pred_weight_stats_list = []
        social_weight_stats_list = []

        for ind in self.original_pop:
            ind.fitness_calc(self.kill_count)
            # list of individual fitnesses
            self.ind_fitness_list.append(ind.fitness)
            a_stats_list.append(a_list_decoder(ind.evolution_vars[0]))
            pred_weight_stats_list.append(ind.evolution_vars[1])
            social_weight_stats_list.append(round(ind.evolution_vars[2], 1))

        mean_fitness = np.mean(self.ind_fitness_list)

        stats_row = [self.kill_count, mean_fitness]
        self.stats.iloc[self.generation] = stats_row
        self.total_fitness = np.sum(self.ind_fitness_list)
        self.add_pop_attr()

    def mate_once(self, parents):
        # parents is a tuple of size 2, generated using pick parents
        # returns offspring vars
        child = parents[0].mating(parents[1])
        return child

    def roulette(self, k):
        # selects individuals with replacement from the population proportional to fitness
        selected = []

        for i in range(k):
            u = random.random() * self.total_fitness
            sum = 0
            for ind in self.original_pop:
                sum += ind.fitness
                if sum > u:
                    selected.append(ind)
                    break

        return (selected)

    def generate_children(self, num_children):
        children = []

        for i in range(num_children):
            parents = self.roulette(2)
            children.append(self.mate_once(parents))

        return children

    def tournament(self, tourn_size, k):
        # tourn_size = tournamnet size
        # k = number of selections
        # returns k 'winners'

        chosen = []
        for i in range(k):
            aspirants = tools.selRandom(self.original_pop, tourn_size)
            chosen.append(
                max(aspirants, key=attrgetter('fitness')).evolution_vars)
        return chosen

    def mutate_population(self, external_mutation_prob):
        for i in self.population:
            if random.random() < external_mutation_prob:
                i.mutate()
                i.rewrite_evolution_vars()

    def mutation_prob_calc(self):
        if self.generation < self.num_gens/4:
            return 0.2
        elif self.generation < 2*self.num_gens/2:
            return 0.1
        else:
            return 0.05

    def next_generation(self):
        # creates new population using mating and tournament selection

        self.generation += 1  # generation counter

        num_children = math.floor(self.pop_size/2)
        num_survivors = self.pop_size - num_children

        children_vars = self.generate_children(num_children)  # vars
        parents_vars = self.tournament(
            2, num_survivors)  # using tournament size 2
        new_pop_vars = children_vars + parents_vars
        new_pop = [Boid(vars, self.width, self.height)
                   for vars in new_pop_vars]  # initialise new pop

        self.population = new_pop
        self.mutate_population(self.mutation_prob_calc())
        self.original_pop = self.population.copy()
        self.kill_count = 0

    def save_stats(self, filepath1='stats.csv', filepath2='spacial_stats.csv', filepath3='attributes.csv'):

        self.stats.to_csv(filepath1, mode='w')

        self.spacial_statistics = self.spacial_statistics.dropna(how='all')
        self.spacial_statistics[self.spacial_statistics == 57005] = np.NaN
        self.spacial_statistics.to_csv(filepath2)

        self.pop_attr.to_csv(filepath3)

    def get_pop_attr_row(self, gen_index):
        final_attrs = self.pop_attr.iloc[gen_index]
        a_list = final_attrs[::3]
        pw_list = final_attrs[1::3]
        sw_list = final_attrs[2::3]
        self.a_list = np.array(a_list)
        self.pw_list = np.array(pw_list)
        self.sw_list = np.array(sw_list)
        self.attr_row = self.pop_attr.iloc[gen_index]

    def plot_pw_sw(self, gen_index, three_dimension=False):
        # just does a scatter of pw, sw for now

        self.get_pop_attr_row(gen_index)
        if three_dimension:
            ###### 3D######
            # Create figure and 3D axes
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')

            # Scatter plot
            ax.scatter(self.a_list, self.pw_list, self.sw_list, alpha=0.1)
            ax.scatter(np.mean(self.a_list), np.mean(self.pw_list),
                       np.mean(self.sw_list), color='red')

            # Set labels and title
            ax.set_xlabel('A')
            ax.set_ylabel('PW')
            ax.set_zlabel('SW')
            ax.set_title('3D Scatter Plot')
            ax.set_xlim(0, 1.5)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)

            plt.show()
        else:

            #### 2D####
            plt.scatter(self.pw_list, self.sw_list, alpha=0.1)
            plt.scatter(np.mean(self.pw_list), np.mean(
                self.sw_list), color='red')
            plt.title('pw/sw')
            plt.xlabel('pw')
            plt.ylabel('sw')
            plt.xlim(-0.1, 1.1)
            plt.ylim(-0.1, 1.1)
            plt.draw()

    def calc_MS_cluster(self, gen_index):
        # needs to be expanded to be more variables, currently just pw,sw
        self.get_pop_attr_row(gen_index)
        # self.data_2d =  data_2d = np.column_stack((self.pw_list, self.sw_list)) #weird variable name
        self.MSdata = np.column_stack(
            (self.a_list, self.pw_list, self.sw_list))
        np.savetxt('array_data.csv', self.MSdata, delimiter=',')

        meanshift = MeanShift(bandwidth=np.sqrt(3*0.1**2))

        # Fit the model
        meanshift.fit(self.MSdata)

        # Retrieve cluster centers and labels
        self.cluster_centers = meanshift.cluster_centers_
        self.cluster_labels = meanshift.labels_

    def plot_cluster_pw_sw(self, gen_index, three_dimensions=False):
        # needs to be expanded to be more variable

        self.calc_MS_cluster(gen_index)
        # Number of clusters
        n_clusters = len(self.cluster_centers)

        if three_dimensions:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.MSdata[:, 0], self.MSdata[:, 1],
                       self.MSdata[:, 2], c=self.cluster_labels, cmap='viridis')
            ax.scatter(self.cluster_centers[:, 0], self.cluster_centers[:, 1],
                       self.cluster_centers[:, 2], marker='x', color='red', label='Cluster Centers')
            ax.set_xlabel('A')
            ax.set_ylabel('PW')
            ax.set_zlabel('SW')
            ax.set_title('Mean Shift Clustering of 3D Data')
            ax.set_xlim(0, 1.5)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)

        else:

            # Plotting the clusters - indexing is [a,pw,sw]
            scatter = plt.scatter(
                self.MSdata[:, 1], self.MSdata[:, 2], c=self.cluster_labels, cmap='viridis')
            plt.scatter(self.cluster_centers[:, 1], self.cluster_centers[:,
                        2], marker='x', color='red', label='Cluster Centers')
            plt.xlabel('Predator Weight')
            plt.ylabel('Social Weight')
            plt.title('Mean Shift Clustering of 2D Data')
            plt.xlim(-0.1, 1.1)
            plt.ylim(-0.1, 1.1)

            # Adding legend
            handles, _ = scatter.legend_elements()
            plt.legend(handles, [f'Cluster {i}' for i in range(
                n_clusters)] + ['Cluster Centers'], loc='upper right')
            plt.draw()

    def calc_cluster_stats(self, gen_index):
        # again just does pw/sw but this is the easiest one to adjust, just fiddle with cluster centre values
        self.calc_MS_cluster(gen_index)

        self.cluster_stats = pd.DataFrame(index=range(
            max(self.cluster_labels)+1), columns=['Size', 'Centroid', 'SD'])
        for i in range(max(self.cluster_labels)+1):

            # assigns empty row
            row = [np.NaN, np.NAN, np.NAN]
            # mask to just find members of cluster
            lab_mask1 = self.cluster_labels == i
            # trebled so mask is expanded from 50 3d lists to 150 points
            lab_mask2 = [element for element in lab_mask1 for _ in range(3)]

            masked_data = self.pop_attr.iloc[gen_index][lab_mask2]
            row[0] = len(self.cluster_labels[lab_mask1])
            row[1] = self.cluster_centers[i]

            # iterates through cluster population
            for j in range(row[0]):
                masked_data[j] = np.array(masked_data[j]) - row[1]

            # now find standard deviation from cluster centre  sd = sqrt(sum( (each element- mean)**2 )/N)
            # N = row[0]
            sd_list = [np.NaN, np.NAN, np.NAN]

            # cluster center[0] is dimension of cluster
            for j in range(len(row[1])):
                count = 0
                for k in range(row[0]):
                    count += (masked_data[k][j])**2
                sd_list[j] = np.sqrt(count/50)
            row[2] = sd_list
            self.cluster_stats.iloc[i] = row


def show_message(screen, font, message, location):
    message_render = font.render(message, True, (0, 0, 0))
    message_rect = message_render.get_rect(center=location)
    screen.blit(message_render, message_rect)


def pause_loop(screen, font):
    pause_message1 = 'Press SPACE to continue'
    pause_message2 = 'Press ESC to quit'
    show_message(screen, font, pause_message1, (WIDTH//2, HEIGHT//2 + 50))
    show_message(screen, font, pause_message2, (WIDTH//2, HEIGHT//2+-50))


def esc_check(screen, font):
    esc_check_message1 = 'Are you sure you want to quit?'
    esc_check_message2 = 'SPACE to return to the game'
    esc_check_message3 = 'ESC to quit'
    show_message(screen, font, esc_check_message1, (WIDTH//2, HEIGHT//2-50))
    show_message(screen, font, esc_check_message2, (WIDTH//2, HEIGHT//2))
    show_message(screen, font, esc_check_message3, (WIDTH//2, HEIGHT//2+50))


def intro_scr1(screen, font):
    m1 = 'Hello, welcome to my study.'
    m2 = 'The following game will use your inputs as a predator to evolve a prey population'
    m3 = 'to explore the effects of different predator styles on the behaviour of herding prey.'
    m4 = 'This research is for a MEng Technical Project. Entry in this study is optional and participants'
    m5 = 'can opt out at any time during the game (but not after, '
    m6 = 'as I will not be able to tell the data is from you).'
    m20 = 'Click anywhere to continue'

    show_message(screen, font, m1, (WIDTH//2, HEIGHT*0.2))
    show_message(screen, font, m2, (WIDTH//2, HEIGHT*0.4))
    show_message(screen, font, m3, (WIDTH//2, HEIGHT*0.43))
    show_message(screen, font, m4, (WIDTH//2, HEIGHT*0.46))
    show_message(screen, font, m5, (WIDTH//2, HEIGHT*0.49))
    show_message(screen, font, m6, (WIDTH//2, HEIGHT*0.52))
    show_message(screen, font, m20, (WIDTH//2, HEIGHT*0.75))


def intro_scr2(screen, font):
    m1 = 'In this game, you will control a predator marked by a red arrow which will move towards your mouse.'
    m2 = 'Your aim is to catch as many of the prey as you can in the time limit. Each round will last around'
    m3 = '25 seconds and you will complete 40 rounds to fully evolve the prey. You will do this twice,'
    m4 = ' once where all predator attacks are successful, and another where the predator will be less'
    m5 = 'successful against prey in groups than individuals. The order of the two modes will be randomised.'
    m6 = 'The total duration will be roughly 30 minutes.  '
    m20 = 'Click anywhere to continue'

    show_message(screen, font, m1, (WIDTH//2, HEIGHT*0.37))
    show_message(screen, font, m2, (WIDTH//2, HEIGHT*0.4))
    show_message(screen, font, m3, (WIDTH//2, HEIGHT*0.43))
    show_message(screen, font, m4, (WIDTH//2, HEIGHT*0.46))
    show_message(screen, font, m5, (WIDTH//2, HEIGHT*0.49))
    show_message(screen, font, m6, (WIDTH//2, HEIGHT*0.52))
    show_message(screen, font, m20, (WIDTH//2, HEIGHT*0.75))


def consent_check(screen, font):
    m1 = 'This study will collect your age, gender and the positional data of your predator'
    m2 = 'and mouse in the game. No further data will be recorded and I will delete your'
    m3 = 'email if you send data via email as soon as I have downloaded the attached data.'
    m4 = 'Because all data is anonymous, I cannot delete your data once you have taken part.'
    m5 = 'In case you have questions of concerns about this study, please contact '
    m6 = 'myself (hb20788@bristol.ac.uk),'
    m7 = 'my project supervisor (nikolai.bode@bristol.ac.uk),'
    m8 = 'or research governance (research-governance@bristol.ac.uk).'
    m9 = 'By clicking this box, I consent to the above data being collected:'

    show_message(screen, font, m1, (WIDTH//2, HEIGHT*0.37))
    show_message(screen, font, m2, (WIDTH//2, HEIGHT*0.4))
    show_message(screen, font, m3, (WIDTH//2, HEIGHT*0.43))
    show_message(screen, font, m4, (WIDTH//2, HEIGHT*0.46))
    show_message(screen, font, m5, (WIDTH//2, HEIGHT*0.49))
    show_message(screen, font, m6, (WIDTH//2, HEIGHT*0.52))
    show_message(screen, font, m7, (WIDTH//2, HEIGHT*0.55))
    show_message(screen, font, m8, (WIDTH//2, HEIGHT*0.58))
    show_message(screen, font, m9, (WIDTH//2-50, HEIGHT*0.65))

    pygame.draw.rect(screen, (255, 0, 0), (WIDTH / 2 +
                     225, HEIGHT * 0.65 - 12.5, 25, 25), 5)


def instructions(screen, font):
    m1 = 'You will direct the predator with your mouse and you must aim to catch as many'
    m2 = 'of the prey as you can in each round. The predator will be stationary for the ' 
    m3 = 'start of each round to allow the prey to form groups. The predator will also be'
    m4 = 'stationary when for a short period of time after attempting an attack. Finally, '
    m5 = 'the boundaries for the prey can be passed through resulting in them appearing on '
    m6 = 'the other side of the screen. They are solid for the predator.' 
    m7 = 'Consider how the different game modes could affect your hunting strategies.'
    m8 = 'Press the spacebar at any time to pause the game or exit.'
    m20 = 'Click anywhere to continue'

    show_message(screen, font, m1, (WIDTH//2, HEIGHT*0.37))
    show_message(screen, font, m2, (WIDTH//2, HEIGHT*0.40))
    show_message(screen, font, m3, (WIDTH//2, HEIGHT*0.43))
    show_message(screen, font, m4, (WIDTH//2, HEIGHT*0.46))
    show_message(screen, font, m5, (WIDTH//2, HEIGHT*0.49))
    show_message(screen, font, m6, (WIDTH//2, HEIGHT*0.52))
    show_message(screen, font, m7, (WIDTH//2, HEIGHT*0.55))
    show_message(screen, font, m8, (WIDTH//2, HEIGHT*0.61))
    show_message(screen, font, m20, (WIDTH//2, HEIGHT*0.75))


def init_normal(screen, font):
    m1 = 'In this round, the predator will always be successful'
    m20 = 'Click anywhere to start'
    show_message(screen, font, m1, (WIDTH//2, HEIGHT*0.37))
    show_message(screen, font, m20, (WIDTH//2, HEIGHT*0.75))


def init_success(screen, font):
    m1 = 'In this round, the predators success rate will fall proportional to the'
    m2 = 'number of neighbours the prey has.'
    m20 = 'Click anywhere to start'
    show_message(screen, font, m1, (WIDTH//2, HEIGHT*0.37))
    show_message(screen, font, m2, (WIDTH//2, HEIGHT*0.40))
    show_message(screen, font, m20, (WIDTH//2, HEIGHT*0.75))


def break_screen(screen, font):
    m1 = "Woohoo! You're halfway there! Here's a chance to take a break :)"
    m20 = 'Click anywhere to continue'

    show_message(screen, font, m1, (WIDTH//2, HEIGHT*0.37))
    show_message(screen, font, m20, (WIDTH//2, HEIGHT*0.75))


def finished_screen(screen, font):
    m1 = 'Thanks for participating. Please email me your output file:'
    m2 = 'hb20788@bristol.ac.uk'
    m3 = 'Finish the game by pressing escape'

    show_message(screen, font, m1, (WIDTH//2, HEIGHT*0.37))
    show_message(screen, font, m2, (WIDTH//2, HEIGHT*0.4))
    show_message(screen, font, m3, (WIDTH//2, HEIGHT*0.43))

### FSA STATES ###


STATE_INIT = 0
STATE_SCR2 = 1
STATE_CONSENT = 2
STATE_INSTRUCTIONS = 3

STATE_INIT_NORMAL_PREDATOR = 4
STATE_NORMAL_PREDATOR = 5
STATE_NORMAL_PREDATOR_IT_TICK = 6

STATE_INIT_SUCCESS_PREDATOR = 7
STATE_SUCCESS_PREDATOR = 8
STATE_SUCCESS_PREDATOR_IT_TICK = 9

STATE_BREAK = 20
STATE_FINISHED = 21

STATE_PAUSE = 50
STATE_ESC_CHECK = 51
STATE_QUIT = 52
STATE_NO_STATE = 53


def save_previous_state(previous_state, current_state):
    if current_state == STATE_PAUSE:
        return previous_state
    elif current_state == STATE_ESC_CHECK:
        return previous_state
    else:
        return current_state


def state_update(previous_state, state, run_type_memory, halfway_bool, folder_path):

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            state = STATE_QUIT

        elif event.type == pygame.KEYDOWN:

            if event.key == pygame.K_ESCAPE:

                if state == STATE_ESC_CHECK:
                    state = STATE_QUIT

                elif state == STATE_FINISHED:
                    state = STATE_QUIT
                else:
                    previous_state = save_previous_state(previous_state, state)
                    state = STATE_ESC_CHECK

            elif event.key == pygame.K_SPACE:

                if state == STATE_PAUSE:
                    state = previous_state

                else:
                    previous_state = save_previous_state(previous_state, state)
                    state = STATE_PAUSE

            else:
                pass

        elif state == STATE_NORMAL_PREDATOR or state == STATE_NORMAL_PREDATOR_IT_TICK or state == STATE_SUCCESS_PREDATOR or state == STATE_SUCCESS_PREDATOR_IT_TICK:
            pass

        elif event.type == pygame.MOUSEBUTTONDOWN:
            # If mouse button is clicked, move to second screen
            if state == STATE_INIT:
                previous_state = save_previous_state(previous_state, state)
                state = STATE_SCR2

            elif state == STATE_SCR2:
                previous_state = save_previous_state(previous_state, state)
                state = STATE_CONSENT

            elif state == STATE_CONSENT:
                x, y = pygame.mouse.get_pos()
                if (WIDTH / 2 + 225 < x < WIDTH / 2 + 225 + 25) and (HEIGHT*0.65 - 12.5 < y < HEIGHT*0.65 + 12.5):
                    previous_state = save_previous_state(previous_state, state)
                    state = STATE_INSTRUCTIONS

            elif state == STATE_INSTRUCTIONS:
                previous_state = save_previous_state(previous_state, state)
                state = random.choice(
                    [STATE_INIT_NORMAL_PREDATOR, STATE_INIT_SUCCESS_PREDATOR])
                run_type_memory = state

            elif state == STATE_INIT_SUCCESS_PREDATOR:
                previous_state = save_previous_state(previous_state, state)
                state = STATE_SUCCESS_PREDATOR

            elif state == STATE_INIT_NORMAL_PREDATOR:
                previous_state = save_previous_state(previous_state, state)
                state = STATE_NORMAL_PREDATOR

            elif state == STATE_BREAK:
                if halfway_bool == True:
                    state = STATE_FINISHED
                else:
                    previous_state = save_previous_state(previous_state, state)
                    halfway_bool = True

                    if run_type_memory == STATE_INIT_SUCCESS_PREDATOR:
                        order = 'success then normal'
                        state = STATE_INIT_NORMAL_PREDATOR

                    elif run_type_memory == STATE_INIT_NORMAL_PREDATOR:
                        order = 'normal then success'
                        state = STATE_INIT_SUCCESS_PREDATOR
                    with open(folder_path + 'run_order.txt', "w") as file:
                        file.write(order)

            elif state == STATE_FINISHED:
                pass
            elif state == STATE_PAUSE:
                pass

            else:
                pass

        elif state == STATE_BREAK:
            if halfway_bool == True:
                state = STATE_FINISHED

    return state, previous_state, run_type_memory, halfway_bool


def player_run(NUM_GENS):
    """"
    put save functions outside of the pygame loops?
    Intro screen
    Consent form with tick
    Descibe experiment

    Run a game
    save stats

    Break/ change over

    Run other game
    save stats

    Finish


    """

    state = STATE_INIT
    previous_state = STATE_INIT
    folder_path = write_new_folder('OUTPUT')
    pygame.init()

    # Calculate scaling factor
    screen = pygame.display.set_mode((int(WIDTH), int(HEIGHT)))

    # screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(
        "Using a virtual experiment to model the evolution of sociability and vision in prey")
    font_48 = pygame.font.Font(None, 48)
    font_24 = pygame.font.Font(None, 24)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    gen_num = 0

    running = True
    order_AB = True
    run_type_memory = -1
    halfway_bool = False

    while running:
        # Clear the screen
        screen.fill((255, 255, 255))

        state, previous_state, run_type_memory, halfway_bool = state_update(
            previous_state, state, run_type_memory, halfway_bool, folder_path)

        if state == STATE_INIT:
            intro_scr1(screen, font_24)
        elif state == STATE_SCR2:
            intro_scr2(screen, font_24)

        elif state == STATE_CONSENT:
            consent_check(screen, font_24)

        elif state == STATE_INSTRUCTIONS:
            instructions(screen, font_24)

        elif state == STATE_INIT_NORMAL_PREDATOR:
            init_normal(screen, font_24)
            FLOCK = Flock(N_BOIDS, WIDTH, HEIGHT, NUM_GENS)
            gen_num = 0

        elif state == STATE_NORMAL_PREDATOR:
            if gen_num < NUM_GENS:

                gen_num += 1
                PREDATOR = Predator(WIDTH/2, HEIGHT/2, WIDTH,
                                    HEIGHT, success_bool=False)

                clock = pygame.time.Clock()
                itteration = 0
                state = STATE_NORMAL_PREDATOR_IT_TICK
            else:
                FLOCK.save_stats((folder_path + 'normal_kills.csv'), (folder_path +
                                 'normal_spacial.csv'), (folder_path + 'normal_attributes.csv'))
                previous_state = save_previous_state(previous_state, state)
                state = STATE_BREAK

        elif state == STATE_NORMAL_PREDATOR_IT_TICK:
            if itteration < TIMESTEPS + PRED_PAUSE_TIME*FLOCK.kill_count:

                itteration += 1

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_ESCAPE]:
                        state = STATE_ESC_CHECK
                    elif keys[pygame.K_SPACE]:
                        state = STATE_PAUSE
                screen.fill((0, 0, 0))
                clock.tick(FPS)

                PREDATOR.update(screen, FLOCK.population)
                FLOCK.update_flock(screen, PREDATOR, itteration)

                font = pygame.font.Font(None, 24)
                text = font.render(
                    f"TIME: {round(itteration/FPS,1)}, GENERATION: {FLOCK.generation+1}/{NUM_GENS}, KILLS: {FLOCK.kill_count}/{N_BOIDS}, SUCCESS: CONSTANT", True, (255, 255, 255))
                screen.blit(text, (20, 20))

            else:
                FLOCK.flock_fitness()
                if gen_num != NUM_GENS:
                    FLOCK.next_generation()

                previous_state = save_previous_state(previous_state, state)
                state = STATE_NORMAL_PREDATOR

        elif state == STATE_INIT_SUCCESS_PREDATOR:
            init_success(screen, font_24)
            FLOCK = Flock(N_BOIDS, WIDTH, HEIGHT, NUM_GENS)
            gen_num = 0

        elif state == STATE_SUCCESS_PREDATOR:
            if gen_num < NUM_GENS:
                gen_num += 1
                PREDATOR = Predator(WIDTH/2, HEIGHT/2, WIDTH,
                                    HEIGHT, success_bool=True)

                clock = pygame.time.Clock()
                itteration = 0
                state = STATE_SUCCESS_PREDATOR_IT_TICK
            else:
                FLOCK.save_stats((folder_path + 'success_kills.csv'), (folder_path +
                                 'success_spacial.csv'), (folder_path + 'success_attributes.csv'))
                previous_state = save_previous_state(previous_state, state)
                state = STATE_BREAK

        elif state == STATE_SUCCESS_PREDATOR_IT_TICK:
            if itteration < TIMESTEPS + PRED_PAUSE_TIME*FLOCK.kill_count:
                itteration += 1

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_ESCAPE]:
                        state = STATE_ESC_CHECK
                    elif keys[pygame.K_SPACE]:
                        state = STATE_PAUSE
                screen.fill((0, 0, 0))
                clock.tick(FPS)

                PREDATOR.update(screen, FLOCK.population)
                FLOCK.update_flock(screen, PREDATOR, itteration)

                font = pygame.font.Font(None, 24)
                text = font.render(
                    f"TIME: {round(itteration/FPS,1)}, GENERATION: {FLOCK.generation+1}/{NUM_GENS}, KILLS: {FLOCK.kill_count}/{N_BOIDS}, SUCCESS: VARIABLE", True, (255, 255, 255))
                screen.blit(text, (20, 20))

            else:

                FLOCK.flock_fitness()
                if gen_num != NUM_GENS:
                    FLOCK.next_generation()
                previous_state = save_previous_state(previous_state, state)
                state = STATE_SUCCESS_PREDATOR

        elif state == STATE_BREAK:
            break_screen(screen, font_24)
            gen_num = 0

        elif state == STATE_FINISHED:
            finished_screen(screen, font_24)

        elif state == STATE_PAUSE:
            pause_loop(screen, font_48)

        elif state == STATE_ESC_CHECK:
            esc_check(screen, font_48)

        elif state == STATE_QUIT:
            running = False

        elif state == STATE_NO_STATE:
            running = False
            print('Error in state assignment')

        else:
            running = False
            print('Unknown state')

        pygame.display.flip()


player_run(40)
