from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit,
            wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        # TODO: Implement
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.steer_ratio = steer_ratio
        self.max_lat_accel = max_lat_accel
        self.max_steer_angle = max_steer_angle

        min_speed = 0.1 # m/s
        self.yaw_controller = YawController(self.wheel_base, self.steer_ratio, min_speed,
                self.max_lat_accel, self.max_steer_angle)

        kp_t = 0.3
        ki_t = 0.1
        kd_t = 0.
        min_t = 0.
        max_t = 0.2
        self.throttle_controller = PID(kp_t, ki_t, kd_t, min_t, max_t)

        tau_lpf = 0.5
        ts_lpf = 0.02
        self.vel_lpf = LowPassFilter(tau_lpf, ts_lpf)

        self.last_time = rospy.get_time()


    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0., 0., 0.

        current_vel_lpf = self.vel_lpf.filt(current_vel)

        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel_lpf)

        vel_error = linear_vel - current_vel_lpf
        self.last_vel = current_vel_lpf

        current_time = rospy.get_time()
        sample_time = current_time = self.last_time
        self.last_time = current_time
        # print('linear_vel:', linear_vel)
        # print('angular_vel:', angular_vel)
        # print('current_vel_lpf:', current_vel_lpf)
        # print('vel_error:', vel_error)
        # print('sample_time:', sample_time)

        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0

        if linear_vel == 0. and current_vel < 0.1:
            throttle = 0
            decel_target = 1.0  # m/s^2
            brake = self.vehicle_mass * decel_target * self.wheel_radius

        elif throttle < .1 and vel_error < 0:
            throttle = 0
            decel_target = max(vel_error, self.decel_limit)
            brake = abs(decel_target) * self.vehicle_mass * self.wheel_radius

        # print('throttle:', throttle)
        # print('brake:', brake)
        # print('steering:', steering)

        return throttle, brake, steering
