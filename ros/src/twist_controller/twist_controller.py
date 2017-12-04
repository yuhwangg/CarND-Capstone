from yaw_controller import YawController
from pid import PID

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

class Controller(object):
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle, decel_limit, accel_limit,
                    vehicle_mass, wheel_radius, brake_deadband):
        self.wheel_base = wheel_base
        self.steer_ratio = steer_ratio
        self.min_speed = min_speed
        self.max_lat_accel = max_lat_accel
        self.max_steer_angle = max_steer_angle
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.vehicle_mass = vehicle_mass
        self.wheel_radius = wheel_radius
        self.brake_deadband = brake_deadband

        yaw_controller_params = {
            'wheel_base': self.wheel_base,
            'steer_ratio': self.steer_ratio,
            'min_speed': self.min_speed,
            'max_lat_accel': self.max_lat_accel,
            'max_steer_angle': self.max_steer_angle
        }

        # TODO: tune PID params
        steer_controller_params = {
            'kp': 0.2,
            'ki': 0.00001,
            'kd': 0.3,
            'mn': -max_steer_angle,
            'mx': max_steer_angle
        }
        # TODO: tune PID params and enable (see below)
        throttle_controller_params = {
            'kp': 1,
            'ki': 0,
            'kd': 0.3,
            'mn': decel_limit,
            'mx': accel_limit
        }

        self.yaw_controller = YawController(**yaw_controller_params) #TODO: figure out how to use it ^_^
        self.steer_controller = PID(**steer_controller_params)
        self.throttle_controller = PID(**throttle_controller_params)
        pass

    def control(self, target_vel,  curr_vel, target_ang_vel, curr_ang_vel, dbw_enabled, curr_angle, cte, elapsed):
        # TODO: this is a rough and incomplete implementation, but it kind of follow the waypoints, bit wobboly
        steer = 0.
        throttle = 0.
        brake = 0.
        #print("VEL:", curr_vel, target_vel, target_vel-curr_vel)
        if dbw_enabled: # disable pid for manual drive
            steer = -self.steer_controller.step(cte, elapsed)
            throttle = self.throttle_controller.step(target_vel-curr_vel, elapsed)
            brake = (-throttle * self.vehicle_mass * self.wheel_radius) if throttle < 0 else 0
            throttle = throttle if throttle > 0 else 0

        # Return throttle, brake, steer
        return throttle, brake, steer
