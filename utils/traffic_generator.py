# Archivo: utils/traffic_generator.py
import math
import numpy as np

class TrafficGenerator:
    def __init__(self, total_users=4000, min_duration=60, max_duration=900):
        self.total_users = total_users
        self.min_duration = min_duration
        self.max_duration = max_duration
        
        self.running_fn = False
        self.function_tick_start = 0
        self.time_limit = 0
        self.function_number = 0
        
        self.traffic_strategies = {
            0: self._calc_double_wave,
            1: self._calc_lineal,
            2: self._calc_exponential,
            3: self._calc_step,
            4: self._calc_trend,
            5: self._calc_seasonal,
            6: self._calc_sawtooth,
            7: self._calc_spike_recovery
        }
    
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.running_fn = False
        self.function_tick_start = 0

    
    def get_workload(self, current_time):
        relative_time = current_time - self.function_tick_start

        if not self.running_fn or (relative_time >= self.time_limit):
            self.running_fn = True
            self._generate_new_cycle_parameters(current_time)
            relative_time = 0

        strategy_method = self.traffic_strategies.get(self.function_number, self._calc_fallback)
        user_count = strategy_method(relative_time)

        user_count = max(10.0, user_count)
        user_count += np.random.normal(0, user_count * 0.05)

        prob = np.random.randint(1, 101)
        if prob <= 2: 
            user_count = self.load_min
        elif prob >= 99: 
            user_count = self.load_max

        return max(10, min(user_count, self.total_users))

    def _generate_new_cycle_parameters(self, current_time):
        """Genera todos los parámetros aleatorios para el nuevo ciclo de tráfico."""
        self.time_limit = np.random.randint(self.min_duration, self.max_duration)
        self.function_number = np.random.randint(0, 8)
        self.function_tick_start = current_time
        
        self.load_min = self.total_users * np.random.uniform(0.05, 0.25)
        self.load_max = self.total_users * np.random.uniform(0.65, 1.00)
        
        self.shift_peak_one = np.random.uniform(0.1 , 0.4)
        self.shift_peak_two = np.random.uniform(0.6 , 0.9)
        self.width_peak_one = np.random.uniform(0.05, 0.2)
        self.width_peak_two = np.random.uniform(0.05, 0.2)
        
        self.agresiveness_coeficient = np.random.uniform(0.7, 1.5) 
        self.scaling_rate = np.random.randint(1, 10)
        self.step_count = np.random.randint(3, 20)
        
        self.trend_direction = np.random.choice([1, -1])
        self.trend_rate = np.random.uniform(1.0, 5.0)
        
        self.period = np.random.randint(30, 100)
        self.amplitude = np.random.uniform(0.3, 0.8)
        
        self.sawtooth_period = np.random.randint(20, 50)
        self.rise_ratio = np.random.uniform(0.15, 0.35)
        
        self.spike_start = np.random.uniform(0.2, 0.5)
        self.recovery_slope = np.random.uniform(0.05, 0.15)

    def _calc_double_wave(self, relative_time):
        wave_one = (self.load_max - self.load_min) * math.exp(
            -(((relative_time - self.time_limit * self.shift_peak_one) /
               (self.time_limit * self.width_peak_one)) ** 2))
        wave_two = (self.load_max - self.load_min) * math.exp(
            -(((relative_time - self.time_limit * self.shift_peak_two) /
               (self.time_limit * self.width_peak_two)) ** 2))
        return self.load_min + wave_one + wave_two

    def _calc_lineal(self, relative_time):
        mid_time = self.time_limit / 2
        slope = (self.load_max - self.load_min) / max(1, mid_time) * self.agresiveness_coeficient
        if relative_time <= mid_time:
            return self.load_min + slope * relative_time
        else:
            peak = self.load_min + slope * mid_time
            return peak - slope * (relative_time - mid_time)

    def _calc_exponential(self, relative_time):
        peak_time = self.time_limit * 0.7
        if relative_time <= peak_time:
            k = math.log(self.load_max / max(1, self.load_min)) / max(1, peak_time) * self.agresiveness_coeficient
            try:
                return self.load_min * math.exp(k * relative_time)
            except OverflowError:
                return self.load_max
        else:
            drop_rate = (self.load_max - self.load_min) / max(1, self.time_limit - peak_time) * self.agresiveness_coeficient
            return self.load_max - drop_rate * (relative_time - peak_time)

    def _calc_step(self, relative_time):
        mid_time = self.time_limit / 2
        step_duration = max(1, self.time_limit // self.step_count)
        users_per_step = (self.load_max - self.load_min) / max(1, (self.step_count / 2))
        
        if relative_time <= mid_time:
            return self.load_min + users_per_step * (relative_time // step_duration)
        else:
            peak = self.load_min + users_per_step * (mid_time // step_duration)
            return peak - users_per_step * ((relative_time - mid_time) // step_duration)

    def _calc_trend(self, relative_time):
        load_range = self.load_max - self.load_min
        rate = (load_range / max(1, self.time_limit)) * self.trend_rate
        result = self.load_min + (self.trend_direction * rate * relative_time)
        return max(self.load_min, min(self.load_max, result))

    def _calc_seasonal(self, relative_time):
        midpoint = (self.load_max + self.load_min) / 2
        half_range = (self.load_max - self.load_min) / 2
        return midpoint + half_range * self.amplitude * math.sin(
            (2 * math.pi * relative_time) / self.period)

    def _calc_sawtooth(self, relative_time):
        cycle_pos = (relative_time % self.sawtooth_period) / self.sawtooth_period
        if cycle_pos < self.rise_ratio:
            return self.load_min + (self.load_max - self.load_min) * (cycle_pos / self.rise_ratio)
        else:
            return self.load_max - (self.load_max - self.load_min) * (
                (cycle_pos - self.rise_ratio) / (1.0 - self.rise_ratio))

    def _calc_spike_recovery(self, relative_time):
        spike_trigger = self.time_limit * self.spike_start
        if relative_time < spike_trigger:
            return self.load_min
        elif relative_time < spike_trigger + 5:
            return self.load_max
        else:
            recovery_time = relative_time - spike_trigger - 5
            return self.load_min + (self.load_max - self.load_min) * math.exp(
                -self.recovery_slope * recovery_time)

    def _calc_fallback(self, relative_time):
        """Fallback de seguridad si el function_number no está mapeado."""
        return self.load_min
