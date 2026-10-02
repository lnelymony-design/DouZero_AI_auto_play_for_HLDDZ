import json
import os.path

class Config:
    window_width: int
    window_height: int
    window_class_name: str
    window_title: str
    platform: str
    resize_window: bool
    screenshot_image_logs: bool
    template_match_image_logs: bool
    animation_image_compare_logs: bool
    king_color_compare_image_logs: bool
    screenshot_areas: dict[str, str]
    bid_threshold: float
    redouble_threshold: float
    super_redouble_threshold: float
    mingpai_threshold: float
    inference_sample_count: int
    inference_pass_penalty: float
    inference_friendly_pass_penalty: float
    inference_play_behavior_floor: float
    inference_play_behavior_strength: float
    inference_behavior_temperature: float
    inference_min_effective_sample_ratio: float
    inference_residual_behavior_floor: float
    inference_residual_behavior_strength: float
    inference_residual_behavior_temperature: float
    inference_async_enabled: bool
    inference_max_job_age_seconds: float
    inference_shutdown_timeout_seconds: float
    risk_adjustment_enabled: bool
    risk_adjustment_weight: float
    risk_adjustment_min_ess_ratio: float
    risk_adjustment_min_risk_gain: float
    risk_adjustment_max_model_gap_fraction: float
    risk_adjustment_max_model_rank: int
    risk_adjustment_response_samples: int
    rollout_enabled: bool
    rollout_shadow_mode: bool
    rollout_max_worlds: int
    rollout_min_worlds: int
    rollout_max_steps: int
    rollout_time_budget_seconds: float
    rollout_min_ess_ratio: float
    rollout_min_value_gain: float
    rollout_device: str
    rollout_cpu_threads: int
    rollout_max_job_age_seconds: float
    rollout_shutdown_timeout_seconds: float
    rollout_result_max_screen_age_seconds: float
    
    def __init__(self, **kwargs) -> None:
        self.window_width = kwargs.get('window_width', 1600)
        self.window_height = kwargs.get('window_height', 900)
        self.window_class_name = kwargs.get('window_class_name', '')
        self.window_title = kwargs.get('window_title', '')
        self.platform = kwargs.get('platform', 'qq_game_hall')
        self.resize_window = kwargs.get('resize_window', True)
        self.screenshot_image_logs = kwargs.get('screenshot_image_logs', False)
        self.template_match_image_logs = kwargs.get('template_match_image_logs', False)
        self.animation_image_compare_logs = kwargs.get('animation_image_compare_logs', False)
        self.king_color_compare_image_logs = kwargs.get('king_color_compare_image_logs', False)
        self.screenshot_areas = kwargs.get('screenshot_areas', {})
        self.bid_threshold = kwargs.get('bid_threshold', 0.4)
        self.redouble_threshold = kwargs.get('redouble_threshold', 0.7)
        self.super_redouble_threshold = kwargs.get('super_redouble_threshold', 0.8)
        self.mingpai_threshold = kwargs.get('mingpai_threshold', 1.2)
        self.inference_sample_count = kwargs.get('inference_sample_count', 1600)
        self.inference_pass_penalty = kwargs.get('inference_pass_penalty', 0.62)
        self.inference_friendly_pass_penalty = kwargs.get('inference_friendly_pass_penalty', 0.86)
        self.inference_play_behavior_floor = kwargs.get('inference_play_behavior_floor', 0.18)
        self.inference_play_behavior_strength = kwargs.get('inference_play_behavior_strength', 1.0)
        self.inference_behavior_temperature = kwargs.get('inference_behavior_temperature', 0.75)
        self.inference_min_effective_sample_ratio = kwargs.get('inference_min_effective_sample_ratio', 0.28)
        self.inference_residual_behavior_floor = kwargs.get('inference_residual_behavior_floor', 0.35)
        self.inference_residual_behavior_strength = kwargs.get('inference_residual_behavior_strength', 1.0)
        self.inference_residual_behavior_temperature = kwargs.get('inference_residual_behavior_temperature', 0.55)
        self.inference_async_enabled = kwargs.get('inference_async_enabled', True)
        self.inference_max_job_age_seconds = kwargs.get('inference_max_job_age_seconds', 6.0)
        self.inference_shutdown_timeout_seconds = kwargs.get('inference_shutdown_timeout_seconds', 1.0)
        self.risk_adjustment_enabled = kwargs.get('risk_adjustment_enabled', False)
        self.risk_adjustment_weight = kwargs.get('risk_adjustment_weight', 0.45)
        self.risk_adjustment_min_ess_ratio = kwargs.get('risk_adjustment_min_ess_ratio', 0.45)
        self.risk_adjustment_min_risk_gain = kwargs.get('risk_adjustment_min_risk_gain', 0.25)
        self.risk_adjustment_max_model_gap_fraction = kwargs.get('risk_adjustment_max_model_gap_fraction', 0.25)
        self.risk_adjustment_max_model_rank = kwargs.get('risk_adjustment_max_model_rank', 3)
        self.risk_adjustment_response_samples = kwargs.get('risk_adjustment_response_samples', 240)
        self.rollout_enabled = kwargs.get('rollout_enabled', True)
        self.rollout_shadow_mode = kwargs.get('rollout_shadow_mode', True)
        self.rollout_max_worlds = kwargs.get('rollout_max_worlds', 8)
        self.rollout_min_worlds = kwargs.get('rollout_min_worlds', 3)
        self.rollout_max_steps = kwargs.get('rollout_max_steps', 72)
        self.rollout_time_budget_seconds = kwargs.get('rollout_time_budget_seconds', 8.0)
        self.rollout_min_ess_ratio = kwargs.get('rollout_min_ess_ratio', 0.35)
        self.rollout_min_value_gain = kwargs.get('rollout_min_value_gain', 0.15)
        self.rollout_device = kwargs.get('rollout_device', 'cpu')
        self.rollout_cpu_threads = kwargs.get('rollout_cpu_threads', 1)
        self.rollout_max_job_age_seconds = kwargs.get('rollout_max_job_age_seconds', 12.0)
        self.rollout_shutdown_timeout_seconds = kwargs.get('rollout_shutdown_timeout_seconds', 1.5)
        self.rollout_result_max_screen_age_seconds = kwargs.get('rollout_result_max_screen_age_seconds', 2.0)
    
    @classmethod
    def load(cls):
        if not os.path.exists('config.json'):
            return cls()
        with open('config.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)