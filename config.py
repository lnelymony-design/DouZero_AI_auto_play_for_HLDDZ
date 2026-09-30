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
    risk_adjustment_enabled: bool
    risk_adjustment_weight: float
    risk_adjustment_min_ess_ratio: float
    
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
        self.risk_adjustment_enabled = kwargs.get('risk_adjustment_enabled', True)
        self.risk_adjustment_weight = kwargs.get('risk_adjustment_weight', 0.45)
        self.risk_adjustment_min_ess_ratio = kwargs.get('risk_adjustment_min_ess_ratio', 0.40)
    
    @classmethod
    def load(cls):
        if not os.path.exists('config.json'):
            return cls()
        with open('config.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)