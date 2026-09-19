import numpy as np

class FairnessValidator:
    def __init__(self, target_cv=0.10, allowed_ratio_error=0.3):
        self.target_cv = target_cv
        self.allowed_ratio_error = allowed_ratio_error
        
    def calculate_cv(self, per_stream_fps_values):
        """
        Calculate coefficient of variation (CV = std / mean).
        Expects a list or array of FPS values.
        """
        if not per_stream_fps_values or len(per_stream_fps_values) < 2:
            return 0.0
            
        std_dev = np.std(per_stream_fps_values)
        mean = np.mean(per_stream_fps_values)
        
        if mean == 0:
            return 0.0
            
        return float(std_dev / mean)
        
    def check_proportionality(self, stream_fps_dict, weights_dict):
        """
        Compare actual frame ratios vs weight ratios.
        Returns detailed mismatch dict if unfair.
        """
        # Pick a baseline stream (e.g., stream 0 or first available with non-zero weight)
        baseline_stream = None
        for sid, w in weights_dict.items():
            if w > 0:
                baseline_stream = sid
                break
                
        if baseline_stream is None:
            return True, {} # No streams have weight? Valid I suppose.
            
        base_weight = weights_dict[baseline_stream]
        base_fps = stream_fps_dict.get(baseline_stream, 0.0)
        
        if base_fps == 0:
            # Cannot compare proportionality if baseline has 0 FPS
            return False, {"error": f"Baseline stream {baseline_stream} has 0 FPS"}
            
        discrepancies = {}
        is_proportional = True
        
        for sid, weight in weights_dict.items():
            if weight == 0:
                continue
                
            expected_ratio = weight / base_weight
            actual_fps = stream_fps_dict.get(sid, 0.0)
            actual_ratio = actual_fps / base_fps
            
            # Check if within allowed error
            if abs(actual_ratio - expected_ratio) > self.allowed_ratio_error:
                is_proportional = False
                discrepancies[sid] = {
                    "expected_ratio": expected_ratio,
                    "actual_ratio": actual_ratio,
                    "fps": actual_fps
                }
                
        return is_proportional, discrepancies

    def check_starvation(self, stream_fps_dict):
        """
        Flag any stream with 0 frames processed.
        """
        starved_streams = [sid for sid, fps in stream_fps_dict.items() if fps <= 0.0]
        return len(starved_streams) == 0, starved_streams

    def validate_fairness(self, stream_fps_dict, weights_dict=None):
        """
        Run all checks, return {is_fair, cv, starvation, proportionality}.
        """
        fps_values = list(stream_fps_dict.values())
        
        # 1. Calculate CV (Only makes sense if weights are equal, but we log it anyway)
        cv = self.calculate_cv(fps_values)
        cv_fair = cv <= self.target_cv
        
        # 2. Check Starvation
        no_starvation, starved_streams = self.check_starvation(stream_fps_dict)
        
        # 3. Check Proportionality (if weights provided)
        is_proportional = True
        discrepancies = {}
        if weights_dict:
            is_proportional, discrepancies = self.check_proportionality(stream_fps_dict, weights_dict)
            
        # Overall assessment
        # If weights are uniformly 1.0, CV is the main metric. 
        # If weights vary, proportionality is the main metric.
        weights_uniform = True
        if weights_dict:
            w_vals = list(weights_dict.values())
            weights_uniform = all(w == w_vals[0] for w in w_vals)
            
        overall_fair = False
        if weights_uniform:
            overall_fair = cv_fair and no_starvation
        else:
            overall_fair = is_proportional and no_starvation

        return {
            "is_fair": overall_fair,
            "cv": cv,
            "cv_pass": cv_fair,
            "no_starvation": no_starvation,
            "starved_streams": starved_streams,
            "is_proportional": is_proportional,
            "proportionality_errors": discrepancies
        }
