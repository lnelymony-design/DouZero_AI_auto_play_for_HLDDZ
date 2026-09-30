import base64
import zlib
from collections import Counter

import cv2
import numpy as np


REFERENCE_WIDTH = 1291
REFERENCE_HEIGHT = 770

# Normalized 40x52 binary glyph prototypes extracted from the user's WeChat
# miniapp screenshots. Red/black suits are intentionally discarded; only rank
# shapes are retained.
_RANK_TEMPLATE_B64 = {'2': 'eNrd1DsOgDAMA9Dc/9KGiQHs2BJUUfFYvaZVP6laHZxJzJVMSQmSht2mcManPZyq3zAByRLMNVuOHVY7pA6pQ+pgzoVBvHElnbsk+VRD5h/Br9nGWxtkbDhkvmGx4aZc4PQ/N01/yGHIfZgDvMOOjg==', 'A': 'eNrN1MsOwCAIRFH+/6dvky66UIqD71nqiZogmF0S3qiO3Y6kY68DDR5yZJzwFzLOso6max34ba5xBE7pJnSntCf9jlFXRmQeXOGEQcNcV6/5L+x39uuEkcmAcy4mOo94hpbV86tZV3m2uzsPCUw54w==', 'Q': 'eNrtlTsOwCAMQ7n/pd0OlfgoNkZpNjKaJ0IkYrdWXXjLgr7yKEmaHGCB46EA5xMKrjrhAjkkTS6jJd6CHzicc7jc5Qr+VVd2ux4aC92jGYTk+i3gC4yx3XY26YGEUy6pHRXrANR5TX9O5YLbOp82Bzk3gm5yFqX3A7G3yFQ=', 'J': 'eNrt1bESADAEA9D8/0/rqj1Kr0ySkTcwBWiLOOl2hs2dmfwHdHR0dAOcHvtun3+7YxE5vLnLu3ohoZOIGZWV7MqKuqrJApe3/RE=', 'T': 'eNrt1UsOgCAQA9De/9J1YSBMLaMoGwyzEnh8YrQAvysythg72gFhFrbdpzEwTi+PAllLlw6Q6twM49hxcr7LCQz0ji9c3YEfHLsOgw7bbbeAm/Xd3/9vmO3wMF/SvIIGm3cmAbOYZPNyyhgyB5P31iX3x9p1AEBH0Us=', '9': 'eNrVlcEOwCAIQ/v/P80Oyw6TAp2oyTjCM6JQADab3SZBBWqjiRgHX6EYHAIR5/wcJPdQkDgZZ4mzwaHDlW8L/+r3HDZxvYLMNEKSC+9A8+IrVZlLpBbxGI3Fbkzsa0bXMc4fjOsBsTcwM2C+cTgwcyBKq8rucQs1o1uhaL18b5m43qBRi+wCt3uwbA==', '7': 'eNrtlMEKACAIQ/f/P233zJgw0aAd5TGZikCB7K7hnMsRxNsLwRjSGEgsxREY4qy+eOTYy1BhJXYYbWeP27WmUHNNbWWcfe7FtdGfRaAFiBLXNw==', '6': 'eNrVlVEOgDAIQ3v/S9fEHzOlpVE3I5/wxsJWAJhu3C2COpSjhZgih6AGzxHBXROUYHGP4dBxKh3bdGXB/IbTZ9FziF7FKeAt7qy2RHhOpEfAq5nSHDc4BKdrG7jsy4nGyVBDazjc5vBLLm3MFb4HA8b3SN/T1bQ3855OGao7w7WV7cGpy3sD5syieg==', '5': 'eNrtlUEOgEAIA/v/T+NJoxGGmogelCMZwmZbQJqJKGOUi8e45Hnut9zM6aMclaxcKdiBQ2nJB4hB81O+cT1ldlku3ZJdj9x+Lif675+LTo+Ec0V/mWv9co0z7Wdwqhe7eD6aQefpLReHeTC906rRWACGl4iU', '3': 'eNrN1csOwCAIRFH+/6en2xoGudqalqU5EF9oxMnQLZiqpXLM2JBRMZeVXVE+Dbt5NOW481CwoOAEzfJecSILdvVi06UtbG/hV4x0CGfYcWhPsZE/htaJNdIk1122zumQC+aCuc0WVLV/zxx81rr3r7ot43D7gSx8SP1XyNRiRz6MC83DYro=', '8': 'eNrl1TEWgDAIA1Duf2kcWy0h4YGDT0b8XSqhZm+Wb9V2fpTIQnn/gqXmgm7myNHq2b85Z676P2S3+mxePFVgTq3hTEKn7Oatk0vI2Fp4NhEUXTJqnrbCrrojii6/q+84lJnprM7sDSnnwjvDnJ5zNb8jdQHEivkj', '4': 'eNrN1NEKgDAIheG9/0v/EVFUzHk2NPPy+MFgoq19VuwV55AclDg0R4kDyVHjjq7nzq7ilLmJ7mqNHROuhbl7PnCP2HYkOH+P3pnhSHDCnveT/hvRzq5q96O7m3bHvYulQM2p/zw3j9XaANw4b60=', 'K': 'eNrF1UsOwCAIBFDuf+nptgGGX0tgSV7UsVZFrgq6bPfIuUs0PTeLbrHMbecy4+Az7VBzTA0dZ56TxAWjvV2kBg47Ls2x5pJ9hoTww/ctn5ffzl8QZfZ/8AGHjsKpY1H2XfOeVLDYS+7sbi99P1B8Z1quOu9NPRmQerA='}
_COUNT_DIGIT_TEMPLATE_B64 = {
    "0": "eJyt0cEKACAIA9D9/08vIijMWZLtlM9LKvAYjmSdM1ptJ3BqX5VtlN/0js2RdjXLTz//8zZXfSe9lIdx9wodWk0Lf9MA1u4p5Q==",
    "1": "eJxjYCAT/AcDosX/wwB2USLFcZmDJDcqPiLFsQgTBQBDIqBg",
    "2": "eJytkdEKACAIA/3/n14QQZRzreie5BoLNOIRdEyPSaGXF+6xk36s/Jgd/2E+7ohsQpTo8uRluRvXYdOzy66e3cvw0PkrGvGvHPI=",
    "3": "eJytkcEKACAIQ/3/n16HiHA502gnfQ5kavYoTNU4tgR2kxqH51Stjh0Zp02fMsdmzeXtxNpGgBJ3Fubocpye/I9RXoEvN6hoAJBMEP4=",
    "4": "eJytz0EKACAIRFHvf+kpiCwaA2X8u3nSIjMhzAoORAd4kqPHF8TOP9g76b5yfkbB72L1F1Wng719uNVZlQYz3hn1",
    "5": "eJylksEOACAIQvn/n6ZLtbQoNW6+NgYmUBSnApxGijPJvc85pAqvO126MstDdUeeMHeO6uGfr/7bR+DJ+2Dzuy0QRQ59hHE1NyEc8g==",
    "6": "eJylkEsSADAEQ93/0um0qxZRn+y8GILIRAAKHEcEp7mP4yG1LCyJsf78saA14vZs2xRxb2jEU3sdviv1CnIX5fSjlyE9LRShJ+c=",
    "7": "eJxjYCAD/EcF+MX/YwHkiGOTweEyXC4mWpwSYxDytLUUbxBgmoJDnMSAxK+cWMeQZQo2PzHgEicaAAC8KPIO",
    "8": "eJxjYCAP/IcAYsX/IwB2URQZHOL/8YqjM6knjuBhdScucQL+ojgc0EL7PyaHCHFUu3AEyX9yzMflTsx4JCSOM3yQpRjIAwCgUW2h",
    "9": "eJzNkDEOADAIAv3/p+ngYgQTdOqNhwnEiCNIPI/KoEugPTqisQZ89IPnnds/UDB92vF9GHltfX/ptLekF9rmActZIO4="
}

_LANDLORD_FLAG_B64 = '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'
_PASS_TEMPLATE_B64 = 'eNq9l9sOAyEIRPn/n54m3SZVGe62vq3CHlHBUeR2A+QPDe/2JwyCYJ9xbYUiBtF4sm2OUTzLN7oc9VcYPbN4EHN2lx7HxMACX142kyPtU5DbHmQOKfJJOeLIbzlZzJCD9Ezy6dPaHDtZpMEp1eb6smUwcw6QD2fAQYlTS23uvpf1Nif0duMecbozqnFKc7qIkeakfA4xwByU8Coc8twVcP5vdUmDuBGipGHiwRW43IaqkmOxnPpg/dPj6G2z1YNTZ4NbwVZMXsXLxcO3+QzAK60mx+hydBm8+zW1bPJk+leJE3VOprbKdm2xjxOOrv3M5RTk6mATp6fLFgrrOH1gfDhS49B3VXSXao52Ilbaxa8xQjlx1/jFDItz+wXsD78AmPImLw=='


class WechatCardRecognizer:
    def __init__(self):
        self.rank_templates = {
            rank: self._decode_template(data, (52, 40))
            for rank, data in _RANK_TEMPLATE_B64.items()
        }
        self.count_digit_templates = {
            digit: self._decode_template(data, (32, 24))
            for digit, data in _COUNT_DIGIT_TEMPLATE_B64.items()
        }
        self.landlord_flag = self._decode_template(
            _LANDLORD_FLAG_B64, (81, 41)
        )
        self.pass_template = self._decode_template(
            _PASS_TEMPLATE_B64, (40, 104)
        )

    @staticmethod
    def _decode_template(data, shape):
        raw = zlib.decompress(base64.b64decode(data))
        return np.frombuffer(raw, dtype=np.uint8).reshape(shape)

    @staticmethod
    def _to_bgr(image):
        if image is None:
            return None
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return image
        arr = np.asarray(image)
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _ink_mask(bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        dark = gray < 115
        red = (
            (sat > 110)
            & (val > 70)
            & ((hue < 15) | (hue > 170))
        )
        return ((dark | red).astype(np.uint8) * 255)

    @staticmethod
    def _normalize_mask(mask, out_width=40, out_height=52, pad=2):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return np.zeros((out_height, out_width), dtype=np.uint8)

        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1
        crop = mask[y0:y1, x0:x1]

        scale = min(
            (out_width - 2 * pad) / crop.shape[1],
            (out_height - 2 * pad) / crop.shape[0],
        )
        width = max(1, int(round(crop.shape[1] * scale)))
        height = max(1, int(round(crop.shape[0] * scale)))
        resized = cv2.resize(
            crop, (width, height), interpolation=cv2.INTER_NEAREST
        )

        canvas = np.zeros((out_height, out_width), dtype=np.uint8)
        left = (out_width - width) // 2
        top = (out_height - height) // 2
        canvas[top:top + height, left:left + width] = resized
        return canvas

    @staticmethod
    def _iou(a, b):
        aa = a > 0
        bb = b > 0
        union = np.logical_or(aa, bb).sum()
        if union == 0:
            return 0.0
        return float(np.logical_and(aa, bb).sum() / union)

    def _classify_rank(self, normalized):
        best_rank = None
        best_score = -1.0
        for rank, prototype in self.rank_templates.items():
            iou = self._iou(normalized, prototype)
            corr = float(
                cv2.matchTemplate(
                    normalized, prototype, cv2.TM_CCOEFF_NORMED
                )[0, 0]
            )
            score = 0.7 * iou + 0.3 * corr
            if score > best_score:
                best_rank = rank
                best_score = score
        return best_rank, best_score

    @staticmethod
    def _scaled(value, current, reference):
        return max(1, int(round(value * current / reference)))

    def _recognize_rank_band(
        self,
        bgr,
        region,
        min_h_ref,
        max_h_ref,
        min_area_ref,
        score_threshold=0.62,
    ):
        height, width = bgr.shape[:2]
        x0 = int(region[0] * width)
        x1 = int(region[1] * width)
        y0 = int(region[2] * height)
        y1 = int(region[3] * height)

        roi = bgr[y0:y1, x0:x1]
        if roi.size == 0:
            return []

        mask = self._ink_mask(roi)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

        min_h = self._scaled(min_h_ref, height, REFERENCE_HEIGHT)
        max_h = self._scaled(max_h_ref, height, REFERENCE_HEIGHT)
        min_area = max(
            20,
            int(
                min_area_ref
                * (width / REFERENCE_WIDTH)
                * (height / REFERENCE_HEIGHT)
            ),
        )

        components = []
        for index in range(1, count):
            x, y, w, h, area = [int(v) for v in stats[index]]
            if (
                min_h <= h <= max_h
                and area >= min_area
                and self._scaled(8, width, REFERENCE_WIDTH)
                <= w
                <= self._scaled(55, width, REFERENCE_WIDTH)
            ):
                components.append(
                    {
                        "x": x + x0,
                        "y": y + y0,
                        "w": w,
                        "h": h,
                        "area": area,
                    }
                )

        components.sort(key=lambda item: item["x"])

        # "10" is the only normal rank made of two disconnected glyphs.
        groups = []
        used = set()
        max_center_y_delta = self._scaled(4, height, REFERENCE_HEIGHT)
        max_pair_dx = self._scaled(30, width, REFERENCE_WIDTH)
        max_pair_gap = self._scaled(5, width, REFERENCE_WIDTH)

        for i, component in enumerate(components):
            if i in used:
                continue

            paired_index = None
            for j in range(i + 1, min(i + 3, len(components))):
                if j in used:
                    continue
                other = components[j]
                cy1 = component["y"] + component["h"] / 2
                cy2 = other["y"] + other["h"] / 2
                x_gap = other["x"] - (component["x"] + component["w"])

                if (
                    abs(cy1 - cy2) <= max_center_y_delta
                    and abs(component["h"] - other["h"])
                    <= max_center_y_delta
                    and x_gap <= max_pair_gap
                    and other["x"] - component["x"] <= max_pair_dx
                ):
                    paired_index = j
                    break

            if paired_index is not None:
                other = components[paired_index]
                used.add(i)
                used.add(paired_index)
                left = min(component["x"], other["x"])
                top = min(component["y"], other["y"])
                right = max(
                    component["x"] + component["w"],
                    other["x"] + other["w"],
                )
                bottom = max(
                    component["y"] + component["h"],
                    other["y"] + other["h"],
                )
                groups.append((left, top, right - left, bottom - top))
            else:
                groups.append(
                    (
                        component["x"],
                        component["y"],
                        component["w"],
                        component["h"],
                    )
                )

        full_mask = self._ink_mask(bgr)
        results = []
        for x, y, w, h in groups:
            symbol_mask = full_mask[y:y + h, x:x + w]
            normalized = self._normalize_mask(symbol_mask)
            rank, score = self._classify_rank(normalized)
            if rank is not None and score >= score_threshold:
                results.append((x, rank, score))

        results.sort(key=lambda item: item[0])
        return results

    def _detect_jokers(self, bgr, region, compact=False):
        height, width = bgr.shape[:2]
        x0 = int(region[0] * width)
        x1 = int(region[1] * width)
        y0 = int(region[2] * height)
        y1 = int(region[3] * height)

        roi = bgr[y0:y1, x0:x1]
        if roi.size == 0:
            return []

        mask = self._ink_mask(roi)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

        if compact:
            # Opponent-play Jokers are much smaller than Jokers in the local
            # hand.  The 3rd recording contains a small Joker at ~104s whose
            # individual J/O/K/E/R glyphs are only 6-15 px wide.
            min_w = self._scaled(4, width, REFERENCE_WIDTH)
            max_w = self._scaled(35, width, REFERENCE_WIDTH)
            min_h = self._scaled(8, height, REFERENCE_HEIGHT)
            max_h = self._scaled(30, height, REFERENCE_HEIGHT)
            min_area = max(
                12,
                int(
                    25
                    * (width / REFERENCE_WIDTH)
                    * (height / REFERENCE_HEIGHT)
                ),
            )
        else:
            min_w = self._scaled(15, width, REFERENCE_WIDTH)
            max_w = self._scaled(40, width, REFERENCE_WIDTH)
            min_h = self._scaled(14, height, REFERENCE_HEIGHT)
            max_h = self._scaled(40, height, REFERENCE_HEIGHT)
            min_area = max(
                15,
                int(
                    120
                    * (width / REFERENCE_WIDTH)
                    * (height / REFERENCE_HEIGHT)
                ),
            )

        components = []
        for index in range(1, count):
            x, y, w, h, area = [int(v) for v in stats[index]]
            if (
                min_w <= w <= max_w
                and min_h <= h <= max_h
                and area >= min_area
            ):
                components.append(
                    (
                        x + x0,
                        y + y0,
                        w,
                        h,
                        area,
                        x + x0 + w / 2,
                        y + y0 + h / 2,
                    )
                )

        clusters = []
        x_tolerance = self._scaled(
            12 if compact else 8, width, REFERENCE_WIDTH
        )
        for component in sorted(components, key=lambda item: item[5]):
            assigned = False
            for cluster in clusters:
                mean_x = sum(item[5] for item in cluster) / len(cluster)
                if abs(component[5] - mean_x) <= x_tolerance:
                    cluster.append(component)
                    assigned = True
                    break
            if not assigned:
                clusters.append([component])

        results = []
        min_span = self._scaled(
            45 if compact else 55, height, REFERENCE_HEIGHT
        )

        for cluster in clusters:
            center_ys = [item[6] for item in cluster]
            span = max(center_ys) - min(center_ys) if len(center_ys) > 1 else 0

            # A normal card contributes at most rank + suit components. The
            # vertical JOKER word produces a stable stack of 4+ components.
            if len(cluster) < 4 or span < min_span:
                continue

            left = min(item[0] for item in cluster)
            top = min(item[1] for item in cluster)
            right = max(item[0] + item[2] for item in cluster)
            bottom = max(item[1] + item[3] for item in cluster)

            # The actual vertical JOKER word is narrow.  With the relaxed
            # compact thresholds, UI text/buttons can otherwise form wide fake
            # vertical clusters.  In the recorded small Joker the cluster is
            # ~15 px wide; 26 px leaves comfortable scaling margin.
            if compact:
                max_cluster_width = self._scaled(
                    26, width, REFERENCE_WIDTH
                )
                if right - left > max_cluster_width:
                    continue

            patch = bgr[top:bottom, left:right]
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

            red = (
                (hsv[:, :, 1] > 100)
                & (hsv[:, :, 2] > 70)
                & ((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 170))
            ).sum()
            dark = (gray < 115).sum()

            rank = "D" if red > max(20, dark * 0.25) else "X"
            results.append((left, rank, 1.0))

        results.sort(key=lambda item: item[0])
        return results

    @staticmethod
    def _merge_cards(normal_cards, jokers):
        merged = list(normal_cards) + list(jokers)
        merged.sort(key=lambda item: item[0])

        # Joker letter fragments can occasionally cause one low-confidence
        # normal-rank candidate at the same x position. Keep the Joker.
        cleaned = []
        for item in merged:
            if item[1] in ("D", "X"):
                cleaned = [
                    old for old in cleaned
                    if abs(old[0] - item[0]) > 35
                ]
                cleaned.append(item)
                continue

            if any(
                joker[1] in ("D", "X")
                and abs(joker[0] - item[0]) <= 35
                for joker in merged
            ):
                continue
            cleaned.append(item)

        cleaned.sort(key=lambda item: item[0])
        return "".join(item[1] for item in cleaned)

    def recognize_my_hand(self, image, is_landlord=False):
        bgr = self._to_bgr(image)
        if bgr is None:
            return ""

        # Video calibration: both 17-card and 20-card layouts are safe to scan
        # almost edge-to-edge.  Selected cards rise by roughly 20-30 px, so the
        # rank band must start higher than the old 0.655 crop.  This wider band
        # recovered every card in the recorded 20-card landlord hand, including
        # a raised 7 that the old crop dropped.
        rank_region = (0.005, 0.995, 0.60, 0.755)
        joker_region = (0.005, 0.995, 0.60, 0.87)

        normal = self._recognize_rank_band(
            bgr,
            region=rank_region,
            min_h_ref=40,
            max_h_ref=58,
            min_area_ref=240,
            score_threshold=0.62,
        )
        jokers = self._detect_jokers(
            bgr, region=joker_region
        )
        return self._merge_cards(normal, jokers)

    def recognize_bottom_cards(self, image):
        bgr = self._to_bgr(image)
        if bgr is None:
            return ""

        # 1291x770 WeChat layout: the three landlord cards are centered around
        # x ~= 0.46..0.53.  The previous crop started at x=0.49, which cut off
        # the first two cards in samples such as 9-6-3 and left only the final
        # "3".  Keep the normal-rank band vertically tight so the multiplier
        # badge below the cards is not treated as a fourth rank.
        normal = self._recognize_rank_band(
            bgr,
            region=(0.455, 0.545, 0.055, 0.115),
            min_h_ref=24,
            max_h_ref=46,
            min_area_ref=90,
            score_threshold=0.62,
        )
        jokers = self._detect_jokers(
            bgr, region=(0.45, 0.55, 0.05, 0.18)
        )
        return self._merge_cards(normal, jokers)

    def recognize_left_played(self, image, expected_count=None):
        # Use a narrow strip by default so the previous centre-table play is
        # not mistaken for the left player's next single.  Only long
        # combinations widen toward the centre/lower row.
        region = (
            (0.08, 0.46, 0.25, 0.47)
            if expected_count is None or expected_count <= 4
            else (0.08, 0.48, 0.25, 0.52)
        )
        return self._recognize_played_with_expected_count(
            image, region, expected_count
        )

    def recognize_right_played(self, image, expected_count=None):
        # The recordings include a 10-card right-side play (AKQJ1098765).
        # Small plays use a centre-safe crop; long plays widen only when the
        # remaining-card drop tells us how many cards to expect.
        region = (
            (0.54, 0.92, 0.25, 0.47)
            if expected_count is None or expected_count <= 4
            else (0.52, 0.92, 0.25, 0.52)
        )
        return self._recognize_played_with_expected_count(
            image, region, expected_count
        )

    def recognize_my_played(self, image, expected_count=None):
        return self._recognize_played_with_expected_count(
            image, (0.32, 0.68, 0.38, 0.62), expected_count
        )

    def _recognize_played_with_expected_count(
        self, image, region, expected_count=None
    ):
        if expected_count is None:
            return self._recognize_played_region(image, region)

        # Remaining-card drops tell us exactly how many cards were played.
        # Re-scan at several thresholds and accept a candidate only when the
        # visible card count agrees with that hard observation.
        for threshold in (0.68, 0.64, 0.60, 0.56):
            cards = self._recognize_played_region(
                image, region, score_threshold=threshold
            )
            if len(cards) == expected_count:
                return cards
        return ""

    def _recognize_played_region(self, image, region, score_threshold=0.64):
        bgr = self._to_bgr(image)
        if bgr is None:
            return ""

        normal = self._recognize_rank_band(
            bgr,
            region=region,
            min_h_ref=28,
            max_h_ref=50,
            min_area_ref=110,
            score_threshold=score_threshold,
        )
        # Played Jokers use the same vertical word treatment but may be smaller.
        joker_region = (
            region[0],
            region[1],
            region[2],
            min(0.65, region[3] + 0.11),
        )
        jokers = self._detect_jokers(
            bgr, joker_region, compact=True
        )
        return self._merge_cards(normal, jokers)

    def _classify_count_digit(self, normalized):
        best_digit = None
        best_score = -1.0
        for digit, prototype in self.count_digit_templates.items():
            iou = self._iou(normalized, prototype)
            corr = float(
                cv2.matchTemplate(
                    normalized, prototype, cv2.TM_CCOEFF_NORMED
                )[0, 0]
            )
            score = 0.7 * iou + 0.3 * corr
            if score > best_score:
                best_digit = digit
                best_score = score
        return best_digit, best_score

    def recognize_remaining_count(self, image, side, expected=None):
        """Read the blue left/right remaining-card badge.

        Digit prototypes come from the user's WeChat miniapp screenshots and
        the three supplied recordings; 0-9 now all have direct glyph samples.
        The optional expected value is retained only as a narrow fallback for
        heavily animated/partially occluded badges.
        """
        bgr = self._to_bgr(image)
        if bgr is None or side not in ("left", "right"):
            return None

        height, width = bgr.shape[:2]
        # Read only the inner digit strip.  The previous wider crop included
        # the bright blue badge border and caused false values such as 14 -> 5.
        if side == "left":
            region = (0.107, 0.130, 0.452, 0.500)
        else:
            region = (0.871, 0.897, 0.452, 0.500)

        x0 = int(region[0] * width)
        x1 = int(region[1] * width)
        y0 = int(region[2] * height)
        y1 = int(region[3] * height)
        roi = bgr[y0:y1, x0:x1]
        if roi.size == 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mask = (
            (gray > 175)
            & (hsv[:, :, 1] < 120)
        ).astype(np.uint8) * 255

        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        min_h = self._scaled(14, height, REFERENCE_HEIGHT)
        max_h = self._scaled(28, height, REFERENCE_HEIGHT)
        min_area = max(
            20,
            int(
                35
                * (width / REFERENCE_WIDTH)
                * (height / REFERENCE_HEIGHT)
            ),
        )

        components = []
        for index in range(1, count):
            x, y, w, h, area = [int(v) for v in stats[index]]
            if (
                min_h <= h <= max_h
                and self._scaled(3, width, REFERENCE_WIDTH)
                <= w
                <= self._scaled(20, width, REFERENCE_WIDTH)
                and area >= min_area
            ):
                components.append((x, y, w, h, area))

        components.sort(key=lambda item: item[0])
        if not components:
            return None

        digits = []
        confidence = 1.0
        for x, y, w, h, _ in components:
            patch = mask[y:y + h, x:x + w]
            normalized = self._normalize_mask(
                patch, out_width=24, out_height=32, pad=2
            )
            digit, score = self._classify_count_digit(normalized)
            if digit is None:
                return None
            digits.append(digit)
            confidence = min(confidence, score)

        try:
            value = int("".join(digits))
        except ValueError:
            return None

        if 0 <= value <= 20 and confidence >= 0.56:
            return value

        if expected in (3, 8) and len(components) == 1:
            return int(expected)

        return None

    def detect_landlord_side(self, image):
        bgr = self._to_bgr(image)
        if bgr is None:
            return None

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        scale = min(
            width / REFERENCE_WIDTH,
            height / REFERENCE_HEIGHT,
        )
        template = self.landlord_flag
        if abs(scale - 1.0) > 0.02:
            template = cv2.resize(
                template,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
            )

        if gray.shape[0] < template.shape[0] or gray.shape[1] < template.shape[1]:
            return None

        response = cv2.matchTemplate(
            gray, template, cv2.TM_CCOEFF_NORMED
        )
        _, score, _, location = cv2.minMaxLoc(response)
        if score < 0.72:
            return None

        center_x = location[0] + template.shape[1] / 2
        center_y = location[1] + template.shape[0] / 2

        if center_y > height * 0.55:
            return "me"
        return "left" if center_x < width * 0.5 else "right"

    def detect_pass_sides(self, image):
        bgr = self._to_bgr(image)
        if bgr is None:
            return set()

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = (
            (hsv[:, :, 0] >= 90)
            & (hsv[:, :, 0] <= 115)
            & (hsv[:, :, 1] < 125)
            & (hsv[:, :, 2] > 180)
        ).astype(np.uint8) * 255

        height, width = mask.shape[:2]
        scale = min(
            width / REFERENCE_WIDTH,
            height / REFERENCE_HEIGHT,
        )
        template = self.pass_template
        if abs(scale - 1.0) > 0.02:
            template = cv2.resize(
                template,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_NEAREST,
            )

        if mask.shape[0] < template.shape[0] or mask.shape[1] < template.shape[1]:
            return set()

        response = cv2.matchTemplate(
            mask, template, cv2.TM_CCOEFF_NORMED
        )
        ys, xs = np.where(response >= 0.72)
        if len(xs) == 0:
            return set()

        points = []
        for x, y in sorted(
            zip(xs.tolist(), ys.tolist()),
            key=lambda point: float(response[point[1], point[0]]),
            reverse=True,
        ):
            cx = x + template.shape[1] / 2
            cy = y + template.shape[0] / 2
            if all(
                abs(cx - old_x) > template.shape[1] * 0.6
                or abs(cy - old_y) > template.shape[0] * 0.6
                for old_x, old_y in points
            ):
                points.append((cx, cy))
            if len(points) >= 3:
                break

        sides = set()
        for center_x, center_y in points:
            # Pass text for the local player is near the horizontal center.
            if width * 0.42 <= center_x <= width * 0.58:
                sides.add("me")
            elif center_x < width * 0.5:
                sides.add("left")
            else:
                sides.add("right")
        return sides
