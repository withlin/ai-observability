#!/usr/bin/env python3
"""Estimate a daily and total travel budget from city, days, and style."""
import argparse
import json

DAILY = {
    "tokyo": {"budget": (70, 110), "mid-range": (140, 220), "comfort": (260, 420)},
    "lisbon": {"budget": (55, 90), "mid-range": (110, 180), "comfort": (210, 340)},
    "toronto": {"budget": (75, 120), "mid-range": (150, 240), "comfort": (280, 450)},
    "berlin": {"budget": (60, 95), "mid-range": (120, 190), "comfort": (220, 360)},
}
DEFAULT_DAILY = {"budget": (60, 100), "mid-range": (120, 200), "comfort": (220, 380)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--days", type=int, required=True)
    parser.add_argument("--style", choices=["budget", "mid-range", "comfort"], default="mid-range")
    args = parser.parse_args()
    city = args.city.strip().lower()
    low, high = DAILY.get(city, DEFAULT_DAILY)[args.style]
    result = {
        "city": args.city.strip(),
        "days": args.days,
        "style": args.style,
        "currency": "USD",
        "daily_low": low,
        "daily_high": high,
        "total_low": low * args.days,
        "total_high": high * args.days,
        "note": "Planning estimate; excludes international flights.",
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
