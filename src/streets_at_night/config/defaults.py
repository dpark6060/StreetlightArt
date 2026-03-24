"""Default configuration values matching the TOML schema.

These defaults are used as the base layer when loading a config file.
The user's TOML only needs to override values they want to change.
"""

DEFAULTS: dict = {
    "camera": {
        "lat": 34.234880959491356,
        "lon": -117.31157772457728,
        "altitude_ft": 5186.0,
        "target_lat": 34.10834722093596,
        "target_lon": -117.29681238769862,
        "target_altitude_ft": 1000.0,
        "fov": 60.0,
        "near": 100.0,
        "far": 60000.0,
    },
    "bounds": {
        "north": 34.35,
        "south": 33.95,
        "east": -117.05,
        "west": -117.55,
    },
    "quality": {
        "low": {
            "width": 1920,
            "height": 1080,
            "illumination_resolution_multiplier": 0.5,
            "ground_tile_resolution": 4.0,
        },
        "medium": {
            "width": 3840,
            "height": 2160,
            "illumination_resolution_multiplier": 0.75,
            "ground_tile_resolution": 2.0,
        },
        "high": {
            "width": 7680,
            "height": 4320,
            "illumination_resolution_multiplier": 1.0,
            "ground_tile_resolution": 1.0,
        },
    },
    "seed": 42,
    "zones": {
        "house": {
            "exterior_colors": {
                "colors": [
                    ("#0D0A08", 1.0),
                    ("#0C0907", 1.0),
                    ("#0B0808", 1.0),
                    ("#0A0907", 1.0),
                    ("#0C0A09", 1.0),
                ]
            },
            "yard_base_color": "#060A04",
            "yard_variation": 2,
            "min_lights": 0,
            "max_lights": 5,
            "light_weights": [5.0, 3.0, 1.5, 0.5, 0.3, 0.1],
            "footprint_coverage": 0.65,
            "lot_width": 15.0,
            "lot_depth": 25.0,
            "setback": 10.0,
        },
        "colors": {
            "commercial": "#0A0A0C",
            "industrial": "#080808",
            "park_forest": "#040804",
            "water": "#040408",
            "road_surface": "#0C0C0C",
            "default_ground": "#080706",
        },
        "tile_size_meters": 256.0,
        "tile_resolution_px": 128,
    },
    "roads": {
        "export": {
            # Dark asphalt tones — roads are barely visible without illumination.
            # Slightly brighter than ground (#080706) so lit roads pop.
            "highway":     {"color": "#141210", "width": 4},
            "major_road":  {"color": "#121010", "width": 3},
            "collector":   {"color": "#100E0E", "width": 2},
            "residential": {"color": "#0E0C0C", "width": 1},
            "service":     {"color": "#0C0A0A", "width": 1},
            "minor":       {"color": "#0A0908", "width": 1},
        },
        "preview": {
            # Bright colors for interactive navigation
            "highway":     {"color": "#FFD700", "width": 4},
            "major_road":  {"color": "#FF6B35", "width": 3},
            "collector":   {"color": "#FFFFFF", "width": 2},
            "residential": {"color": "#888888", "width": 1},
            "service":     {"color": "#555555", "width": 1},
            "minor":       {"color": "#64647A", "width": 1},
        },
    },
    "lights": {
        "streetlights": {
            # Highway: always lit — 90% high-pressure sodium, 10% LED retrofit
            "highway": [
                {
                    "weight": 0.9,
                    "profile": {
                        "color": "#FFA040",
                        "height_min": 12.0,
                        "height_max": 15.0,
                        "brightness_min": 0.85,
                        "brightness_max": 1.0,
                        "glow_radius": 25.0,
                        "illumination_radius": 40.0,
                        "illumination_intensity": 0.9,
                        "spacing_min": 40.0,
                        "spacing_max": 60.0,
                        "side": "both",
                    },
                },
                {
                    "weight": 0.1,
                    "profile": {
                        "color": "#E0E8FF",
                        "height_min": 12.0,
                        "height_max": 15.0,
                        "brightness_min": 0.85,
                        "brightness_max": 1.0,
                        "glow_radius": 25.0,
                        "illumination_radius": 40.0,
                        "illumination_intensity": 0.9,
                        "spacing_min": 40.0,
                        "spacing_max": 60.0,
                        "side": "both",
                    },
                },
            ],
            # Major road: always lit — 55% sodium, 35% LED white, 10% warm LED
            "major_road": [
                {
                    "weight": 0.55,
                    "profile": {
                        "color": "#FFA040",
                        "height_min": 8.0,
                        "height_max": 10.0,
                        "brightness_min": 0.75,
                        "brightness_max": 0.9,
                        "glow_radius": 18.0,
                        "illumination_radius": 30.0,
                        "illumination_intensity": 0.75,
                        "spacing_min": 35.0,
                        "spacing_max": 50.0,
                        "side": "both",
                    },
                },
                {
                    "weight": 0.35,
                    "profile": {
                        "color": "#E0E8FF",
                        "height_min": 8.0,
                        "height_max": 10.0,
                        "brightness_min": 0.75,
                        "brightness_max": 0.9,
                        "glow_radius": 18.0,
                        "illumination_radius": 30.0,
                        "illumination_intensity": 0.75,
                        "spacing_min": 35.0,
                        "spacing_max": 50.0,
                        "side": "both",
                    },
                },
                {
                    "weight": 0.1,
                    "profile": {
                        "color": "#FFE080",
                        "height_min": 8.0,
                        "height_max": 10.0,
                        "brightness_min": 0.75,
                        "brightness_max": 0.9,
                        "glow_radius": 18.0,
                        "illumination_radius": 30.0,
                        "illumination_intensity": 0.75,
                        "spacing_min": 35.0,
                        "spacing_max": 50.0,
                        "side": "both",
                    },
                },
            ],
            # Collector: 40% sodium, 40% LED white, 20% warm yellow
            "collector": [
                {
                    "weight": 0.4,
                    "profile": {
                        "color": "#FFA040",
                        "height_min": 7.0,
                        "height_max": 9.0,
                        "brightness_min": 0.65,
                        "brightness_max": 0.85,
                        "glow_radius": 14.0,
                        "illumination_radius": 22.0,
                        "illumination_intensity": 0.65,
                        "spacing_min": 45.0,
                        "spacing_max": 65.0,
                        "side": "alternating",
                    },
                },
                {
                    "weight": 0.4,
                    "profile": {
                        "color": "#E0E8FF",
                        "height_min": 7.0,
                        "height_max": 9.0,
                        "brightness_min": 0.65,
                        "brightness_max": 0.85,
                        "glow_radius": 14.0,
                        "illumination_radius": 22.0,
                        "illumination_intensity": 0.65,
                        "spacing_min": 45.0,
                        "spacing_max": 65.0,
                        "side": "alternating",
                    },
                },
                {
                    "weight": 0.2,
                    "profile": {
                        "color": "#FFE080",
                        "height_min": 7.0,
                        "height_max": 9.0,
                        "brightness_min": 0.65,
                        "brightness_max": 0.85,
                        "glow_radius": 14.0,
                        "illumination_radius": 22.0,
                        "illumination_intensity": 0.65,
                        "spacing_min": 45.0,
                        "spacing_max": 65.0,
                        "side": "alternating",
                    },
                },
            ],
            # Residential: 25% no lights, 30% warm yellow, 25% LED white,
            #              15% sodium, 5% decorative warm
            "residential": [
                {"weight": 0.25, "profile": None},
                {
                    "weight": 0.3,
                    "profile": {
                        "color": "#FFE080",
                        "height_min": 5.0,
                        "height_max": 7.0,
                        "brightness_min": 0.5,
                        "brightness_max": 0.7,
                        "glow_radius": 10.0,
                        "illumination_radius": 15.0,
                        "illumination_intensity": 0.5,
                        "spacing_min": 80.0,
                        "spacing_max": 120.0,
                        "side": "one",
                    },
                },
                {
                    "weight": 0.25,
                    "profile": {
                        "color": "#E0E8FF",
                        "height_min": 5.0,
                        "height_max": 7.0,
                        "brightness_min": 0.5,
                        "brightness_max": 0.7,
                        "glow_radius": 10.0,
                        "illumination_radius": 15.0,
                        "illumination_intensity": 0.5,
                        "spacing_min": 80.0,
                        "spacing_max": 120.0,
                        "side": "one",
                    },
                },
                {
                    "weight": 0.15,
                    "profile": {
                        "color": "#FFA040",
                        "height_min": 5.0,
                        "height_max": 7.0,
                        "brightness_min": 0.5,
                        "brightness_max": 0.7,
                        "glow_radius": 10.0,
                        "illumination_radius": 15.0,
                        "illumination_intensity": 0.5,
                        "spacing_min": 80.0,
                        "spacing_max": 120.0,
                        "side": "one",
                    },
                },
                {
                    "weight": 0.05,
                    "profile": {
                        "color": "#FFCC80",
                        "height_min": 4.0,
                        "height_max": 5.0,
                        "brightness_min": 0.45,
                        "brightness_max": 0.65,
                        "glow_radius": 8.0,
                        "illumination_radius": 12.0,
                        "illumination_intensity": 0.45,
                        "spacing_min": 60.0,
                        "spacing_max": 80.0,
                        "side": "both",
                    },
                },
            ],
            # Service: 70% no lights, 20% LED white, 10% sodium
            "service": [
                {"weight": 0.7, "profile": None},
                {
                    "weight": 0.2,
                    "profile": {
                        "color": "#E0E8FF",
                        "height_min": 4.0,
                        "height_max": 6.0,
                        "brightness_min": 0.4,
                        "brightness_max": 0.6,
                        "glow_radius": 8.0,
                        "illumination_radius": 12.0,
                        "illumination_intensity": 0.4,
                        "spacing_min": 80.0,
                        "spacing_max": 150.0,
                        "side": "one",
                    },
                },
                {
                    "weight": 0.1,
                    "profile": {
                        "color": "#FFA040",
                        "height_min": 4.0,
                        "height_max": 6.0,
                        "brightness_min": 0.4,
                        "brightness_max": 0.6,
                        "glow_radius": 8.0,
                        "illumination_radius": 12.0,
                        "illumination_intensity": 0.4,
                        "spacing_min": 80.0,
                        "spacing_max": 150.0,
                        "side": "one",
                    },
                },
            ],
        },
        "cars": {
            # Headlight color pool: 50% bright white, 30% warm white,
            #                       15% yellow, 5% blue-white HID
            "headlight_colors": {
                "colors": [
                    ("#FFFFFF", 0.5),
                    ("#FFF4E0", 0.3),
                    ("#FFEE80", 0.15),
                    ("#CCE0FF", 0.05),
                ]
            },
            # Taillight color pool: 60% red, 25% dark red, 15% red-orange
            "taillight_colors": {
                "colors": [
                    ("#FF2020", 0.6),
                    ("#CC1010", 0.25),
                    ("#FF3010", 0.15),
                ]
            },
            "separation_min": 1.6,
            "separation_max": 2.4,
            # Spacing between cars in meters (min/max per road type)
            "density_highway_min": 25.0,
            "density_highway_max": 50.0,
            "density_major_min": 40.0,
            "density_major_max": 80.0,
            "density_collector_min": 80.0,
            "density_collector_max": 150.0,
            "density_residential_min": 150.0,
            "density_residential_max": 400.0,
            "density_service_min": 300.0,
            "density_service_max": 800.0,
        },
        "commercial": {
            # Sign colors: warm white 35%, red 12%, blue 12%, green 10%,
            #              yellow 10%, purple 8%, orange 8%, pink 5%
            "sign_colors": {
                "colors": [
                    ("#FFF8E0", 0.35),
                    ("#FF2020", 0.12),
                    ("#2040FF", 0.12),
                    ("#20CC40", 0.10),
                    ("#FFE020", 0.10),
                    ("#8020CC", 0.08),
                    ("#FF8020", 0.08),
                    ("#FF40A0", 0.05),
                ]
            },
            "sign_brightness_min": 0.6,
            "sign_brightness_max": 0.9,
            "sign_glow_radius": 12.0,
            "sign_illumination_radius": 20.0,
            # Parking lot: 70% LED white, 30% sodium
            "parking_colors": {
                "colors": [
                    ("#E0E8FF", 0.7),
                    ("#FFA040", 0.3),
                ]
            },
            "parking_brightness_min": 0.75,
            "parking_brightness_max": 0.9,
            "parking_glow_radius": 20.0,
            "parking_illumination_radius": 30.0,
            "parking_grid_spacing_min": 20.0,
            "parking_grid_spacing_max": 30.0,
            # Storefront: 60% warm white, 25% warm yellow, 15% fluorescent
            "storefront_colors": {
                "colors": [
                    ("#FFE8CC", 0.6),
                    ("#FFE080", 0.25),
                    ("#F0F0FF", 0.15),
                ]
            },
            "storefront_brightness_min": 0.5,
            "storefront_brightness_max": 0.7,
            "storefront_glow_radius": 8.0,
            "storefront_illumination_radius": 12.0,
            "storefront_spacing_min": 5.0,
            "storefront_spacing_max": 8.0,
        },
        "house_lights": {
            # Color pool: 45% warm yellow, 25% warm white, 15% amber, 15% TV blue
            "colors": {
                "colors": [
                    ("#FFE4A0", 0.45),
                    ("#FFF0D0", 0.25),
                    ("#FFCC80", 0.15),
                    ("#D0E0FF", 0.15),
                ]
            },
            "brightness_min": 0.3,
            "brightness_max": 0.6,
            "glow_radius": 6.0,
            "illumination_radius": 8.0,
            "illumination_intensity": 0.35,
        },
    },
}
