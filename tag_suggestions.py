#!/usr/bin/env python3
"""Helper script to generate tag suggestions for audio features."""

# Popular audio feature tags that users can reference
POPULAR_AUDIO_TAGS = {
    # MP3-specific tags
    "tcop": "Track Commercial Orientation",
    "key": "Musical key",
    
    # Spotify/YouTube-style audio features
    "acousticness": "How acoustic a track is (0.0-1.0)",
    "danceability": "How suitable for dancing (0.0-1.0)", 
    "energy": "Energy level (0.0-1.0)",
    "instrumentalness": "Instrumentalness score (0.0-1.0)",
    "liveness": "Detection of live performance (0.0-1.0)",
    "popularity": "Popularity score (0-100)",
    "speechiness": "Speechiness score (0.0-1.0)",
    "valence": "Musical positiveness (0.0-1.0)",
    "tempo": "Tempo in BPM",
    
    # Additional common tags
    "bpm": "Beats Per Minute",
    "mood": "General mood",
    "genre": "Genre classification",
    "artist": "Artist name",
    "title": "Track title",
    "album": "Album name",
    "tracknumber": "Track number",
    "date": "Release date",
    "comment": "Comment field",
    "composer": "Composer",
    "discnumber": "Disc number",
}

print("Popular Audio Feature Tags:")
print("=" * 50)
for tag, description in POPULAR_AUDIO_TAGS.items():
    print(f"{tag:15} - {description}")

print("\nExample usage in TaggingConfig:")
print("=" * 50)
print('TaggingConfig(')
print('    tags={')
print('        "artist": "{tag_artist}",')
print('        "title": "{tag_title}",')
print('        "album": "{tag_album}",')
print('        "genre": "{meta_genre}",')
print('    },')
print('    # Audio features:')
print('    extra={')
print('        "acousticness": "{meta_acousticness}",')
print('        "danceability": "{meta_danceability}",')
print('        "key": "{meta_key}",')
print('        "tcop": "{meta_tcop}",')
print('    }')
print(')')
